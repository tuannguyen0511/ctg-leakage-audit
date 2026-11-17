"""P21-T08b [ADAPT]: preregistered full repeat x seed grid, CPU-only, resumable.

Independent from and never writes T08 or T02-T06. Based on the EXACT locked
T08 source (SHA256 prefix 9dc58ea2). Only Director-authorized T08b changes:
(a) full repeat x seed grid; (b) segment-centering, log CWT and fold-train z;
(c) gradient clipping at 1.0; (d) fit/test probability diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

for env_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[env_key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import yaml
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

try:
    import pywt
except ImportError:
    pywt = None
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    torch = None
    nn = None
    F = None
    TensorDataset = None
    DataLoader = None

PREREG = '450c09ed1301a835baf004ff24ba4feeac97f845a425794ab642b79d4a228174'
T08_ORIGINAL_SCRIPT_SHA256 = '9dc58ea2a4482a278997d80ce7b63cf7fe7500098d1bbb24643a9e8e23d0b142'
T08_ORIGINAL_SUMMARY_SHA8 = '4b86a0c1'
T02_CTG_FEATURES_SHA256 = 'd81bd7924184a4479171fd83f1fd2ce1875843aba2987f52f70460aee02fabf6'
T02_CONFIG_SHA256 = '2c0b3d2e97011db6cbc5fe1fdcc8c6d93ea8ff77ecf2385ab244a512f6b4694d'
VALID_PROTOCOLS = ('S_img', 'L_img')
VALID_LABELS = ('y705', 'y715')
REPEATS = (0, 1, 2)
SEEDS = (0, 1, 2)
FOLDS = tuple(range(5))


def ensure(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def path_rel(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def manifest_sha_records(text):
    """Accept T02 sha8/path or sha256sum lines, ignoring comments and unrelated lines."""
    records = {}
    for line in text.splitlines():
        line = line.strip().lstrip('\ufeff')
        if not line or line.startswith('#'):
            continue
        m = re.fullmatch(r'([a-fA-F0-9]{8}|[a-fA-F0-9]{64})\s+\*?(.+?)\s*', line)
        if not m:
            # Earlier project manifests may use path,hash CSV lines.
            q=next(csv.reader([line])) if ',' in line else []
            if len(q)==2 and re.fullmatch(r'[0-9a-fA-F]{8}|[0-9a-fA-F]{64}',q[1].strip()):
                name,digest=q[0].strip(),q[1].strip().lower()
            else:
                continue
        else:
            digest, name = m.group(1).lower(), m.group(2).strip()
        name=name.replace('\\', '/')
        name = name.removeprefix('./')
        if name.lower().startswith('d:/tuan/ai_training/'):
            name = name[len('D:/Tuan/AI_training/'):]
        if name.startswith('P21_CTG_Acidemia/'):
            name = name[len('P21_CTG_Acidemia/'):]
        records[name] = digest
    return records


def verify_t02_locks(cfg):
    path = Path(cfg['input_manifest'])
    ensure(path.is_file(), f'Missing T02 frozen manifest: {path}')
    locks = manifest_sha_records(path.read_text(encoding='utf-8-sig'))
    root = Path(cfg['input_dir'])
    report = {}
    for name in ('labels.parquet', 'split.csv'):
        source = root / name
        ensure(source.is_file(), f'Missing T02 input: {source}')
        actual = sha256(source)
        candidates = [f'data_proc/ctu-uhb-ctgdb/1.0.0/{name}', name,
                      f'outputs/t02_inputs/{name}']
        found = [locks[x] for x in candidates if x in locks]
        ensure(found, f'MISSING_FROZEN_SHA for {name} in {path}; DO NOT self-freeze')
        ensure(all(actual.startswith(s) for s in found),
               f'FROZEN_SHA_MISMATCH {name}: actual={actual[:8]}, locked={found}')
        report[name] = actual
    return report


def fixed_config(cfg):
    expected = {
        'task': 'P21-T08b', 'run_name': 't08b_deepfhr',
        'prereg_version': '1.5', 'prereg_sha256': PREREG,
        'fs_hz': 4, 'stage1_minutes': 60, 'segment_minutes': 10,
        'segments_per_record': 6, 'max_missing_fraction': 0.50,
        'wavelet': 'morl', 'frequencies_hz': [0.003, 1.0],
        'scales': 64, 'image_size': 64, 'protocols': list(VALID_PROTOCOLS),
        'labels': list(VALID_LABELS), 'repeats': list(REPEATS),
        'seeds': list(SEEDS), 'folds': 5,
        'image_split_seed_base': 2026, 'validation_fraction': 0.2,
        'channels': [16, 32, 64, 64], 'kernel_size': 3, 'dropout': 0.3,
        'optimizer': 'Adam', 'learning_rate': 0.001, 'weight_decay': 0.0001,
        'batch_size': 32, 'max_epochs': 60, 'patience': 10, 'torch_threads': 4,
        'gradient_clip_norm': 1.0,
        'bootstrap_resamples': 10000, 'bootstrap_seed': 42,
        'control_permutation_seed': 42, 'control_protocol': 'L_img',
        'control_label': 'y705', 'control_repeat': 0, 'control_seed': 0,
        'control_bounds': [0.40, 0.60],
    }
    for k,v in expected.items():
        ensure(cfg.get(k) == v, f'Frozen T08b config changed: {k}; got={cfg.get(k)!r}')
    ensure(set(cfg) == set(expected) | {'raw_dir','input_dir','project_dir','input_manifest','t02_config'},
           'Config contains unexpected or missing keys')
    for name in ('raw_dir','input_dir','project_dir','input_manifest','t02_config'):
        ensure(str(cfg[name]).strip(), f'Missing path {name}')


def load_t02_cfg(cfg):
    path = Path(cfg['t02_config'])
    ensure(path.is_file(), f'T02 cleaning config missing: {path}')
    ensure(sha256(path)==T02_CONFIG_SHA256,
           'T02 cleaning YAML differs from original T02 release; request Director review')
    t02 = yaml.safe_load(path.read_text(encoding='utf-8'))
    for key, value in {'fs_hz':4,'fhr_min_bpm':50,'fhr_max_bpm':200,
                       'fhr_max_step_bpm':25}.items():
        ensure(t02.get(key) == value, f'T02 cleaning configuration inconsistent: {key}')
    return t02


def load_labels_split(cfg):
    root = Path(cfg['input_dir'])
    labels = pd.read_parquet(root/'labels.parquet')
    split = pd.read_csv(root/'split.csv', dtype={'record_id':str})
    ensure(len(labels)==552 and labels.record_id.astype(str).is_unique, 'T02 labels n/key invalid')
    labels = labels.assign(record_id=labels.record_id.astype(str)).set_index('record_id')
    ensure(labels.excluded.notna().all() and labels.excluded.astype(bool).sum()==34,
           'T02 expected 34 excluded records')
    keep = labels.loc[~labels.excluded.astype(bool)].copy()
    ensure(len(keep)==518,'T08b requires 518 eligible records')
    for label in VALID_LABELS:
        ensure(keep[label].isin((0,1)).all() and len(keep[label].unique())==2,
               f'Nonbinary/one-class label: {label}')
    ensure(list(split.columns)==['record_id','repeat','outer_fold','inner_fold'],
           'Unexpected T02 split.csv schema')
    ensure(set(split.record_id)==set(keep.index), 'T02 split/labels ID mismatch')
    ensure(set(split.repeat)==set(range(10)) and set(split.outer_fold)==set(range(5)),
           'Unexpected T02 split design')
    ensure(not split.duplicated(['record_id','repeat','outer_fold']).any(),
           'Duplicate T02 split records')
    for rep in REPEATS:
        all_test=[]
        for fold in FOLDS:
            s=split[(split.repeat==rep)&(split.outer_fold==fold)]
            ensure(len(s)==518 and set(s.record_id)==set(keep.index), 'Incomplete S fold rows')
            ensure(set(s.inner_fold).issubset(set(range(-1,5))), 'Invalid inner_fold')
            t=s.loc[s.inner_fold==-1,'record_id'].tolist()
            ensure(bool(t) and len(t)==len(set(t)), 'Empty/nonunique S test fold')
            all_test.extend(t)
        ensure(len(all_test)==518 and len(set(all_test))==518,
               f'S does not partition 518 records in repeat {rep}')
    return keep,split


def read_raw_signal(raw_dir, rid):
    # Reuse T01 WFDB read implementation used unchanged in T02.
    from common.io_wfdb import read_record
    record = read_record(Path(raw_dir), rid)
    return record


def first_stage_fhr(record, t02cfg, minutes=60):
    from P21_CTG_Acidemia.src.ctg_features import clean_signals
    ensure(int(record.fs)==4, 'Unexpected raw sampling rate')
    names = {str(k).upper(): i for i,k in enumerate(record.channel_names)}
    ensure('FHR' in names and 'UC' in names, 'Missing FHR/UC channels')
    fhr, uc = clean_signals(record.signal[:,names['FHR']],
                            record.signal[:,names['UC']],t02cfg)
    try:
        pos=float(record.clinical.get('Pos. II.st.'))
    except (ValueError,TypeError):
        pos=np.nan
    end=len(fhr) if not (np.isfinite(pos) and pos>=0) else int(pos)
    ensure(0<=end<=len(fhr), 'Invalid first-stage end index')
    width=int(minutes*60*record.fs)
    x=fhr[max(0,end-width):end]
    if len(x)<width:
        x=np.pad(x,(width-len(x),0),constant_values=np.nan)
    ensure(len(x)==width,'Unexpected 60-minute FHR length')
    return x


def scales_and_frequencies(cfg):
    ensure(pywt is not None, 'PyWavelets is required: python -m pip install PyWavelets')
    hi=float(cfg['frequencies_hz'][1]); lo=float(cfg['frequencies_hz'][0])
    freqs=np.geomspace(hi,lo,int(cfg['scales'])).astype(np.float64)
    scales=(float(pywt.central_frequency('morl')) *float(cfg['fs_hz'])/freqs)
    return scales,freqs


def cwt_image(fhr10, scales, cfg):
    ensure(pywt is not None and torch is not None, 'PyWavelets and CPU PyTorch required')
    signal=np.asarray(fhr10,dtype=np.float64)
    ensure(signal.ndim==1 and np.isfinite(signal).all(), 'CWT requires finite 1-D input')
    signal=signal-np.mean(signal,dtype=np.float64)
    coefficients, _=pywt.cwt(signal,scales,'morl',sampling_period=1.0/cfg['fs_hz'],method='fft')
    values=np.abs(coefficients).astype(np.float32)
    ensure(values.shape==(64,len(signal)), 'Incorrect raw CWT shape')
    with torch.no_grad():
        tens=torch.from_numpy(np.ascontiguousarray(values))[None,None]
        resized=F.interpolate(tens,size=(64,64),mode='bilinear',align_corners=False)[0,0].numpy()
    scaled=np.log1p(resized)
    ensure(scaled.shape==(64,64) and np.isfinite(scaled).all(), 'Nonfinite raw log1p CWT image')
    return scaled.astype(np.float32,copy=False)


def segment_fhr(fhr60,cfg):
    """Mask and reject before interpolation; interpolate entire 60-min window."""
    fhr60=np.asarray(fhr60,dtype=np.float64)
    fs=int(cfg['fs_hz']); n=int(cfg['segment_minutes']*60*fs)
    ensure(len(fhr60)==6*n, 'Expected exactly six 10-min segments')
    valid=np.isfinite(fhr60)
    if valid.any():
        filled=np.interp(np.arange(len(fhr60)),np.flatnonzero(valid),fhr60[valid])
    else:
        filled=np.zeros_like(fhr60)
    for j in range(6):
        a=j*n; b=(j+1)*n
        missing=float((~valid[a:b]).mean())
        keep=missing<=float(cfg['max_missing_fraction']) and valid[a:b].any()
        yield j, missing, keep, np.asarray(filled[a:b], dtype=np.float64)


def atomic_new(path,writer):
    """Windows-compatible fail-if-exists rename, no half-written target."""
    path=Path(path)
    ensure(not path.exists(), f'Refusing overwrite: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.t08b_pending_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            writer(f)
            f.flush();os.fsync(f.fileno())
        ensure(not path.exists(), f'File appeared during write: {path}')
        # Windows os.rename fails when destination already exists.
        os.rename(tmp,path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path,obj):
    encoded=json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')
    atomic_new(path,lambda f:f.write(encoded))


def atomic_parquet(path,df):
    atomic_new(path,lambda f:df.to_parquet(f,index=False))


def atomic_npz(path,**data):
    atomic_new(path,lambda f:np.savez_compressed(f,**data))


def raw_hashes(raw_dir,rid):
    paths=[Path(raw_dir)/(rid+s) for s in ('.hea','.dat')]
    ensure(all(p.is_file() for p in paths), f'Missing .hea/.dat for {rid}')
    return {p.suffix:sha256(p) for p in paths}


def image_cache_path(out,rid):
    ensure(re.fullmatch(r'[A-Za-z0-9._-]+',rid) is not None, f'Unsafe record ID: {rid}')
    return Path(out)/'image_cache'/(rid+'.npz')


def build_one_record(rid,labelrow,cfg,t02cfg,scales,out,scriptsha,configsha):
    path=image_cache_path(out,rid)
    rawsha=raw_hashes(cfg['raw_dir'],rid)
    if path.is_file():
        with np.load(path,allow_pickle=False) as z:
            ensure(set(z.files)=={'images','segments','missing_fraction','metadata'},
                   f'Bad cache schema for {rid}')
            images=z['images']; seg=z['segments']; missing=z['missing_fraction']
            info=json.loads(str(z['metadata'].item()))
        ensure(info=={'record_id':rid,'raw_hashes':rawsha,'script_sha256':scriptsha,
                     'config_sha256':configsha},f'Cache metadata conflict for {rid}')
        ensure(images.shape==(len(seg),64,64) and len(missing)==6,
               f'Cache shape conflict for {rid}')
        ensure(all(0<=i<6 for i in seg) and len(set(seg.tolist()))==len(seg),
               f'Cache segment IDs invalid for {rid}')
        ensure(np.isfinite(images).all() and np.isfinite(missing).all(),
               f'Cache NaN for {rid}')
        return images,seg,missing,True
    rec=read_raw_signal(cfg['raw_dir'],rid)
    fhr=first_stage_fhr(rec,t02cfg,cfg['stage1_minutes'])
    images=[]; segments=[];miss=[]
    for segid, frac, keep, data in segment_fhr(fhr,cfg):
        miss.append(frac)
        if keep:
            images.append(cwt_image(data,scales,cfg))
            segments.append(segid)
    ensure(images,'Record without images: '+rid)
    arr=np.stack(images).astype(np.float32)
    seg=np.asarray(segments,dtype=np.int8)
    missing=np.asarray(miss,dtype=np.float32)
    info={'record_id':rid,'raw_hashes':rawsha,'script_sha256':scriptsha,
          'config_sha256':configsha}
    atomic_npz(path,images=arr,segments=seg,missing_fraction=missing,
               metadata=np.asarray(json.dumps(info,sort_keys=True)))
    return arr,seg,missing,False


def prepare_images(labels,cfg,t02cfg,out,source):
    scales,_=scales_and_frequencies(cfg)
    images=[];meta=[];ids=[];recids=[]
    for i,(rid,row) in enumerate(labels.iterrows()):
        im,seg,missing,resumed=build_one_record(str(rid),row,cfg,t02cfg,scales,
                                                out,source['script_sha256'],source['config_sha256'])
        for j,frame in zip(seg,im):
            images.append(frame)
            ids.append(f'{rid}_s{int(j)}')
            recids.append(rid)
        meta.append({'record_id':rid,'n_images':int(len(seg)),
                     'n_dropped':int(6-len(seg)),
                     'missing_fraction_mean':float(np.mean(missing)),
                     'retained_segment_ids':','.join(map(str,seg.tolist()))})
        if (i+1)%25==0 or i+1==len(labels):
            print(f'T08B_IMAGE_CACHE_PROGRESS {i+1}/{len(labels)}; resumed={resumed}',flush=True)
    array=np.stack(images).astype(np.float32)[:,None,:,:]
    frame=pd.DataFrame(meta)
    ensure(len(frame)==518 and frame.record_id.is_unique, 'Image metadata record count invalid')
    ensure(int(frame.n_images.sum())==len(array) and (frame.n_images>=1).all(),
           'Images/per-record metadata inconsistent')
    ensure(int((frame.n_images+frame.n_dropped).sum())==518*6,
           'Unexpected number of potential segments')
    ensure(len(set(ids))==len(ids) and np.isfinite(array).all(), 'Duplicate/nonfinite images')
    if (Path(out)/'images_meta.parquet').is_file():
        old=pd.read_parquet(Path(out)/'images_meta.parquet')
        ensure(old.equals(frame),'images_meta.parquet mismatch on resume')
    else:
        atomic_parquet(Path(out)/'images_meta.parquet',frame)
    return array,np.asarray(ids,dtype=str),np.asarray(recids,dtype=str),frame


def strict_folds(split,repeat,record_ids,image_records):
    folds=[]
    for fold in FOLDS:
        s=split[(split.repeat==repeat)&(split.outer_fold==fold)]
        test_rids=set(s.loc[s.inner_fold==-1,'record_id'])
        mask=np.isin(image_records,list(test_rids))
        test=np.flatnonzero(mask); train=np.flatnonzero(~mask)
        ensure(len(test)>0 and len(train)>0,'Empty S image fold')
        ensure(not(set(image_records[train]) & set(image_records[test])),
               'S_img record overlap detected')
        folds.append((train,test))
    assert_full_coverage(folds,len(image_records))
    return folds


def leaky_folds(image_y,repeat,image_records):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=2026+repeat)
    folds=list(skf.split(np.zeros(len(image_y)),image_y))
    assert_full_coverage(folds,len(image_records))
    overlap=[len(set(image_records[tr])&set(image_records[te])) for tr,te in folds]
    ensure(all(i>0 for i in overlap), 'L_img expected record overlap but none found')
    return folds


def assert_full_coverage(folds,n):
    counts=np.zeros(n,dtype=np.int8)
    for tr,te in folds:
        tr=np.asarray(tr);te=np.asarray(te)
        ensure(len(tr)>0 and len(te)>0 and not np.intersect1d(tr,te).size,
               'Fold train/test image overlap')
        ensure(len(np.unique(tr))==len(tr) and len(np.unique(te))==len(te),
               'Duplicate image index in fold')
        counts[te]+=1
    ensure(np.all(counts==1),'Images not tested exactly once across 5 folds')


def validation_indices(tr,te,records,y,protocol,repeat,fold,seed=0):
    tr=np.asarray(tr,dtype=np.int64);te=np.asarray(te,dtype=np.int64)
    ensure(not np.intersect1d(tr,te).size,'Outer-test image in training')
    rng=1000*repeat+10*fold+seed
    splitter=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=rng)
    if protocol=='S_img':
        unique_ids=np.unique(records[tr]); mapping={rid:int(y[tr][np.flatnonzero(records[tr]==rid)[0]]) for rid in unique_ids}
        ys=np.asarray([mapping[r] for r in unique_ids],dtype=np.int64)
        ia,ib=next(splitter.split(np.zeros(len(unique_ids)),ys))
        fit=tr[np.isin(records[tr],unique_ids[ia])]
        val=tr[np.isin(records[tr],unique_ids[ib])]
        ensure(not(set(records[fit])&set(records[val])), 'S_img validation record overlap')
        ensure(not(set(records[fit])&set(records[te])) and not(set(records[val])&set(records[te])),
               'S_img fit/val/test record overlap')
    else:
        ia,ib=next(splitter.split(np.zeros(len(tr)),y[tr]))
        fit=tr[ia];val=tr[ib]
    ensure(not np.intersect1d(fit,val).size and not np.intersect1d(fit,te).size
           and not np.intersect1d(val,te).size,'Image fit/val/test overlap')
    ensure(len(np.unique(y[fit]))==2 and len(np.unique(y[val]))==2,
           'Validation split contains one class')
    return fit,val


if nn is not None:
    class CWT2DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            blocks=[];c=1
            for out_c in (16,32,64,64):
                blocks += [nn.Conv2d(c,out_c,kernel_size=3,padding=1),
                           nn.BatchNorm2d(out_c),nn.ReLU(),nn.MaxPool2d(2)]
                c=out_c
            self.encoder=nn.Sequential(*blocks)
            self.pool=nn.AdaptiveAvgPool2d(1)
            self.dropout=nn.Dropout(0.3)
            self.out=nn.Linear(64,1)
        def forward(self,x):
            x=self.pool(self.encoder(x)).flatten(1)
            return self.out(self.dropout(x)).flatten()


def cpu_torch():
    ensure(torch is not None,'PyTorch is required: install torch CPU-only')
    ensure(pywt is not None,'PyWavelets is required: python -m pip install PyWavelets')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    ensure(torch.get_num_threads()==4,'PyTorch thread count != 4')
    return True


def seed_all(seed):
    cpu_torch()
    np.random.seed(int(seed));random.seed(int(seed));torch.manual_seed(int(seed))


def probs(model,images,batch=32):
    model.eval();outputs=[]
    with torch.no_grad():
        for start in range(0,len(images),batch):
            x=torch.from_numpy(np.ascontiguousarray(images[start:start+batch]))
            outputs.append(torch.sigmoid(model(x)).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def fold_train_zscore(images, outer_train, fit, val, test):
    """Z-score raw cached CWT with outer-train images only (no test statistics).

    The outer-train fold is the same pool used to form fit and validation.
    One global mean and population SD covers every pixel of every train image.
    Return only the fit/validation/test arrays to limit memory use.
    """
    outer_train=np.asarray(outer_train,dtype=np.int64)
    fit=np.asarray(fit,dtype=np.int64)
    val=np.asarray(val,dtype=np.int64)
    test=np.asarray(test,dtype=np.int64)
    ensure(len(outer_train)>0 and len(fit)>0 and len(val)>0 and len(test)>0,
           'Empty split in normalization')
    ensure(not np.intersect1d(outer_train,test).size,'TEST LEAKAGE in z-score statistics')
    ensure(set(fit).issubset(set(outer_train)) and set(val).issubset(set(outer_train)),
           'Fit/val outside fold train')
    source=np.asarray(images[outer_train],dtype=np.float64)
    mean=float(source.mean(dtype=np.float64))
    std=float(source.std(dtype=np.float64,ddof=0))
    ensure(np.isfinite(mean) and np.isfinite(std) and std>0,
           'Invalid fold-train normalization statistics')
    def transform(indices):
        z=(np.asarray(images[indices],dtype=np.float64)-mean)/std
        z=np.ascontiguousarray(z.astype(np.float32))
        ensure(np.isfinite(z).all(),'Nonfinite normalized fold image')
        return z
    return transform(fit),transform(val),transform(test),{'train_mean':mean,'train_std':std,
            'n_normalization_train_images':int(len(outer_train))}


def train_one(images,y,outer_train,fit,val,test,seed,cfg):
    t0=time.perf_counter()
    seed_all(seed)
    xfit,xval,xtest,zinfo=fold_train_zscore(images,outer_train,fit,val,test)
    yfit=y[fit].astype(np.float32);yval=y[val].astype(np.int64)
    pos=int(yfit.sum()); neg=len(yfit)-pos
    ensure(pos>0 and neg>0,'BCE loss needs two classes')
    model=CWT2DCNN().cpu()
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],dtype=torch.float32))
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    gen=torch.Generator().manual_seed(int(seed))
    loader=DataLoader(TensorDataset(torch.from_numpy(xfit),torch.from_numpy(yfit)),
                      batch_size=cfg['batch_size'],shuffle=True,generator=gen,num_workers=0)
    best_auc=-np.inf;best_epoch=0;best_state=None;wait=0;last=0
    for epoch in range(1,cfg['max_epochs']+1):
        model.train()
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True)
            l=lossfn(model(xb),yb)
            ensure(torch.isfinite(l).item(),'Nonfinite training loss')
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=cfg['gradient_clip_norm'])
            opt.step()
        vp=probs(model,xval,cfg['batch_size'])
        score=float(roc_auc_score(yval,vp))
        ensure(np.isfinite(score),'Nonfinite validation AUROC')
        last=epoch
        if score>best_auc:
            best_auc=score;best_epoch=epoch
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            wait=0
        else:
            wait+=1
            if wait>=cfg['patience']:
                break
    ensure(best_state is not None, 'Missing best validation checkpoint')
    model.load_state_dict(best_state)
    train_p=probs(model,xfit,cfg['batch_size'])
    train_auc=float(roc_auc_score(yfit.astype(np.int64),train_p))
    ensure(np.isfinite(train_auc),'Nonfinite training AUROC at best checkpoint')
    p=probs(model,xtest,cfg['batch_size'])
    ensure(np.isfinite(p).all() and np.all((p>=0)&(p<=1)),
           'Nonfinite or invalid prediction')
    return p,{'runtime_s':float(time.perf_counter()-t0),
              'epoch_stop':last,'best_epoch':best_epoch,'best_val_auroc':best_auc,
              'n_train_images':len(fit),'n_validation_images':len(val),
              'n_positive_train_images':pos,'n_negative_train_images':neg,
              'train_auroc':train_auc,'test_probability_sd':float(np.std(p,ddof=0)),
              'fold_normalization':zinfo}


def all_units():
    for label in VALID_LABELS:
        for repeat in REPEATS:
            for seed in SEEDS:
                for protocol in VALID_PROTOCOLS:
                    for fold in FOLDS:
                        yield (protocol,label,repeat,seed,fold)
    for fold in FOLDS:
        yield ('NC_L_img','y705',0,0,fold)


def unit_file(out,key):
    protocol,label,repeat,seed,fold=key
    return Path(out)/'units'/f'{protocol}_{label}_s{seed}_r{repeat}_f{fold}.npz'


def test_split_for_unit(key,foldcache,permuted_y):
    protocol,label,repeat,seed,fold=key
    if protocol=='NC_L_img':
        return foldcache[('NC_L_img','y705',0)][fold],permuted_y
    return foldcache[(protocol,label,repeat)][fold],None

def make_permuted_y(labels,records,seed=42):
    ids=np.asarray(labels.index,dtype=str)
    y=labels.loc[ids,'y705'].to_numpy(dtype=np.int64)
    perm=np.random.default_rng(seed).permutation(y)
    lookup=dict(zip(ids,perm))
    return np.asarray([lookup[r] for r in records],dtype=np.int64)


def make_fold_cache(labels,split,records):
    cache={};permuted=make_permuted_y(labels,records)
    for label in VALID_LABELS:
        y=labels.loc[records,label].to_numpy(dtype=np.int64)
        for repeat in REPEATS:
            cache[('S_img',label,repeat)]=strict_folds(split,repeat,labels.index,records)
            cache[('L_img',label,repeat)]=leaky_folds(y,repeat,records)
    cache[('NC_L_img','y705',0)]=leaky_folds(permuted,0,records)
    return cache,permuted


def identity(cfg,config_path):
    feature_path=Path(cfg['project_dir'])/'src/ctg_features.py'
    ensure(sha256(feature_path)==T02_CTG_FEATURES_SHA256,
           'T02 ctg_features.py SHA mismatch; do not modify or proceed')
    sources={
        'script_sha256':sha256(__file__),
        'config_sha256':sha256(config_path),
        't02_cfg_sha256':sha256(cfg['t02_config']),
        'ctg_features_sha256':sha256(feature_path),
    }
    return sources


def assert_lock(out,cfg,source,inputhashes,resume):
    lockpath=Path(out)/'run_lock.json'
    expected={'task':'P21-T08b','prereg_v1.5_sha256':PREREG,
              'sources':source,'frozen_inputs_sha256':inputhashes,
              'python':sys.version.split()[0],
              'torch':str(torch.__version__), 'pywt':str(pywt.__version__),
              'numpy':str(np.__version__), 'pandas':str(pd.__version__),
              'sklearn':str(sklearn.__version__)}
    if Path(out).exists():
        ensure(resume, 'STOP: outputs/t08b_deepfhr exists; specify --resume to continue')
        ensure(lockpath.is_file(), 'STOP: existing output has no run_lock.json')
        existing=json.loads(lockpath.read_text(encoding='utf-8'))
        ensure(existing==expected,'STOP: existing run lock mismatch; no overwrite')
    else:
        ensure(not resume,'--resume requires an existing T08b output directory')
        Path(out).mkdir(parents=True,exist_ok=False)
        atomic_json(lockpath,expected)
    return expected


def _unit_meta(key,source,inputhashes,test_img_ids,train,val,records):
    protocol,label,repeat,seed,fold=key
    return {'task':'P21-T08b','prereg_sha256':PREREG,
            'protocol':protocol,'label':label,'repeat':int(repeat),'fold':int(fold),
            'seed':int(seed),'script_sha256':source['script_sha256'],
            'config_sha256':source['config_sha256'],
            'labels_sha256':inputhashes['labels.parquet'],
            'split_sha256':inputhashes['split.csv'],
            'test_image_ids_sha256':hashlib.sha256('\n'.join(test_img_ids).encode('utf-8')).hexdigest(),
            'n_fit':int(len(train)),'n_val':int(len(val)),
            'fit_indices_sha256':hashlib.sha256(np.asarray(train,dtype='<i8').tobytes()).hexdigest(),
            'val_indices_sha256':hashlib.sha256(np.asarray(val,dtype='<i8').tobytes()).hexdigest(),
            'n_train_test_record_overlap':None,
            'no_outer_test_image_leakage':True}


def read_unit(path,meta,imageids,test,y):
    with np.load(path,allow_pickle=False) as z:
        ensure(set(z.files)=={'image_id','record_id','y','p','meta_json','fit_info'},f'Bad unit schema {path}')
        ids=z['image_id'].astype(str);rec=z['record_id'].astype(str)
        yt=z['y'].astype(np.int64);p=z['p'].astype(np.float64)
        got=json.loads(str(z['meta_json'].item()))
        fitinfo=json.loads(str(z['fit_info'].item()))
    ensure(all(got.get(k)==v for k,v in meta.items()),'Resumed unit identity differs: '+str(path))
    ensure(np.array_equal(ids,imageids[test]) and np.array_equal(yt,y[test])
           and len(p)==len(test) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),
           'Resumed unit predictions mismatch: '+str(path))
    for required in ('train_auroc','test_probability_sd','fold_normalization','best_epoch','runtime_s'):
        ensure(required in fitinfo,f'Missing diagnostic {required} in {path}')
    ensure(np.isfinite(float(fitinfo['train_auroc'])) and
           np.isfinite(float(fitinfo['test_probability_sd'])),f'Nonfinite diagnostics in {path}')
    ensure(np.isclose(float(fitinfo['test_probability_sd']),float(np.std(p,ddof=0)),atol=1e-12),
           f'Test SD does not match stored probabilities: {path}')
    return p,fitinfo


def run_unit(key,out,images,imageids,records,y,folds,source,inputhashes,cfg):
    protocol,label,repeat,seed,fold=key
    tr,te=folds[fold];tr=np.asarray(tr);te=np.asarray(te)
    if protocol=='S_img':
        ensure(not(set(records[tr]) & set(records[te])), 'S_img train/test subject overlap')
    val_protocol='S_img' if protocol=='S_img' else 'L_img'
    fit,val=validation_indices(tr,te,records,y,val_protocol,repeat,fold,seed=seed)
    meta=_unit_meta(key,source,inputhashes,imageids[te].tolist(),fit,val,records)
    meta['n_train_test_record_overlap']=int(len(set(records[tr])&set(records[te])))
    if protocol=='S_img':
        ensure(meta['n_train_test_record_overlap']==0,'Strict record overlap')
    else:
        ensure(meta['n_train_test_record_overlap']>0,'Expected leaky record overlap')
    path=unit_file(out,key)
    if path.is_file():
        p,fitinfo=read_unit(path,meta,imageids,te,y)
        return p,fitinfo,True
    p,fitinfo=train_one(images,y,tr,fit,val,te,seed,cfg)
    atomic_npz(path,image_id=imageids[te],record_id=records[te],y=y[te].astype(np.int8),
               p=p,meta_json=np.asarray(json.dumps(meta,sort_keys=True)),
               fit_info=np.asarray(json.dumps(fitinfo,sort_keys=True)))
    read_unit(path,meta,imageids,te,y)
    return p,fitinfo,False


def one_repeat_seed_oof(out,protocol,label,repeat,seed,fold_cache,labels,records,imageids,permuted,source,inputhashes,cfg):
    actual='L_img' if protocol=='NC_L_img' else protocol
    y=permuted if protocol=='NC_L_img' else labels.loc[records,label].to_numpy(dtype=np.int64)
    foldset=fold_cache[(protocol,label,repeat)]
    parts=[]
    for fold in FOLDS:
        key=(protocol,label,repeat,seed,fold)
        tr,te=foldset[fold]
        fit,val=validation_indices(np.asarray(tr),np.asarray(te),records,y,actual,repeat,fold,seed=seed)
        meta=_unit_meta(key,source,inputhashes,imageids[te].tolist(),fit,val,records)
        meta['n_train_test_record_overlap']=int(len(set(records[tr])&set(records[te])))
        p,_=read_unit(unit_file(out,key),meta,imageids,np.asarray(te),y)
        parts.append(pd.DataFrame({'image_id':imageids[te],'record_id':records[te],
                                   'y':y[te],'p':p,'protocol':protocol,
                                   'label':label,'repeat':repeat,'seed':seed,'fold':fold}))
    frame=pd.concat(parts,ignore_index=True)
    ensure(len(frame)==len(imageids) and frame.image_id.is_unique
           and set(frame.image_id)==set(imageids) and np.isfinite(frame.p).all(),
           'Incomplete OOF image predictions')
    return frame

def record_reduce(frame):
    # Mean probability of all OOF images belonging to one record.
    g=frame.groupby('record_id',as_index=False,sort=False).agg(y=('y','first'),p=('p','mean'))
    ensure(g.record_id.is_unique and g.y.isin((0,1)).all(),'Record aggregation invalid')
    ensure(frame.groupby('record_id').y.nunique().eq(1).all(), 'Record y inconsistent')
    return g


def roc_weighted(y,p,w):
    """Exactly weighted pairwise AUROC; preserves tied-score half credit."""
    y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=float);w=np.asarray(w,dtype=np.float64)
    order=np.argsort(p,kind='mergesort')
    yy=y[order];pp=p[order];ww=w[order]
    starts=np.r_[0,np.flatnonzero(pp[1:]!=pp[:-1])+1]
    gp=np.add.reduceat(ww*yy,starts)
    gn=np.add.reduceat(ww*(1-yy),starts)
    positives=gp.sum();negatives=gn.sum()
    if positives<=0 or negatives<=0:
        return np.nan
    lower_neg=np.cumsum(gn)-gn
    return float(np.dot(gp,lower_neg+0.5*gn)/(positives*negatives))


def ci(values):
    finite=np.asarray(values,dtype=float)
    finite=finite[np.isfinite(finite)]
    ensure(len(finite)>0,'No valid bootstrap replicates')
    return [float(x) for x in np.percentile(finite,[2.5,97.5])]


def bootstrap_pair(s_frame,l_frame,n_resamples=10000,seed=42):
    """Paired record-cluster bootstrap; resample records not images.

    Returns point scores and 10k replicate arrays, used both for repeat CIs
    and mean-of-repeats CIs with shared deterministic resamples.
    """
    left=s_frame.set_index('image_id',drop=False).sort_index()
    right=l_frame.set_index('image_id',drop=False).sort_index()
    ensure(np.array_equal(left.index.to_numpy(),right.index.to_numpy()),'S/L images not aligned')
    ensure(np.array_equal(left.record_id.to_numpy(),right.record_id.to_numpy())
           and np.array_equal(left.y.to_numpy(),right.y.to_numpy()),'S/L y or records differ')
    image_rec=left.record_id.to_numpy(dtype=str);image_y=left.y.to_numpy(dtype=np.int8)
    ids=np.unique(image_rec)
    id_index={rid:j for j,rid in enumerate(ids)}
    rec_ix=np.asarray([id_index[r] for r in image_rec],dtype=np.int64)
    s_prob=left.p.to_numpy(dtype=float);l_prob=right.p.to_numpy(dtype=float)
    rec_y=np.asarray([image_y[np.flatnonzero(rec_ix==i)[0]] for i in range(len(ids))],dtype=np.int8)
    num_rec=len(ids)
    rec_s=np.bincount(rec_ix,weights=s_prob,minlength=num_rec)/np.bincount(rec_ix,minlength=num_rec)
    rec_l=np.bincount(rec_ix,weights=l_prob,minlength=num_rec)/np.bincount(rec_ix,minlength=num_rec)
    ensure(np.isfinite(rec_s).all() and np.isfinite(rec_l).all(), 'Nonfinite record probabilities')
    point={'S_image_auroc':float(roc_auc_score(image_y,s_prob)),
           'L_image_auroc':float(roc_auc_score(image_y,l_prob)),
           'S_record_auroc':float(roc_auc_score(rec_y,rec_s)),
           'L_record_auroc':float(roc_auc_score(rec_y,rec_l))}
    point['delta_image']=point['L_image_auroc']-point['S_image_auroc']
    point['delta_record']=point['L_record_auroc']-point['S_record_auroc']
    keys=list(point)
    reps=np.full((n_resamples,len(keys)),np.nan,dtype=np.float64)
    rng=np.random.default_rng(seed)
    # This sampling is deterministic and identical for each repeat (same sorted record IDs).
    for i in range(n_resamples):
        sampled=rng.integers(0,num_rec,num_rec)
        counts=np.bincount(sampled,minlength=num_rec)
        if not np.isin((0,1),np.unique(rec_y[sampled])).all():
            continue
        imweights=counts[rec_ix]
        a=roc_weighted(image_y,s_prob,imweights)
        b=roc_weighted(image_y,l_prob,imweights)
        c=roc_weighted(rec_y,rec_s,counts)
        d=roc_weighted(rec_y,rec_l,counts)
        reps[i]=[a,b,c,d,b-a,d-c]
    ensure(np.isfinite(reps).all(),'Bootstrap produced NaN; label prevalence insufficient')
    return {'point':point,
            'ci95':{key:ci(reps[:,j]) for j,key in enumerate(keys)},
            'bootstrap_requested':int(n_resamples), 'bootstrap_valid':int(len(reps)),
            'bootstrap_unit':'record','bootstrap_seed':seed},reps


def collect_assay_diagnostics(out):
    vals=[];near=[];main=[]
    for key in all_units():
        path=unit_file(out,key)
        ensure(path.is_file(),f'Missing completed unit {path}')
        with np.load(path,allow_pickle=False) as z:
            fi=json.loads(str(z['fit_info'].item()))
            p=z['p'].astype(float)
        ensure(np.isfinite(p).all(),'Nonfinite test prediction in diagnostics')
        score=float(fi['train_auroc'])
        sd=float(fi['test_probability_sd'])
        ensure(np.isfinite(score) and np.isfinite(sd),'Nonfinite diagnostic')
        ensure(np.isclose(sd,float(p.std(ddof=0)),atol=1e-12),'SD/p mismatch')
        if key[0]=='NC_L_img':
            continue
        vals.append(score)
        if key[0]=='S_img' and key[1]=='y705':
            main.append(score)
        if sd<1e-3:
            near.append({'unit':path.name,'protocol':key[0],'label':key[1],
                         'repeat':key[2],'seed':key[3],'fold':key[4],
                         'test_probability_sd':sd})
    ensure(len(vals)==180 and len(main)==45,'Incorrect assay diagnostic count')
    frac_high=float(np.mean(np.asarray(main)>0.75))
    frac_near=float(len(near)/len(vals))
    sensitivity_pass=frac_high>=0.8 and frac_near<=0.1
    return {'median_train_auroc':float(np.median(vals)),
            'train_auroc_iqr':[float(x) for x in np.percentile(vals,[25,75])],
            'n_main_S_img_y705':int(len(main)),
            'S_img_y705_train_auroc_gt_0_75':int(np.sum(np.asarray(main)>0.75)),
            'S_img_y705_fraction_above_0_75':frac_high,
            'S_img_y705_fraction_threshold':0.80,
            'n_main_units':int(len(vals)),
            'near_constant_probability_count':int(len(near)),
            'near_constant_probability_fraction':frac_near,
            'near_constant_probability_sd_threshold':0.001,
            'near_constant_probability_fraction_max':0.10,
            'near_constant_units':near,
            'pass':bool(sensitivity_pass),
            'verdict':'assay sensitivity PASS' if sensitivity_pass else 'assay sensitivity FAIL'}


def t08_reference(cfg):
    base=Path(cfg['project_dir'])/'outputs/t08_deepfhr/summary.json'
    source=Path(cfg['project_dir'])/'src/t08_deepfhr.py'
    ensure(source.is_file() and sha256(source)==T08_ORIGINAL_SCRIPT_SHA256,
           'Original T08 script does not match locked SHA; never edit T08')
    ensure(base.is_file() and sha256(base).startswith(T08_ORIGINAL_SUMMARY_SHA8),
           'Original T08 summary not found or SHA8 mismatch; do not replace')
    orig=json.loads(base.read_text(encoding='utf-8'))
    ensure(orig['prereg_v1.4_sha256']=='85161286492355667fefb0d55d1643cedc5b2736d8234cac1914023c05744c58',
           'Original T08 prereg mismatch')
    return orig


def summarize(out,cfg,labels,records,imageids,fold_cache,permuted,source,inputhashes,image_meta,
              prep_runtime_s,training_runtime_s):
    summaries={};oof={'S_img':[],'L_img':[]}
    for label in VALID_LABELS:
        summaries[label]={'by_repeat_seed':{},'by_seed':{},'by_repeat':{},'aggregate':{}}
        replicates=[]
        for rep in REPEATS:
            for seed in SEEDS:
                s=one_repeat_seed_oof(out,'S_img',label,rep,seed,fold_cache,labels,records,
                                 imageids,permuted,source,inputhashes,cfg)
                l=one_repeat_seed_oof(out,'L_img',label,rep,seed,fold_cache,labels,records,
                                 imageids,permuted,source,inputhashes,cfg)
                oof['S_img'].append(s);oof['L_img'].append(l)
                result,boot=bootstrap_pair(s,l,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
                summaries[label]['by_repeat_seed'][f'r{rep}_s{seed}']=result
                replicates.append((rep,seed,result['point'],boot))
        keys=list(replicates[0][2])
        def aggregate(items):
            out={}
            for j,k in enumerate(keys):
                estimates=np.asarray([item[2][k] for item in items],dtype=float)
                bootmean=np.mean(np.stack([item[3][:,j] for item in items]),axis=0)
                out[k]={'mean':float(estimates.mean()),
                        'sd':float(estimates.std(ddof=1)),
                        'ci95_cluster_record':ci(bootmean)}
            return out
        for seed in SEEDS:
            summaries[label]['by_seed'][str(seed)]=aggregate([v for v in replicates if v[1]==seed])
        for rep in REPEATS:
            summaries[label]['by_repeat'][str(rep)]=aggregate([v for v in replicates if v[0]==rep])
        summaries[label]['aggregate']=aggregate(replicates)
    control=one_repeat_seed_oof(out,'NC_L_img','y705',0,0,fold_cache,labels,
                           records,imageids,permuted,source,inputhashes,cfg)
    ncauc=float(roc_auc_score(control.y,control.p))
    in_bounds=(cfg['control_bounds'][0]<=ncauc<=cfg['control_bounds'][1])
    controls={'protocol':'L_img','label':'y705','repeat':0,'seed':0,
              'permutation_level':'record','image_auroc':ncauc,
              'acceptance_interval':cfg['control_bounds'],'pass':bool(in_bounds)}
    for protocol in VALID_PROTOCOLS:
        frame=pd.concat(oof[protocol],ignore_index=True)
        ensure(len(frame)==len(imageids)*len(VALID_LABELS)*len(REPEATS)*len(SEEDS),
               'OOF unexpected number of rows')
        ensure(not frame.duplicated(['label','repeat','seed','image_id']).any(),
               'OOF duplicate image in repeat/seed/label')
        ensure(np.isfinite(frame.p).all() and frame.y.isin((0,1)).all(),
               'OOF NaN or nonbinary y')
        oof_path=Path(out)/f'oof_{protocol}.parquet'
        if oof_path.is_file():
            old=pd.read_parquet(oof_path)
            ensure(old.equals(frame),f'Existing OOF inconsistent: {oof_path}')
        else:
            atomic_parquet(oof_path,frame)
    assay=collect_assay_diagnostics(out)
    totals={'eligible_records':int(len(labels)),'images_total':int(image_meta.n_images.sum()),
            'images_per_record_median':float(image_meta.n_images.median()),
            'images_dropped':int(image_meta.n_dropped.sum()),
            'images_candidate':int(6*len(labels)),
            'records_with_images':int((image_meta.n_images>0).sum())}
    previous=t08_reference(cfg)
    compare={}
    for label in VALID_LABELS:
        compare[label]={}
        for k in summaries[label]['aggregate']:
            t08_old=float(previous['metrics'][label]['aggregate'][k]['mean'])
            new=float(summaries[label]['aggregate'][k]['mean'])
            compare[label][k]={'T08_v1_4_mean':t08_old,'T08b_v1_5_mean':new,
                              'change_new_minus_old':new-t08_old}
    status=('PASS' if in_bounds and assay['pass'] else
           'FAIL_NEGATIVE_CONTROL' if not in_bounds else 'ASSAY_SENSITIVITY_FAIL')
    output={'task':'P21-T08b','tag':'ADAPT_not_exact_replication',
            'status':status,'prereg_v1.5_sha256':PREREG,'source_identity':source,
            'frozen_inputs_sha256':inputhashes,'images':totals,'metrics':summaries,
            'negative_control':controls,'assay_sensitivity':assay,
            'comparison_with_unmodified_T08':compare,
            'T08_original_script_sha256':T08_ORIGINAL_SCRIPT_SHA256,
            'T08_original_summary_sha8':T08_ORIGINAL_SUMMARY_SHA8,
            'timing':{'image_prepare_runtime_current_invocation_s':float(prep_runtime_s),
                      'sum_all_unit_training_runtime_s':float(training_runtime_s)},
            'description':'Record-cluster percentile bootstrap 10000; leakage estimates descriptive, excluded from Holm H2.'}
    atomic_json(Path(out)/'summary.json',output)
    return output

def verify_summary_files(out, labels, n_images):
    for protocol in VALID_PROTOCOLS:
        file=Path(out)/f'oof_{protocol}.parquet'
        ensure(file.is_file(), f'Missing {file}')
        df=pd.read_parquet(file)
        ensure(len(df)==n_images*len(VALID_LABELS)*len(REPEATS)*len(SEEDS),
               'Unexpected number of OOF images')
        ensure(not df.duplicated(['label','repeat','seed','image_id']).any(),
               'Duplicate OOF row')
        ensure(np.isfinite(df.p.to_numpy()).all() and df.y.isin((0,1)).all(),
               'OOF NaN')



def make_output_manifest(out):
    """Freeze generated artifacts at completion, without self-freezing source inputs."""
    out=Path(out)
    manifest=out/'OUTPUT_SHA256.txt'
    lines=[]
    for file in sorted(out.rglob('*')):
        if file.is_file() and file.name!='OUTPUT_SHA256.txt' and not file.name.startswith('.t08b_pending_'):
            lines.append(f'{sha256(file)}  {file.relative_to(out).as_posix()}')
    content='\n'.join(lines)+'\n'
    if manifest.exists():
        ensure(manifest.read_text(encoding='utf-8')==content,
               'Existing output SHA256 manifest mismatch; do not overwrite')
    else:
        atomic_new(manifest,lambda f:f.write(content.encode('utf-8')))


def print_report(out,summary,total_s):
    src=summary['images']
    control=summary['negative_control']
    assay=summary['assay_sensitivity']
    print('=== REPORT P21-T08b ===')
    print(f"status={summary['status']}; mode=ADAPT; PC-A CPU only; pretrained=none")
    print(f'PREREG_v1.5_SHA256={PREREG}')
    print('T02_FROZEN_SHA_PASS=labels.parquet,split.csv; T08_original_preserved=PASS')
    print(f"images_total={src['images_total']}; median_per_record={src['images_per_record_median']}; dropped={src['images_dropped']}; eligible_records=518")
    print(f'unit_count={len(tuple(all_units()))}; units_complete={len(list((Path(out)/"units").glob("*.npz")))}')
    for lab in VALID_LABELS:
        by=summary['metrics'][lab]['by_repeat_seed']
        for rep in REPEATS:
            for seed in SEEDS:
                item=by[f'r{rep}_s{seed}'];q=item['point'];c=item['ci95']
                print(f"{lab}_r{rep}_s{seed}: S_img={q['S_image_auroc']:.4f}[{c['S_image_auroc'][0]:.4f},{c['S_image_auroc'][1]:.4f}]; "
                      f"L_img={q['L_image_auroc']:.4f}[{c['L_image_auroc'][0]:.4f},{c['L_image_auroc'][1]:.4f}]; "
                      f"S_record={q['S_record_auroc']:.4f}[{c['S_record_auroc'][0]:.4f},{c['S_record_auroc'][1]:.4f}]; "
                      f"L_record={q['L_record_auroc']:.4f}[{c['L_record_auroc'][0]:.4f},{c['L_record_auroc'][1]:.4f}]; "
                      f"delta_img={q['delta_image']:+.4f}[{c['delta_image'][0]:+.4f},{c['delta_image'][1]:+.4f}]; "
                      f"delta_record={q['delta_record']:+.4f}[{c['delta_record'][0]:+.4f},{c['delta_record'][1]:+.4f}]")
        print(f'{lab}_aggregate_mean_sd_CI={json.dumps(summary["metrics"][lab]["aggregate"],sort_keys=True)}')
        print(f'{lab}_by_seed={json.dumps(summary["metrics"][lab]["by_seed"],sort_keys=True)}')
        print(f'{lab}_by_repeat={json.dumps(summary["metrics"][lab]["by_repeat"],sort_keys=True)}')
    print(f"ASSAY_SENSITIVITY={assay['verdict']}; S_img_y705_train_auc_above_0.75={assay['S_img_y705_train_auroc_gt_0_75']}/{assay['n_main_S_img_y705']}; "
          f"test_near_constant={assay['near_constant_probability_count']}/{assay['n_main_units']}; "
          f"train_auc_median={assay['median_train_auroc']:.4f}; IQR={assay['train_auroc_iqr']}")
    print(f"NEGATIVE_CONTROL_L_img_y705_r0_s0_AUROC={control['image_auroc']:.4f}; bounds=[0.40,0.60]; pass={control['pass']}")
    print('T08_COMPARISON_NEW_MINUS_OLD='+json.dumps(summary['comparison_with_unmodified_T08'],sort_keys=True))
    print(f'runtime_s={total_s:.2f}; script_sha8={sha256(__file__)[:8]}')
    for file in ['images_meta.parquet','oof_S_img.parquet','oof_L_img.parquet','summary.json']:
        print(f'{file}_sha8={sha256(Path(out)/file)[:8]}')
    print('=== END REPORT P21-T08b ===')


def run(config_path, stage='run', resume=False):
    started=time.perf_counter()
    cfg=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    fixed_config(cfg)
    raw=Path(cfg['raw_dir'])
    ensure(raw.is_dir() and (raw/'RECORDS').is_file(),'Raw CTU-UHB path/RECORDS missing')
    inputhashes=verify_t02_locks(cfg)
    t02cfg=load_t02_cfg(cfg)
    labels,split=load_labels_split(cfg)
    cpu_torch()
    ensure(torch.cuda.is_available() is False or os.environ.get('CUDA_VISIBLE_DEVICES')=='-1',
           'CUDA not disabled')
    t08_reference(cfg)  # Hard check the accepted T08 source/summary, READ ONLY.
    source=identity(cfg,config_path)
    out=Path(cfg['project_dir'])/'outputs/t08b_deepfhr'
    plan=list(all_units())
    ensure(len(plan)==185 and len(set(plan))==185,'Expected 180+5 independent units')
    ensure(len({(k[2],k[3]) for k in plan if k[0]!='NC_L_img'})==9,
           'Repeat x seed grid not complete')
    if stage=='inspect':
        print('T08B_PREFLIGHT_PASS labels=518; folds=5; repeats=3; seeds=3; '
              'repeat_seed_pairs=9; main_units=180; control_units=5; T02_SHA_LOCK=PASS; '
              'T08_ORIGINAL_SHA=PASS; torch_CPU=PASS; no outputs written')
        return 0
    ensure(stage=='run','Only inspect or run stages supported')
    assert_lock(out,cfg,source,inputhashes,resume)
    prep_start=time.perf_counter()
    x,imageids,records,meta=prepare_images(labels,cfg,t02cfg,out,source)
    prep_elapsed=time.perf_counter()-prep_start
    print(f'T08B_IMAGES_PASS n={len(x)} median_per_record={meta.n_images.median():.2f} '
          f'dropped={meta.n_dropped.sum()} prep_runtime_s={prep_elapsed:.2f}',flush=True)
    foldcache,perm=make_fold_cache(labels,split,records)
    elapsed=[];pilot_path=out/'pilot.json'
    first=plan[0]
    first_y=labels.loc[records,first[1]].to_numpy(dtype=np.int64)
    p,info,resumed=run_unit(first,out,x,imageids,records,first_y,
                             foldcache[(first[0],first[1],first[2])],
                             source,inputhashes,cfg)
    if pilot_path.exists():
        pilot=json.loads(pilot_path.read_text(encoding='utf-8'))
        ensure(pilot.get('script_sha256')==source['script_sha256']
               and pilot.get('config_sha256')==source['config_sha256'],
               'Pilot identity mismatch')
    else:
        pilot={'task':'P21-T08b','script_sha256':source['script_sha256'],
               'config_sha256':source['config_sha256'],
               'first_unit':unit_file(out,first).name,'first_fit_runtime_s':info['runtime_s'],
               'estimated_total_fits':185,
               'estimated_training_runtime_s':float(info['runtime_s']*185),
               'estimate_basis':'one S_img/y705/repeat0/seed0/fold0 CPU fit; proxy only'}
        atomic_json(pilot_path,pilot)
    print('=== P21-T08b PILOT ===',flush=True)
    print(f"pilot_fit_s={pilot['first_fit_runtime_s']:.2f}; fits=185; "
          f"estimated_training_hours={pilot['estimated_training_runtime_s']/3600:.2f}; "
          'estimate_is_not_a_deadline=TRUE',flush=True)
    print('=== END PILOT ===',flush=True)
    elapsed.append(float(info['runtime_s']))
    for i,key in enumerate(plan[1:],start=2):
        protocol,label,rep,seed,fold=key
        y=perm if protocol=='NC_L_img' else labels.loc[records,label].to_numpy(dtype=np.int64)
        _,inf,was_resumed=run_unit(key,out,x,imageids,records,y,
                                  foldcache[(protocol,label,rep)],source,inputhashes,cfg)
        elapsed.append(float(inf['runtime_s']))
        print(f'T08B_UNIT {i}/185 {unit_file(out,key).name} '+
              f"{'RESUMED' if was_resumed else 'PASS'} best_epoch={inf['best_epoch']} "
              f"train_auc={inf['train_auroc']:.4f} test_sd={inf['test_probability_sd']:.6g} "
              f"fit_s={inf['runtime_s']:.1f}",flush=True)
    ensure(len(list((out/'units').glob('*.npz')))==185,'Unit count not 185')
    if (out/'summary.json').is_file():
        summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        verify_summary_files(out,labels,len(x))
        # Verify stored diagnostics against individual units on resume.
        ensure(summary['assay_sensitivity']==collect_assay_diagnostics(out),
               'Resumed assay diagnostics differ from finished units')
    else:
        summary=summarize(out,cfg,labels,records,imageids,foldcache,perm,source,inputhashes,
                          meta,prep_elapsed,sum(elapsed))
        verify_summary_files(out,labels,len(x))
    print_report(out,summary,time.perf_counter()-started)
    make_output_manifest(out)
    print(f'T08B_OUTPUT_SHA256_MANIFEST={sha256(out/"OUTPUT_SHA256.txt")[:8]}; file_count='+
          str(len((out/'OUTPUT_SHA256.txt').read_text(encoding='utf-8').splitlines())),flush=True)
    return 0 if summary['status']=='PASS' else 3

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=('inspect','run'),default='inspect')
    p.add_argument('--resume',action='store_true')
    arg=p.parse_args(argv)
    try:
        code=run(arg.config,arg.stage,arg.resume)
    except Exception as e:
        print(f'P21_T08B_STOP: {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        return 2
    return code

if __name__=='__main__':
    sys.exit(main())

"""P21-T07d (PREREG v1.6): WMFB v2 features, nested-CV sensitivity study.

Read-only T02/T03/T04/T05/T07c inputs. No Holm family membership.
Only the existing 11 feature positions are used. T03/T05 code is imported,
not modified. Frozen OOF scores are aligned by record ID, never row order.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import yaml
from common.io_wfdb import read_record
from common.bootstrap import discrimination_ci
from P21_CTG_Acidemia.src import t03_baselines as t03
from P21_CTG_Acidemia.src import t05_leakage as t05
from P21_CTG_Acidemia.src.ctg_features import clean_signals

TASK='P21-T07d'
PREREG='df06f9dd107ffb5b4132af3ade17aa9fec6c71d3f0b8154803c8ddfc3c59a557'
V2={
    'baseline':'baseline_v2', 'n_accel':'n_accel_v2','n_decel':'n_decel_v2',
    'decel_area':'decel_area_v2','n_prolonged_decel':'n_prolonged_decel_v2'
}
COLUMNS=tuple(V2.get(n,n) for n in t03.FEATURES)
NEW=tuple(V2.values())
OLD=tuple(n for n in t03.FEATURES if n not in V2)
EXPECTED_INPUT={
    'T02_FROZEN_SHA.txt':('labels.parquet','features_w30.parquet','split.csv'),
    'T03_FROZEN_SHA.txt':('outputs/t03_baselines/A_primary/oof.parquet',),
    'T04_FROZEN_SHA.txt':('outputs/t04_dl/A_primary/oof.parquet',),
    'T05_FROZEN_SHA.txt':('outputs/t05_leakage/segments_meta.parquet',
                         'outputs/t05_leakage/segments_cache.npz',
                         'outputs/t05_leakage/oof_S_seg.parquet',
                         'outputs/t05_leakage/oof_L1.parquet',
                         'outputs/t05_leakage/oof_L2.parquet',
                         'outputs/t05_leakage/oof_L3.parquet',
                         'outputs/t05_leakage/h2_tests.csv'),
    'T07c_FROZEN_SHA.txt':('wmfb_events.parquet','wmfb_baseline.npz'),
}


def check(ok,msg):
    if not ok: raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def hashes_from_lock(path):
    """Accept sha256sum, SHA8 tabulation, and name_sha8=hash formats.
    Do not guess when a lock contains multiple conflicting candidates.
    """
    found=[]
    digest=re.compile(r'(?<![0-9a-fA-F])(?:[0-9a-fA-F]{64}|[0-9a-fA-F]{8})(?![0-9a-fA-F])')
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        matches=digest.findall(line)
        if len(matches)!=1: continue
        found.append((line.replace('\\','/').lower(),matches[0].lower()))
    return found


def lock_entry(lines,basename,context=None):
    """Context suffix disambiguates identical oof.parquet under configs."""
    target=basename.lower().replace('\\','/')
    # Avoid matching e.g. oof_L1 to oof_L12.
    needle=re.compile(r'(?<![a-z0-9_])'+re.escape(Path(target).name)+r'(?=\b|_sha(?:256)?(?:_8|8)?\b)',re.I)
    entries=[]
    for line,h in lines:
        if not needle.search(line): continue
        if context is not None and context.lower().replace('\\','/') not in line: continue
        entries.append(h)
    vals=set(entries)
    check(len(vals)==1, f'FROZEN_LOCK_MISSING_OR_AMBIGUOUS {basename} context={context} candidates={list(vals)}')
    return vals.pop()


def frozen_paths(cfg):
    project=Path(cfg['project_dir']); proc=Path(cfg['proc_dir'])
    sources={
        'labels.parquet':proc/'labels.parquet',
        'features_w30.parquet':proc/'features_w30.parquet',
        'split.csv':proc/'split.csv',
        'wmfb_events.parquet':proc/'wmfb'/'wmfb_events.parquet',
        'wmfb_baseline.npz':proc/'wmfb'/'wmfb_baseline.npz',
    }
    for relative in EXPECTED_INPUT['T03_FROZEN_SHA.txt']+EXPECTED_INPUT['T04_FROZEN_SHA.txt']+EXPECTED_INPUT['T05_FROZEN_SHA.txt']:
        sources[relative]=project/relative
    return sources


def verify_frozen(cfg):
    sources=frozen_paths(cfg)
    checked={}
    for lock_name,entries in EXPECTED_INPUT.items():
        lock=Path(cfg['project_dir'])/'reports'/lock_name
        check(lock.is_file(),f'MISSING_FROZEN_LOCK {lock}')
        rows=hashes_from_lock(lock)
        summary_chain={}
        if lock_name=='T05_FROZEN_SHA.txt':
            # T05 official summary contains full SHA256s of its own outputs.
            # Trust it as a secondary lock only if frozen SHA explicitly anchors
            # summary.json itself. Without this anchor, FAIL CLOSED.
            try:
                summary_lock=lock_entry(rows,'summary.json')
            except RuntimeError:
                summary_lock=None
            summary_path=Path(cfg['project_dir'])/'outputs/t05_leakage/summary.json'
            if summary_lock is not None:
                check(summary_path.is_file() and sha(summary_path).startswith(summary_lock),
                      'T05_FROZEN_SUMMARY_SHA_MISMATCH')
                obj=json.loads(summary_path.read_text(encoding='utf-8'))
                check(obj.get('task')=='P21-T05' and obj.get('status')=='PASS',
                      'T05_FROZEN_SUMMARY_NOT_PASS')
                summary_chain=obj.get('output_sha256',{})
                checked['SOURCE::t05_summary.json']=sha(summary_path)
        for key in entries:
            path=sources[key]
            check(path.is_file(),f'MISSING_FROZEN_INPUT {path}')
            context='A_primary/oof.parquet' if Path(key).name=='oof.parquet' else None
            try:
                prefix=lock_entry(rows,key,context=context)
            except RuntimeError:
                if context and lock_name in ('T03_FROZEN_SHA.txt','T04_FROZEN_SHA.txt'):
                    # Accept the documented named-config OOF_SHA format too.
                    named={h for line,h in rows if re.search(r'(?<![a-z0-9])a_primary(?![a-z0-9])',line)
                           and re.search(r'(?<![a-z0-9])oof(?:\.parquet|_sha(?:256)?(?:_8|8)?)',line)}
                    check(len(named)==1,f'AMBIGUOUS_A_PRIMARY_OOF_LOCK {lock_name}')
                    prefix=named.pop()
                else:
                    # Only T05's cryptographically anchored summary qualifies.
                    check(lock_name=='T05_FROZEN_SHA.txt' and Path(key).name in summary_chain,
                          f'FROZEN_LOCK_MISSING_HASH {lock_name} {key}')
                    prefix=summary_chain[Path(key).name]
                    check(isinstance(prefix,str) and len(prefix)==64 and
                          re.fullmatch('[0-9a-fA-F]{64}',prefix) is not None,
                          'T05_SUMMARY_INVALID_SHA256')
            digest=sha(path)
            check(digest.startswith(prefix.lower()),
                  f'FROZEN_SHA_MISMATCH {key} expected={prefix} actual={digest[:8]}')
            checked[key]=digest
        checked[f'LOCK::{lock_name}']=sha(lock)
    # Immutable processing code: identity is rechecked on resume.
    for name in ('src/ctg_features.py','src/t03_baselines.py','src/t05_leakage.py',
                 'configs/t02.yaml','configs/t03.yaml','configs/t05.yaml'):
        p=Path(cfg['project_dir'])/name
        check(p.is_file(),f'MISSING_ORIGINAL_SOURCE {name}')
        checked['SOURCE::'+name]=sha(p)
    check(Path(cfg['raw_dir'],'RECORDS').is_file(),'MISSING_RAW_RECORDS')
    return checked


def read_config(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    expected={'task','prereg_version','prereg_sha256','project_dir','raw_dir',
              'proc_dir','output_dir','bootstrap_resamples','bootstrap_seed','cpu_threads'}
    check(set(cfg)==expected,'T07D_CONFIG_KEYS_CHANGED')
    check(cfg['task']==TASK and str(cfg['prereg_version'])=='1.6'
          and cfg['prereg_sha256']==PREREG,'PREREG_MISMATCH')
    check((cfg['bootstrap_resamples'],cfg['bootstrap_seed'],cfg['cpu_threads'])==(10000,42,4),
          'LOCKED_BOOTSTRAP_OR_CPU_CHANGED')
    check(Path(cfg['output_dir'])==Path(cfg['project_dir'])/'outputs'/'t07d_featv2',
          'OUTPUT_TARGET_CHANGED')
    check(tuple(t03.FEATURES)==('baseline','stv_ms','ltv','pct_min_var','n_accel','n_decel',
          'decel_area','n_prolonged_decel','n_contractions','lf_power','hf_power'),
          'T03_ORIGINAL_FEATURE_ORDER_CHANGED')
    cfg03=yaml.safe_load((Path(cfg['project_dir'])/'configs'/'t03.yaml').read_text(encoding='utf-8'))
    t03.locked_config(cfg03)
    cfg05=yaml.safe_load((Path(cfg['project_dir'])/'configs'/'t05.yaml').read_text(encoding='utf-8'))
    t05.config_check(cfg05)
    return cfg,cfg03,cfg05


def validate_events(events,n_by_record):
    expected=['record_id','type','start_s','end_s','pct_missing_raw','flag_missing']
    check(set(events.columns)==set(expected),'WMFB_EVENTS_SCHEMA_MISMATCH')
    events=events.copy()
    events.record_id=events.record_id.astype(str)
    check(events['type'].isin(['acc','dec']).all(),'WMFB_EVENT_UNKNOWN_TYPE')
    check(events['flag_missing'].notna().all() and events['flag_missing'].isin([True,False]).all(),
          'WMFB_FLAG_MISSING_INVALID')
    for rid,g in events.groupby('record_id'):
        check(rid in n_by_record,f'WMFB_EVENT_UNKNOWN_RECORD {rid}')
        a=g[['start_s','end_s','pct_missing_raw']].to_numpy(float)
        check(np.isfinite(a).all() and (a[:,0]>=0).all() and (a[:,1]>a[:,0]).all()
              and (a[:,1]<=n_by_record[rid]/4+1).all(),f'INVALID_WMFB_EVENT {rid}')
        check(((a[:,2]>=0)&(a[:,2]<=100)).all(),f'WMFB_MISSING_PCT_INVALID {rid}')
        check(np.array_equal(np.asarray(g.flag_missing,bool),a[:,2]>50),
              f'FLAG_MISSING_INCONSISTENT {rid}')
    return events


def wmfb_only(baseline,events,clean_fhr,start,end,fs=4):
    """Outcome-blind WMFB features for zero-based half-open sample window.

    Midpoint membership uses seconds; deceleration area is restricted to the
    present window and T02-cleaned valid samples, even for straddling events.
    """
    baseline=np.asarray(baseline,dtype=float).reshape(-1)
    clean_fhr=np.asarray(clean_fhr,dtype=float).reshape(-1)
    check(len(baseline)==len(clean_fhr),'WMFB_BASELINE_LENGTH_VS_RAW')
    check(0<=start<end<=len(baseline),'INVALID_WINDOW')
    part=baseline[start:end]
    check(np.isfinite(part).all(),'WMFB_NONFINITE_BASELINE')
    got={'baseline_v2':float(np.median(part)), 'n_accel_v2':0,
         'n_decel_v2':0,'n_prolonged_decel_v2':0,'decel_area_v2':0.0}
    for row in events:
        # Supports dataframe itertuples and dictionaries, no phenotype needed.
        typ=row.type if hasattr(row,'type') else row['type']
        a=float(row.start_s if hasattr(row,'start_s') else row['start_s'])
        z=float(row.end_s if hasattr(row,'end_s') else row['end_s'])
        bad=bool(row.flag_missing if hasattr(row,'flag_missing') else row['flag_missing'])
        if bad or not (start/fs <= (a+z)/2 < end/fs): continue
        if typ=='acc':got['n_accel_v2']+=1
        elif typ=='dec':
            got['n_decel_v2']+=1
            if z-a>=120:got['n_prolonged_decel_v2']+=1
            lo=max(start,int(np.rint(a*fs)))
            hi=min(end,int(np.rint(z*fs)))
            if hi>lo:
                ok=np.isfinite(clean_fhr[lo:hi]) & np.isfinite(baseline[lo:hi])
                got['decel_area_v2']+=float(np.maximum(
                    baseline[lo:hi][ok]-clean_fhr[lo:hi][ok],0).sum()/fs)
        else:raise ValueError('UNSUPPORTED_EVENT_TYPE')
    return got


def assemble_v2(old_features, fresh, id_column):
    """Retain the six original T02/T05 columns exactly, no phenotype input."""
    frame=old_features.copy()
    check(set(t03.FEATURES).issubset(frame.columns),'SIX_SOURCE_FEATURES_MISSING')
    check(len(fresh)==len(frame),'WMFB_FEATURE_ALIGNMENT_LENGTH')
    for original,new in V2.items():
        check(new in fresh.columns,f'MISSING_NEW_FEATURE {new}')
        frame=frame.drop(columns=original)
        frame[new]=fresh[new].to_numpy(float)
    cols=[id_column]+list(COLUMNS)
    if id_column not in frame.columns:frame.insert(0,id_column,fresh[id_column].astype(str).to_numpy())
    frame=frame[cols]
    check(frame[id_column].is_unique,'DUPLICATE_FEATURE_ID')
    check(not np.isinf(frame[list(COLUMNS)].to_numpy(float)).any(), 'INFINITE_FEATURE_V2')
    for name in OLD:
        check(np.allclose(frame[name].to_numpy(float),old_features[name].to_numpy(float),
                          equal_nan=True,rtol=0,atol=0),f'UNMODIFIED_COLUMN_CHANGED {name}')
    return frame


def prepare_features(cfg,cfg05,existing_identity=None):
    out=Path(cfg['output_dir']); proc=Path(cfg['proc_dir']); raw=Path(cfg['raw_dir'])
    from P21_CTG_Acidemia.src.ctg_features import clean_signals
    t02cfg=t05.t02_cfg(Path(cfg['project_dir']))
    originals=pd.read_parquet(proc/'features_w30.parquet')
    check(len(originals)==552 and originals.record_id.astype(str).is_unique,'FROZEN_W30_BAD')
    originals.record_id=originals.record_id.astype(str)
    with np.load(Path(cfg['project_dir'])/'outputs/t05_leakage/segments_cache.npz',allow_pickle=False) as npz:
        old_ids=npz['row_id'].astype(str)
        old_rec=npz['record_id'].astype(str)
        old_feat=npz['features'].astype(float)
    check(len(set(old_ids))==len(old_ids) and old_feat.shape==(len(old_ids),11),
          'FROZEN_T05_SEGMENT_CACHE_SHAPE')
    segorig=pd.DataFrame(old_feat,columns=t03.FEATURES)
    segorig.insert(0,'row_id',old_ids)
    segmeta=pd.read_parquet(Path(cfg['project_dir'])/'outputs/t05_leakage/segments_meta.parquet')
    segmeta.record_id=segmeta.record_id.astype(str)
    check(len(segmeta)==518 and segmeta.record_id.is_unique and
          int(segmeta.n_segments.sum())==len(old_ids),'FROZEN_SEGMENTS_META_MISMATCH')
    seg_by_id={rid:[] for rid in segmeta.record_id}
    for ix,(rowid,rid) in enumerate(zip(old_ids,old_rec)):
        check(rowid.startswith(rid+'__'),'BAD_T05_SEGMENT_ROW_ID')
        start=int(rowid.rsplit('__',1)[1]);seg_by_id[rid].append((ix,start))
    names=(raw/'RECORDS').read_text(encoding='utf-8-sig').splitlines()
    names=[x.strip() for x in names if x.strip()]
    check(len(names)==552 and len(set(names))==552 and set(names)==set(originals.record_id),
          'CTU_552_RECORDS_MISMATCH')
    with np.load(proc/'wmfb/wmfb_baseline.npz',allow_pickle=False) as npz:
        check(len(npz.files)==552 and set(npz.files)==set(names),'WMFB_BASELINE_KEY_MISMATCH')
        n_by_record={rid:np.asarray(npz[rid]).size for rid in names}
    events=validate_events(pd.read_parquet(proc/'wmfb/wmfb_events.parquet'),n_by_record)
    event_map={rid:list(group.itertuples(index=False)) for rid,group in events.groupby('record_id')}
    w30rows=[];segrows={};stage_checked=0
    started=time.perf_counter()
    with np.load(proc/'wmfb/wmfb_baseline.npz',allow_pickle=False) as npz:
        for i,rid in enumerate(names,1):
            rec=read_record(raw,rid)
            check(rec.fs==4,'CTU_FS_4_REQUIRED')
            channels={str(k).upper():j for j,k in enumerate(rec.channel_names)}
            check('FHR' in channels and 'UC' in channels,'CTU_FHR_UC_REQUIRED')
            fhr,uc=clean_signals(rec.signal[:,channels['FHR']],
                                 rec.signal[:,channels['UC']],t02cfg)
            try:stage=float(rec.clinical.get('Pos. II.st.'))
            except (ValueError,TypeError):stage=float('nan')
            end=len(fhr) if not np.isfinite(stage) or stage<0 else int(stage)
            check(0<end<=len(fhr),'STAGE1_END_INVALID')
            b=np.asarray(npz[rid],dtype=float).reshape(-1)
            check(len(b)==end and np.isfinite(b).all(),f'WMFB_LENGTH_MISMATCH {rid}')
            start=max(0,end-30*60*4)
            f=wmfb_only(b,event_map.get(rid,()),fhr[:end],start,end)
            w30rows.append({'record_id':rid,**f})
            if rid in seg_by_id:
                expected=t05.stage1_segments(fhr[:end],uc[:end],fs=4,minutes=10,
                                              step_minutes=5,max_missing=50)
                kept,dropped=expected
                expected_starts=[s for s,_,_ in kept]
                actual_starts=[s for _,s in seg_by_id[rid]]
                check(expected_starts==actual_starts,f'T05_SEGMENT_CUTOFF_MISMATCH {rid}')
                row=segmeta.loc[segmeta.record_id==rid].iloc[0]
                check(int(row.n_segments)==len(kept) and int(row.n_dropped_missing)==dropped,
                      f'T05_SEGMENT_META_INCONSISTENT {rid}')
                stage_checked+=1
                for (ix,s),(s2,z,_) in zip(seg_by_id[rid],kept):
                    check(s==s2,'SEGMENT_OFFSET_MISMATCH')
                    segrows[ix]={'row_id':old_ids[ix],'record_id':rid,
                                 'start_sample':int(s),'end_sample':int(z),
                                 **wmfb_only(b,event_map.get(rid,()),fhr[:end],s,z)}
            if i%100==0:print(f'T07D_PREP_PROGRESS={i}/552',flush=True)
    check(stage_checked==518 and len(segrows)==len(old_ids),'T05_SEGMENT_RECONSTRUCTION_INCOMPLETE')
    featw=pd.DataFrame(w30rows)
    frozen=originals.set_index('record_id').loc[featw.record_id].reset_index()
    fw=assemble_v2(frozen,featw,'record_id')
    segment_extra=pd.DataFrame([segrows[i] for i in range(len(old_ids))])
    sx=assemble_v2(segorig,segment_extra,'row_id')
    sx.insert(1,'record_id',segment_extra.record_id)
    sx.insert(2,'start_sample',segment_extra.start_sample)
    sx.insert(3,'end_sample',segment_extra.end_sample)
    check(len(fw)==552 and len(sx)==len(old_ids),'V2_FEATURE_COUNT_MISMATCH')
    return fw,sx,time.perf_counter()-started


def validate_oof(oof,ids,repeats,pcols,group='record_id'):
    check({group,'repeat','y',*pcols}.issubset(oof.columns),'OOF_SCHEMA_MISMATCH')
    check(len(oof)==len(ids)*repeats,'OOF_ROW_COUNT_MISMATCH')
    check(set(oof[group])==set(ids),'OOF_RECORD_ID_MISMATCH')
    check(oof.duplicated([group,'repeat']).sum()==0,'OOF_DUPLICATE_RECORD_REPEAT')
    check(oof.groupby(group).size().eq(repeats).all(),'OOF_INCOMPLETE_REPEATS')
    check(np.isfinite(oof[pcols].to_numpy(float)).all(),'OOF_NONFINITE')
    check(((oof[pcols]>=0)&(oof[pcols]<=1)).all().all(),'OOF_INVALID_PROBABILITY')


def means(frame,pcol,group='record_id'):
    check({'y',group,pcol}.issubset(frame.columns),'MEAN_INPUT_SCHEMA')
    check(frame.groupby(group).y.nunique().eq(1).all(),'MEAN_LABEL_INCONSISTENT')
    return frame.groupby(group,sort=True).agg(y=('y','first'),p=(pcol,'mean'))


def align_two(a,b):
    left=a[['y','p']].rename(columns={'p':'a'})
    right=b[['y','p']].rename(columns={'p':'b'})
    joined=left.join(right,how='inner',rsuffix='_b',validate='one_to_one')
    check(len(joined)==len(a)==len(b) and np.array_equal(joined.y,joined.y_b),
          'PAIRED_RECORD_ALIGNMENT_FAILED')
    return joined.y.to_numpy(dtype=int),joined.a.to_numpy(float),joined.b.to_numpy(float)


def prepare_run_context(cfg,cfg05):
    proc=Path(cfg['proc_dir']);project=Path(cfg['project_dir']);out=Path(cfg['output_dir'])
    labels=pd.read_parquet(proc/'labels.parquet')
    labels.record_id=labels.record_id.astype(str)
    check(len(labels)==552 and labels.record_id.is_unique,'LABELS_552_INVALID')
    eligible=labels.loc[~labels.excluded.astype(bool)].set_index('record_id').sort_index()
    check(len(eligible)==518 and int(eligible.y705.sum())==30,'ELIGIBLE_COHORT_CHANGED')
    features=pd.read_parquet(out/'features_w30_v2.parquet')
    features.record_id=features.record_id.astype(str)
    check(features.record_id.is_unique and set(eligible.index).issubset(features.record_id),
          'V2_FEATURE_ID_MISMATCH')
    feat=features.set_index('record_id').loc[eligible.index,list(COLUMNS)].to_numpy(float)
    segments=pd.read_parquet(out/'segments_v2.parquet')
    check(segments.row_id.is_unique and set(segments.record_id)==set(eligible.index),
          'SEGMENT_V2_ELIGIBLE_MISMATCH')
    segment_meta=pd.read_parquet(project/'outputs/t05_leakage/segments_meta.parquet')
    check(len(segments)==int(segment_meta.n_segments.sum()),'SEGMENT_COUNT_MISMATCH')
    segments.record_id=segments.record_id.astype(str)
    segments['y']=segments.record_id.map(eligible.y705)
    check(segments.y.notna().all(),'SEGMENT_LABEL_MAPPING_FAILED')
    splits=pd.read_csv(proc/'split.csv',dtype={'record_id':str})
    check(t03.verify_nested_splits(splits,eligible.index.to_numpy(),10,5,5),
          'STRICT_SPLIT_INVALID')
    # Deliberately do not modify T05 code: reuse its contexts (L1/L2/L3) and SMOTE.
    data={'ids':eligible.index.to_numpy(dtype=str), 'y':eligible.y705.to_numpy(dtype=int),
          'features':feat, 'tensor':np.zeros((518,1,1),dtype=np.float32),
          'splits':splits, 'labels':eligible,
          'input_shas':{'features_w30.parquet':sha(out/'features_w30_v2.parquet')}}
    seg={'row_id':segments.row_id.to_numpy(dtype=str),
         'record_id':segments.record_id.to_numpy(dtype=str),
         'y':segments.y.to_numpy(dtype=int),
         'features':segments[list(COLUMNS)].to_numpy(float),
         'tensor':np.zeros((len(segments),1,1),dtype=np.float32)}
    check(not np.isinf(feat).any() and not np.isinf(seg['features']).any(),
          'NONFINITE_V2_FEATURE')
    return data,seg,features,segments


def unit_name(protocol,model,repeat,fold):
    return f'{protocol}_{model}_r{repeat:02d}_f{fold}.npz'


def all_units():
    for r in range(10):
        for fold in range(5):
            for model in ('M0','M1'):
                yield ('S',model,r,fold)
    for p in ('S_seg','L1','L2','L3'):
        for r in range(3):
            for fold in range(5):
                yield (p,'M1',r,fold)


def get_context(data,seg,cfg05,selected,protocol,model,repeat,fold,t05_fn=None):
    if protocol!='S':
        if t05_fn is None:t05_fn,_=t05.contexts(cfg05,data,seg,selected)
        return t05_fn(protocol,'M1',0,repeat,fold)
    block=data['splits'].loc[(data['splits'].repeat==repeat)&(data['splits'].outer_fold==fold)]
    train=block.loc[block.inner_fold>=0]
    test=block.loc[block.inner_fold==-1]
    lookup={rid:i for i,rid in enumerate(data['ids'])}
    tr_ids=train.record_id.to_numpy(str);te_ids=test.record_id.to_numpy(str)
    check(not set(tr_ids)&set(te_ids),'STRICT_OUTER_RECORD_OVERLAP')
    ixtr=np.asarray([lookup[x] for x in tr_ids]);ixte=np.asarray([lookup[x] for x in te_ids])
    return {'row_id':te_ids,'record_id':te_ids,'y':data['y'][ixte],
            'x_train':data['features'][ixtr], 'y_train':data['y'][ixtr],
            'x_test':data['features'][ixte],
            'inner':train.inner_fold.to_numpy(int),'train_row_ids':tr_ids,
            'train_record_ids':tr_ids,'test_record_ids':te_ids}


def unit_meta_expected(ctx,key,identity):
    return {'key':list(key),'identity_sha':identity,
            'train_rows_sha':t05.row_hash(ctx['train_row_ids']),
            'test_rows_sha':t05.row_hash(ctx['row_id'])}


def read_unit(path,expected,ctx):
    with np.load(path,allow_pickle=False) as saved:
        check(set(saved.files)=={'row_id','record_id','y','p','meta_json'},
              f'UNIT_SCHEMA_CHANGED {path.name}')
        rows=saved['row_id'].astype(str);rec=saved['record_id'].astype(str)
        y=saved['y'].astype(int);p=saved['p'].astype(float)
        metadata=json.loads(str(saved['meta_json'].item()))
    for k,v in expected.items():check(metadata.get(k)==v,f'UNIT_IDENTITY_CHANGED {k} {path.name}')
    check(np.array_equal(rows,ctx['row_id']) and np.array_equal(rec,ctx['record_id'])
          and np.array_equal(y,ctx['y']) and len(p)==len(y) and np.isfinite(p).all()
          and ((p>=0)&(p<=1)).all(),f'UNIT_PREDICTION_INVALID {path.name}')
    return p,metadata


def train_one_unit(out,data,seg,cfg03,cfg05,selected,key,identity,t05_fn=None):
    protocol,model,r,fold=key
    ctx=get_context(data,seg,cfg05,selected,protocol,model,r,fold,t05_fn)
    expected=unit_meta_expected(ctx,key,identity)
    file=out/'units'/unit_name(*key)
    if file.exists():
        p,meta=read_unit(file,expected,ctx)
        return p,meta,True
    started=time.perf_counter()
    if protocol=='L3':
        use=selected['feature_indices'];params=selected['params']
        estimator=t03.make_model('M1',params)
        estimator.fit(ctx['x_train'][:,use],ctx['y_train'])
        pred=estimator.predict_proba(ctx['x_test'][:,use])[:,1]
        fit={'params':params,'leaky_global_selection':selected['features']}
    else:
        fitcfg=cfg03 if protocol=='S' else t05.t03_cfg(cfg05)
        pred,params,score=t03.tune_and_predict(model,ctx['x_train'],ctx['y_train'],
                 ctx['inner'],ctx['x_test'],fitcfg,ctx['train_row_ids'],ctx['row_id'])
        fit={'params':params,'mean_inner_auroc':float(score)}
    pred=np.asarray(pred,dtype=float).reshape(-1)
    check(np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all(),
          'NONFINITE_UNIT_PREDICTIONS')
    meta={**expected,**fit,'runtime_s':time.perf_counter()-started}
    def writer(f):np.savez_compressed(f,row_id=ctx['row_id'].astype(str),
                   record_id=ctx['record_id'].astype(str),y=ctx['y'].astype(np.int8),
                   p=pred,meta_json=np.asarray(json.dumps(t05.json_ready(meta),sort_keys=True)))
    t05.atomic_new(file,writer)
    read_unit(file,expected,ctx)
    return pred,meta,False


def aggregate_units(out,data,seg,cfg03,cfg05,selected,identity,t05_fn=None):
    result={}
    for protocol in ('S','S_seg','L1','L2','L3'):
        rows=[]
        for r in (range(10) if protocol=='S' else range(3)):
            for fold in range(5):
                for model in (('M0','M1') if protocol=='S' else ('M1',)):
                    key=protocol,model,r,fold
                    ctx=get_context(data,seg,cfg05,selected,*key,t05_fn)
                    expected=unit_meta_expected(ctx,key,identity)
                    p,_=read_unit(out/'units'/unit_name(*key),expected,ctx)
                    if protocol=='S':
                        rows.extend({'record_id':rid,'repeat':r,'outer_fold':fold,'y':int(y),
                                     'model':model,'seed':0,'row_id':rid,'p':float(value)}
                                    for rid,y,value in zip(ctx['record_id'],ctx['y'],p))
                    else:
                        rows.extend({'record_id':rid,'row_id':row,'repeat':r,
                                     'outer_fold':fold,'y':int(y), 'model':model,
                                     'seed':0,'p':float(value)}
                                    for row,rid,y,value in zip(ctx['row_id'],ctx['record_id'],ctx['y'],p))
        frame=pd.DataFrame(rows)
        check(not frame.empty and np.isfinite(frame.p.to_numpy()).all(),f'OOF_EMPTY_OR_NAN {protocol}')
        if protocol=='S':
            a=frame.loc[frame.model=='M0'].drop(columns=['model','seed','row_id']).rename(columns={'p':'p_M0'})
            b=frame.loc[frame.model=='M1'][['record_id','repeat','p']].rename(columns={'p':'p_M1'})
            frame=a.merge(b,on=['record_id','repeat'],validate='one_to_one').sort_values(['repeat','record_id']).reset_index(drop=True)
            validate_oof(frame,data['ids'],10,['p_M0','p_M1'])
        else:
            check(frame.groupby(['row_id','repeat']).size().eq(1).all() and
                  frame.groupby('row_id').repeat.nunique().eq(3).all(),
                  f'SEGMENT_REPEATS_INCOMPLETE_OR_DUPLICATE {protocol}')
            if protocol in ('S_seg','L1'):
                check(set(frame.row_id)==set(seg['row_id']),
                      f'SEGMENT_ID_SET_CHANGED {protocol}')
            if protocol=='L3':
                check(set(frame.row_id)==set(data['ids']),
                      'L3_RECORD_COVERAGE_CHANGED')
        result[protocol]=frame
    return result


def as_model_record_mean(oof,model):
    return means(oof,'p_'+model)


def as_t05_average(frame):
    return t05.average_oof(frame,'M1').set_index('row_id')


def compare_results(cfg,data,seg,oofs,project):
    old03=pd.read_parquet(project/'outputs/t03_baselines/A_primary/oof.parquet')
    old04=pd.read_parquet(project/'outputs/t04_dl/A_primary/oof.parquet')
    for f in (old03,old04):f.record_id=f.record_id.astype(str)
    s=oofs['S'];validate_oof(s,data['ids'],10,['p_M0','p_M1'])
    old03=old03.loc[old03.repeat.isin(range(10))].copy()
    validate_oof(old03,data['ids'],10,['p_M0','p_M1'])
    prior=s[['record_id','repeat','outer_fold','y']].merge(
        old03[['record_id','repeat','outer_fold','y']],
        on=['record_id','repeat'],suffixes=('_v2','_old'),validate='one_to_one')
    check(len(prior)==518*10 and
          (prior.outer_fold_v2==prior.outer_fold_old).all() and
          (prior.y_v2==prior.y_old).all(),'T03_V2_OOF_SPLIT_MISMATCH')
    check(set(old04.record_id)==set(data['ids']) and len(old04)==518*10 and
          not old04.duplicated(['record_id','repeat']).any(),'T04_OOF_PRIMARY_INCOMPLETE')
    prior4=s[['record_id','repeat','outer_fold','y']].merge(
        old04[['record_id','repeat','outer_fold','y']],
        on=['record_id','repeat'],suffixes=('_v2','_old'),validate='one_to_one')
    check(len(prior4)==518*10 and
          (prior4.outer_fold_v2==prior4.outer_fold_old).all() and
          (prior4.y_v2==prior4.y_old).all(),'T04_V2_OOF_SPLIT_MISMATCH')
    check(all(f'p_{m}_s{k}' in old04.columns for m in ('M2','M3') for k in range(3)),
          'T04_M2_M3_SEED_COLUMNS_MISSING')
    m3=old04.assign(p=old04[[f'p_M3_s{k}' for k in range(3)]].mean(axis=1))
    m2=old04.assign(p=old04[[f'p_M2_s{k}' for k in range(3)]].mean(axis=1))
    summary={};rows=[]
    pred={
        'M0-v2':as_model_record_mean(s,'M0'), 'M1-v2':as_model_record_mean(s,'M1'),
        'M0':means(old03,'p_M0'), 'M1':means(old03,'p_M1'),
        'M2':means(m2,'p'), 'M3':means(m3,'p')}
    B=int(cfg['bootstrap_resamples']);seed=int(cfg['bootstrap_seed'])
    for name,table in pred.items():
        y=table.y.to_numpy(int);p=table.p.to_numpy(float)
        ci=discrimination_ci(y,p,B,seed)
        summary[name]={'auroc':float(t05.auc_fast(y,p)),
                       'ci95':ci['auroc_ci95'], 'n':int(len(y))}
    def paired(name,a,b,analysis='SENSITIVITY',groups=None):
        y,aa,bb=align_two(a,b)
        result=t05.bootstrap_auc_delta(y,aa,bb,groups=groups,B=B,seed=seed)
        rows.append({'analysis':analysis,'contrast':name,'delta_AUROC':result['delta'],
             'ci95_low':result['ci95'][0],'ci95_high':result['ci95'][1],
             'p_raw':result['p_raw'],'bootstrap_valid':result['bootstrap_valid'],
             'n':len(y),'in_holm_family':False})
        return result
    paired('M1-v2 - M1',pred['M1-v2'],pred['M1'])
    paired('M0-v2 - M0',pred['M0-v2'],pred['M0'])
    paired('M3 - M1-v2',pred['M3'],pred['M1-v2'],analysis='H1_SENSITIVITY')
    paired('M2 - M1-v2',pred['M2'],pred['M1-v2'],analysis='H1_SENSITIVITY')
    baseline_s=pred['M1-v2']
    leakage={}
    for protocol in ('L1','L2','L3'):
        leaky=as_t05_average(oofs[protocol]);
        if protocol=='L1':
            strict=as_t05_average(oofs['S_seg'])
            y,aa,bb=align_two(leaky,strict)
            clusters=leaky.loc[strict.index,'record_id'].to_numpy(str)
            row=t05.bootstrap_auc_delta(y,aa,bb,groups=clusters,B=B,seed=seed)
            rows.append({'analysis':'LEAKAGE','contrast':'L1 - S_seg_v2 (segment)',
                'delta_AUROC':row['delta'],'ci95_low':row['ci95'][0],
                'ci95_high':row['ci95'][1], 'p_raw':row['p_raw'],
                'bootstrap_valid':row['bootstrap_valid'],'n':len(y),'in_holm_family':False})
            a=leaky.groupby('record_id').agg(y=('y','first'),p=('p','mean'))
            b=strict.groupby('record_id').agg(y=('y','first'),p=('p','mean'))
            rec=paired('L1 - S_seg_v2 (record)',a,b,analysis='LEAKAGE')
            leakage[protocol]={'segment':row,'record':rec}
        else:
            real=leaky if protocol=='L3' else leaky.loc[leaky.index.str.startswith('real::')]
            check(len(real)==518,'L2_L3_REAL_ROW_518_REQUIRED')
            real=real.set_index('record_id')
            rec=paired(f'{protocol} - S_v2 (real record)',real,baseline_s,analysis='LEAKAGE')
            leakage[protocol]={'real_record':rec,
                               'auroc_all_augmented_descriptive':float(t05.auc_fast(
                                   leaky.y.to_numpy(int),leaky.p.to_numpy(float))) if protocol=='L2' else None}
    # Old T05 leakage contrasts, if available. Only frozen comparisons are read.
    h2=pd.read_csv(project/'outputs/t05_leakage/h2_tests.csv')
    originals={}
    for p in ('L1','L2','L3'):
        subset=h2.loc[h2.protocol.eq(p)&h2.model.eq('M1')]
        check(len(subset)==1,f'ORIGINAL_T05_H2_M1_ROW_MISSING {p}')
        r=subset.iloc[0]
        originals[p]={'delta':float(r['delta']),
                      'ci95':[float(r['ci95_low']),float(r['ci95_high'])]}
    # Recompute the original frozen L1 record-level paired contrast because T05
    # tabulates segment-level H2 but only separate record-level AUC intervals.
    frozen_L1=pd.read_parquet(project/'outputs/t05_leakage/oof_L1.parquet')
    frozen_seg=pd.read_parquet(project/'outputs/t05_leakage/oof_S_seg.parquet')
    a1=t05.average_oof(frozen_L1,'M1').groupby('record_id',as_index=True).agg(
        y=('y','first'),p=('p','mean'))
    b1=t05.average_oof(frozen_seg,'M1').groupby('record_id',as_index=True).agg(
        y=('y','first'),p=('p','mean'))
    yy,aa,bb=align_two(a1,b1)
    oldrec=t05.bootstrap_auc_delta(yy,aa,bb,B=B,seed=seed)
    originals['L1']['record']=oldrec
    rows.append({'analysis':'ORIGINAL_T05_REFERENCE',
      'contrast':'L1 - S_seg_original (record)',
      'delta_AUROC':oldrec['delta'],'ci95_low':oldrec['ci95'][0],
      'ci95_high':oldrec['ci95'][1],'p_raw':oldrec['p_raw'],
      'bootstrap_valid':oldrec['bootstrap_valid'],'n':len(yy),'in_holm_family':False})
    for p,orig in originals.items():
        rows.append({'analysis':'ORIGINAL_T05_REFERENCE',
          'contrast':f'{p} - S_original ({"segment" if p=="L1" else "real record"})',
          'delta_AUROC':orig['delta'],'ci95_low':orig['ci95'][0],
          'ci95_high':orig['ci95'][1],
          'p_raw':float(h2.loc[h2.protocol.eq(p)&h2.model.eq('M1'),'p_raw'].iloc[0]),
          'bootstrap_valid':int(h2.loc[h2.protocol.eq(p)&h2.model.eq('M1'),
                                 'bootstrap_valid'].iloc[0]),
          'n':len(seg['row_id']) if p=='L1' else 518,'in_holm_family':False})
    # E2: locked subgroup rule (<= median vs > median), record-level average predictions.
    ytable=pred['M1-v2'].join(data['labels'][['fhr_nan_pct_w30']],how='left')
    med=float(ytable.fhr_nan_pct_w30.median())
    e2={}
    for group,mask in (('le_median',ytable.fhr_nan_pct_w30<=med),
                       ('gt_median',ytable.fhr_nan_pct_w30>med)):
        ids=ytable.index[mask]
        aa=pred['M3'].loc[ids]; bb=pred['M1-v2'].loc[ids]
        yy,ap,bp=align_two(aa,bb)
        check(len(np.unique(yy))==2, f'E2_GROUP_ONE_CLASS {group}')
        res=paired(f'E2_v2 {group} M3 - M1-v2',aa,bb,analysis='EXPLORATORY_E2')
        e2[group]={'n':len(ids),'positives':int(sum(yy)),'median_missing_cut':med,
                   'm3_auroc':float(t05.auc_fast(yy,ap)),'m3_ci95':res['ci95_l'],
                   'm1v2_auroc':float(t05.auc_fast(yy,bp)),'m1v2_ci95':res['ci95_s'],
                   'delta':res}
    return pd.DataFrame(rows),{'model_metrics':summary,'leakage':leakage,
                                'original_T05_leakage':originals,'E2_v2':e2}


def save_new(path,payload,kind='json'):
    if kind=='json':t05.save_json(path,payload)
    elif kind=='csv':t05.atomic_new(path,lambda f: f.write(payload.to_csv(index=False).encode('utf-8')))
    elif kind=='parquet':t05.atomic_new(path,lambda f: payload.to_parquet(f,index=False))
    elif kind=='text':t05.atomic_new(path,lambda f: f.write(payload.encode('utf-8')))
    else:raise ValueError('BAD_FILE_WRITE_KIND')


def run_status(cfg,cfg03,cfg05,stage,resume):
    out=Path(cfg['output_dir']);project=Path(cfg['project_dir']);start=time.perf_counter()
    checked=verify_frozen(cfg)
    manifest={'task':TASK,'prereg_sha256':PREREG,'script_sha':sha(__file__),
              'config_sha':sha(cfg['_config_file']), 'inputs':checked}
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if stage=='inspect':
        check(not out.exists(),'STOP_OUTPUT_ALREADY_EXISTS')
        print(f'T07D_INSPECT_PASS frozen_inputs={len(checked)} strict_records=518 '
              'segments=FROM_FROZEN_T05 cache dataset=552 cpu_only=YES '
              'T03_T05_REUSE=YES no_outputs_written=YES')
        return 0
    if stage=='prepare':
        check(not out.exists(),'STOP_OUTPUT_ALREADY_EXISTS')
        out.mkdir(parents=True,exist_ok=False)
        save_new(out/'run_identity.json',{**manifest,'identity_sha256':identity})
        fw,segments,prep_s=prepare_features(cfg,cfg05)
        save_new(out/'features_w30_v2.parquet',fw,'parquet')
        save_new(out/'segments_v2.parquet',segments,'parquet')
        save_new(out/'prep_runtime.json',{'seconds':prep_s,
                    'missing_features_w30':{c:int(fw[c].isna().sum()) for c in COLUMNS},
                    'missing_features_segments':{c:int(segments[c].isna().sum()) for c in COLUMNS},
                    'rows_w30':len(fw),'rows_segments':len(segments),
                    'features_sha256':sha(out/'features_w30_v2.parquet'),
                    'segments_sha256':sha(out/'segments_v2.parquet')})
        print(f'T07D_PREP_PASS records=552 segment_rows={len(segments)} elapsed_s={prep_s:.2f}')
        return 0
    check(out.is_dir() and (out/'run_identity.json').is_file(),'RUN_PREPARE_FIRST')
    saved=json.loads((out/'run_identity.json').read_text())
    check(saved['identity_sha256']==identity,'RUN_IDENTITY_CHANGED_STOP')
    check((out/'features_w30_v2.parquet').is_file() and (out/'segments_v2.parquet').is_file(),
          'FEATURE_PREPARATION_INCOMPLETE_STOP')
    prep_state=json.loads((out/'prep_runtime.json').read_text())
    check(sha(out/'features_w30_v2.parquet')==prep_state['features_sha256'] and
          sha(out/'segments_v2.parquet')==prep_state['segments_sha256'],
          'PREPARED_FEATURE_SHA_CHANGED')
    data,seg,features,segments=prepare_run_context(cfg,cfg05)
    selected=t05.select_l3_global(data,cfg05,out)
    t05_fn,_=t05.contexts(cfg05,data,seg,selected)
    if stage=='run':
        if not resume:check(not (out/'units').exists(),'STOP_UNITS_EXIST_USE_RESUME')
        (out/'units').mkdir(exist_ok=True)
        units=list(all_units());done=0;reused=0
        for key in units:
            _,meta,reuse=train_one_unit(out,data,seg,cfg03,cfg05,selected,key,identity,t05_fn)
            done+=1;reused+=int(reuse)
            if done%5==0:print(f'T07D_UNITS={done}/{len(units)} reused={reused} last={unit_name(*key)}',flush=True)
        print(f'T07D_RUN_PASS complete={done}/{len(units)} reused={reused} no_nan=YES')
        return 0
    check(stage=='finalize','INVALID_STAGE')
    check(not (out/'summary.json').exists() and not (out/'comparisons.csv').exists(),
          'STOP_FINAL_OUTPUT_ALREADY_EXISTS')
    check(len(list((out/'units').glob('*.npz')))==len(list(all_units())),
          'UNIT_COUNT_INCOMPLETE')
    predictions=aggregate_units(out,data,seg,cfg03,cfg05,selected,identity,t05_fn)
    for protocol,frame in predictions.items():
        name='oof_'+protocol+'.parquet'
        check(not (out/name).exists(),f'STOP_OOF_ALREADY_EXISTS {name}')
        save_new(out/name,frame,'parquet')
    comparisons,results=compare_results(cfg,data,seg,predictions,project)
    save_new(out/'comparisons.csv',comparisons,'csv')
    prep=json.loads((out/'prep_runtime.json').read_text())
    runtime_units=sum(json.loads(str(np.load(p,allow_pickle=False)['meta_json'].item()))['runtime_s']
                      for p in (out/'units').glob('*.npz'))
    result={'task':TASK,'prereg_sha256':PREREG,'status':'PASS','cpu_only':True,
            'analysis':'DESCRIPTIVE_NOT_HOLM','N_eligible':518,
            'n_units':len(list(all_units())),'missing_features':prep,
            'comparisons':results,'runtime_s':{'prepare':prep['seconds'],
                          'units_total':runtime_units,'finalize':time.perf_counter()-start},
            'input_sha8':{k:v[:8] for k,v in checked.items()},
            'script_sha8':sha(__file__)[:8]}
    save_new(out/'summary.json',result)
    digests={p.name:sha(p)[:8] for p in out.iterdir() if p.is_file()}
    save_new(out/'OUTPUT_SHA256.txt','\n'.join(f'{sha(p)}  {p.relative_to(out).as_posix()}'
             for p in sorted(out.rglob('*')) if p.is_file() and p.name!='OUTPUT_SHA256.txt')+'\n',kind='text')
    print_report(cfg,out,result,comparisons,digests)
    return 0


def print_report(cfg,out,result,comp,digests):
    x=result['comparisons']; lines=[
        '=== REPORT P21-T07d ===','status=PASS; MODE=SENSITIVITY; CPU_ONLY=YES; HOLM=NO',
        'PREREG_v1.6_SHA256='+PREREG,
        'T02_T03_T04_T05_T07c_FROZEN_SHA=PASS',
        f'FEATURE_NAN_W30={result["missing_features"]["missing_features_w30"]}',
        f'FEATURE_NAN_SEGMENTS={result["missing_features"]["missing_features_segments"]}',
        'MODEL_AUROC_CI='+json.dumps(x['model_metrics'],separators=(',',':')),
        'PAIRED_COMPARISONS='+comp.loc[~comp.analysis.eq('LEAKAGE')].to_json(orient='records'),
        'LEAKAGE_V2='+json.dumps(x['leakage'],separators=(',',':')),
        'LEAKAGE_ORIGINAL_T05='+json.dumps(x['original_T05_leakage'],separators=(',',':')),
        'E2_V2='+json.dumps(x['E2_v2'],separators=(',',':')),
        f'RUNTIME={result["runtime_s"]}',
        'SHA8='+json.dumps(digests,sort_keys=True),
        f'UNITS_COMPLETE={result["n_units"]}/{result["n_units"]}; no_nan_predictions=YES; original_unchanged=YES',
        '=== END REPORT P21-T07d ===']
    check(len(lines)<=30,'REPORT_LINE_COUNT_EXCEEDED')
    print('\n'.join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='P21_CTG_Acidemia/configs/t07d.yaml')
    p.add_argument('--stage',choices=['inspect','prepare','run','finalize'],required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    cfg,cfg03,cfg05=read_config(a.config)
    cfg['_config_file']=str(Path(a.config).resolve())
    check(a.stage=='run' or not a.resume,'RESUME_ONLY_VALID_FOR_RUN')
    return run_status(cfg,cfg03,cfg05,a.stage,a.resume)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        print(f'T07D_ABORT={type(e).__name__}: {e}',file=sys.stderr,flush=True)
        sys.exit(2)

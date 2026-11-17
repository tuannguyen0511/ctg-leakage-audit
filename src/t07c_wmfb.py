"""P21-T07c (PREREG v1.6). WMFB external validation and CTU stage-1 extraction.

No pH or CTU labels accessed. FHRMA, T02 and T07b code and outputs are read-only.
Allowed new artifacts: outputs/t07c_wmfb, data_proc/.../wmfb.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat, savemat

from P21_CTG_Acidemia.src import t07b_fhrma_validate as t07b

PREREG = 'df06f9dd107ffb5b4132af3ade17aa9fec6c71d3f0b8154803c8ddfc3c59a557'
BASELINE_T07B_SHA8 = {'metrics.json': 'fc6531ea', 'per_record.csv': '6a9e471e'}
T07B_SCRIPT_SHA256 = '06ab7db216155585b2a55fb60e60f098e79d53cacfbe738ddacc7f1058d0aa9f'
EXPERT_MAT_SHA8 = '9f8230e3'
MATLAB_REPO_COMMIT = '407a279ea55458b0512fd709646c5646dc2ef669'
CONFIG_FIELDS = {'task', 'prereg_v1_6_sha256','root','fhrma_repo','ctu_raw',
                 'ctu_proc','out','expected_record_count','fs_hz','boot_replicates','boot_seed'}


def sha(path: Path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def ensure(condition, msg):
    if not condition: raise RuntimeError(msg)


def read_cfg(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    ensure(set(cfg)==CONFIG_FIELDS,'T07C_CONFIG_SCHEMA_MISMATCH')
    for k,v in {'task':'P21-T07c','prereg_v1_6_sha256':PREREG,
                'expected_record_count':552,'fs_hz':4,
                'boot_replicates':2000,'boot_seed':2026}.items():
        ensure(cfg[k]==v,f'T07C_FROZEN_CONFIG_MISMATCH {k}')
    return {k:Path(v) if k in ('root','fhrma_repo','ctu_raw','ctu_proc','out') else v
            for k,v in cfg.items()}


def locked_t02(cfg):
    """Lock the original T02 metadata source, avoiding even reading CTU pH labels."""
    orig=cfg['root']/'P21_CTG_Acidemia/configs/t02.yaml'
    ensure(sha(orig)==t07b.EXPECTED_CFG_SHA,'T02_CONFIG_SHA_MISMATCH')
    lock=cfg['root']/'P21_CTG_Acidemia/reports/T02_FROZEN_SHA.txt'
    ensure(lock.is_file(),'MISSING_T02_FROZEN_SHA')
    # T02 lock applies to stage-1 metadata via clinical.parquet. We do not
    # open clinical.parquet, labels.parquet or any phenotype/label columns.
    # Verify locked processed labels presence/byte SHA only, not values.
    proc=cfg['ctu_proc']
    checks={}
    entries={}
    for line in lock.read_text(encoding='utf-8-sig').splitlines():
        s=line.strip()
        if not s or s.startswith('#'): continue
        pieces=s.split()
        if len(pieces)>=2 and len(pieces[0]) in (8,64) and all(c in '0123456789abcdefABCDEF' for c in pieces[0]):
            file=' '.join(pieces[1:]).lstrip('*').replace('\\','/').lstrip('./')
            if file.startswith('P21_CTG_Acidemia/'):
                file=file[len('P21_CTG_Acidemia/'):]
            entries[file]=pieces[0].lower()
    for n in ('labels.parquet','split.csv'):
        candidates=(n,f'data_proc/ctu-uhb-ctgdb/1.0.0/{n}',f'outputs/t02_inputs/{n}')
        vals=[entries[x] for x in candidates if x in entries]
        ensure(vals,f'MISSING_T02_LOCK_ENTRY {n}')
        target=proc/n
        ensure(target.is_file(),f'MISSING_T02_FILE {n}')
        got=sha(target)
        ensure(all(got.startswith(x) for x in vals),f'T02_FROZEN_SHA_MISMATCH {n}')
        checks[n]=got
    return checks


def check_frozen(cfg):
    ensure(sha(cfg['root']/'P21_CTG_Acidemia/src/t07b_fhrma_validate.py')==T07B_SCRIPT_SHA256,
           'T07B_VALIDATOR_SHA_CHANGED')
    ensure((cfg['fhrma_repo']/'.git').is_dir(),'FHRMA_NOT_CLONED')
    import subprocess
    head=subprocess.run(['git','-C',str(cfg['fhrma_repo']),'rev-parse','HEAD'],
                        capture_output=True,text=True,check=True).stdout.strip()
    ensure(head==MATLAB_REPO_COMMIT,'FHRMA_GIT_COMMIT_CHANGED')
    dirty=subprocess.run(['git','-C',str(cfg['fhrma_repo']), 'status','--porcelain','--untracked-files=no'],
                         capture_output=True,text=True,check=True).stdout.strip()
    ensure(not dirty,'FHRMA_TRACKED_FILES_MODIFIED: toolbox not read-only')
    exp=cfg['fhrma_repo']/'FHRMAdataset/analyses/expertAnalyses.mat'
    ensure(sha(exp).startswith(EXPERT_MAT_SHA8),'FHRMA_EXPERT_SHA_CHANGED')
    for rel,expected in BASELINE_T07B_SHA8.items():
        p=cfg['root']/'P21_CTG_Acidemia/outputs/t07_fhrma'/rel
        ensure(p.is_file() and sha(p).startswith(expected),f'T07B_OUTPUT_SHA_MISMATCH {rel}')
    orig=cfg['root']/'P21_CTG_Acidemia/src/ctg_features.py'
    ensure(sha(orig)==t07b.EXPECTED_FEATURE_SHA,'CTG_FEATURES_SHA_CHANGED')
    t02=locked_t02(cfg)
    records_path=cfg['ctu_raw']/'RECORDS'
    ensure(records_path.is_file(),'CTU_RECORDS_MISSING')
    names=[s.strip() for s in records_path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    ensure(len(names)==552 and len(set(names))==552,'CTU_RECORDS_552_MISMATCH')
    for rid in names:
        ensure(rid.isascii() and rid.replace('_','').replace('-','').isalnum(),f'UNSAFE_RECORD_ID {rid}')
    train,test=t07b.get_inventory(cfg['fhrma_repo'])
    ensure(len(train)==66 and len(test)==90,'FHRMA_COUNTS_CHANGED')
    return names,t02


def locations(cfg):
    return (cfg['ctu_proc']/'wmfb',cfg['out'])


def stage_inspect(cfg):
    names,t02=check_frozen(cfg)
    w,out=locations(cfg)
    ensure(not w.exists() and not out.exists(),'STOP_TARGET_EXISTS: stage inspect refuses existing outputs')
    print(f'T07C_INSPECT_PASS FHRMA_commit={MATLAB_REPO_COMMIT[:8]} '
          f'T07B_locked=PASS T02_frozen=PASS CTU_records={len(names)} '
          f'TARGETS_NOT_EXIST=PASS no_PH_access=YES')
    return 0


def raw_stage1(record):
    """Exactly T02 stage-2 cutoff, but NO T02 cleaning on FHR."""
    ensure(float(record.fs)==4.0,'CTU_FS_NOT_4HZ')
    channels={str(c).upper():i for i,c in enumerate(record.channel_names)}
    ensure('FHR' in channels,'CTU_FHR_CHANNEL_MISSING')
    raw=np.asarray(record.signal[:,channels['FHR']],dtype=np.float64).reshape(-1)
    stage=record.clinical.get('Pos. II.st.')
    try: stage=float(stage)
    except (TypeError,ValueError): stage=float('nan')
    end=len(raw) if not (np.isfinite(stage) and stage>=0) else int(stage)
    ensure(0<end<=len(raw),'INVALID_STAGE2_CUTOFF')
    fhr0=raw[:end].copy()
    fhr0[~np.isfinite(fhr0)]=0
    # Zero values are untouched. Do not clip, filter, impute or merge channels.
    ensure(np.isfinite(fhr0).all(),'RAW_FHR_NONFINITE_AFTER_ZERO_MAPPING')
    return fhr0,end


def save_mat_new(path, payload):
    ensure(not path.exists(),f'STOP_EXISTS {path}')
    tmp=path.with_name(path.name+'.partial')
    ensure(not tmp.exists(),f'STOP_PARTIAL_EXISTS {tmp}')
    with tmp.open('wb') as f: savemat(f,payload,do_compression=True)
    os.replace(tmp,path)


def stage_export(cfg, resume=False):
    started=time.perf_counter();names,t02=check_frozen(cfg)
    w,out=locations(cfg); ins=w/'in'; outs=w/'out'
    if not resume:
        ensure(not w.exists() and not out.exists(),'STOP_OUTPUT_EXISTS')
        w.mkdir(parents=True,exist_ok=False)
        out.mkdir(parents=True,exist_ok=False)
        ins.mkdir();outs.mkdir();(out/'train_pred').mkdir()
        (out/'matlab_errors').mkdir()
        (out/'provenance.json').write_text(json.dumps({
            'task':'P21-T07c','prereg':PREREG,'repo_git_commit':MATLAB_REPO_COMMIT,
            't02_frozen_sha':t02, 't07b_reference_sha8':BASELINE_T07B_SHA8,
            'python_sha256':sha(Path(__file__))},indent=2)+'\n')
    else:
        ensure(w.is_dir() and out.is_dir() and ins.is_dir() and outs.is_dir(),
               'RESUME_TARGETS_MISSING')
        provenance=json.loads((out/'provenance.json').read_text())
        ensure(provenance['python_sha256']==sha(Path(__file__)) and provenance['prereg']==PREREG,
               'RESUME_CODE_IDENTITY_MISMATCH')
    from common.io_wfdb import read_record
    # Re-derive all input metadata on resume; fail if staged bytes differ.
    exported=0;reused=0;failures=[];rows=[]
    for ix,rid in enumerate(names,1):
        try:
            record=read_record(cfg['ctu_raw'],rid)
            fhr0,n=raw_stage1(record)
            input_path=ins/f'{rid}.mat'
            if input_path.exists():
                doc=loadmat(input_path,variable_names=['FHR0','fs','n_stage1'])
                arr=np.asarray(doc['FHR0']).reshape(-1)
                ensure(len(arr)==n and np.array_equal(arr,fhr0),
                       f'RESUME_INPUT_MISMATCH {rid}')
                ensure(float(np.asarray(doc['fs']).item())==4 and int(np.asarray(doc['n_stage1']).item())==n,
                       f'RESUME_INPUT_METADATA_MISMATCH {rid}')
                reused+=1
            else:
                save_mat_new(input_path,{'FHR0': fhr0.astype(np.float64).reshape(1,-1),
                                        'fs':np.array([[4.]]), 'n_stage1':np.array([[n]],dtype=np.int64)})
                exported+=1
            rows.append({'record_id':rid,'n_stage1':n,'n_missing':int(np.sum(fhr0==0)),
                         'input_sha256':sha(input_path),'status':'EXPORTED'})
        except Exception as exc:
            # Do not hide input errors. Emit a single export failures report and abort.
            failures.append({'record_id':rid,'reason':f'{type(exc).__name__}: {exc}'})
        if ix%50==0:print(f'T07C_EXPORT_PROGRESS={ix}/552',flush=True)
    if failures:
        (out/'export_errors.json').write_text(json.dumps(failures,indent=2)+'\n')
        raise RuntimeError(f'CTU_EXPORT_FAILED_COUNT={len(failures)}; see export_errors.json')
    # Copy completed inventory in output area; images not present.
    with (out/'input_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=['record_id','n_stage1','n_missing','input_sha256','status'])
        wr.writeheader();wr.writerows(rows)
    (out/'export_runtime.json').write_text(json.dumps({'export_runtime_s':time.perf_counter()-started,
           'exported':exported,'reused':reused,'records':len(rows)},indent=2)+'\n')
    print(f'T07C_EXPORT_PASS records={len(rows)}/552 new={exported} reused={reused} '
          f'raw_no_t02_cleaning=YES no_ph=YES runtime_s={time.perf_counter()-started:.2f}')
    return 0


def as_2_by_n(raw, kind):
    a=np.asarray(raw,dtype=np.float64)
    if a.size==0:return np.empty((0,2),dtype=np.float64)
    if a.ndim==1 and a.size==2:a=a.reshape(2,1)
    if a.ndim!=2 or a.shape[0]!=2:
        raise ValueError(f'BAD_WMFB_EVENT_SHAPE {kind}: {a.shape}, expected (2,N)')
    intervals=a.T.copy()
    if not np.isfinite(intervals).all() or (intervals[:,0]<0).any() or (intervals[:,1]<=intervals[:,0]).any():
        raise ValueError(f'BAD_WMFB_EVENT_ENDPOINTS {kind}')
    return intervals


def matlab_output(path, n):
    doc=loadmat(path,variable_names=['baseline','accelerations','decelerations','n_stage1'])
    ensure(all(k in doc for k in ('baseline','accelerations','decelerations','n_stage1')),
           f'MATLAB_OUTPUT_SCHEMA_INVALID {path.name}')
    ensure(int(np.asarray(doc['n_stage1']).item())==n,f'MATLAB_LENGTH_META_MISMATCH {path.name}')
    b=np.asarray(doc['baseline'],dtype=np.float64).reshape(-1)
    ensure(len(b)==n and np.isfinite(b).all(),f'MATLAB_BASELINE_NAN_OR_LENGTH_MISMATCH {path.name}')
    ev={k:as_2_by_n(doc[k],k) for k in ('accelerations','decelerations')}
    for k,v in ev.items():
        ensure((v[:,1]<=n/4 + 1/4 +1e-6).all(),f'MATLAB_EVENT_OOB {k}: {path.name}')
    return b,ev


def event_missing_fraction(raw_fhr0, start_s, end_s, fs=4):
    """Fraction missing on raw 0-coded FHR, not on interpolated values."""
    n=len(raw_fhr0)
    a=int(np.round(start_s*fs));b=int(np.round(end_s*fs))
    ensure(0<=a<b<=n+1,'CTU_EVENT_OUT_OF_BOUNDS')
    a=min(a,n);b=min(b,n)
    ensure(b>a,'CTU_EVENT_ZERO_SAMPLES')
    return float(np.mean(np.asarray(raw_fhr0)[a:b]==0))


def wmfb_expert_score(rid, predfile, expert, mod,cfg,train_folder):
    """Reuse T07b validation helpers without altering the original validator."""
    fhr1,fhr2,uc=t07b.read_fhr(train_folder/f'{rid}.fhr')
    n=len(fhr1)
    reference=np.asarray(expert['baseline'],dtype=np.float64).reshape(-1)
    ensure(len(reference)==n,f'EXPERT_BASELINE_LENGTH_MISMATCH {rid}')
    baseline,events=matlab_output(predfile,n)
    feed_ok=np.ones(n,dtype=bool)
    t07b.mask_intervals(feed_ok,expert['unreliableSignal'])
    fhr=np.maximum(fhr1,fhr2)
    fhr[~feed_ok]=np.nan
    clean,_=mod.clean_signals(fhr,uc,cfg)
    valid=np.isfinite(clean)&feed_ok
    t07b.mask_intervals(valid,expert['notToAnalyse'])
    valid[max(0,n-t07b.FS*t07b.FHRMA_END_MASK_SECONDS):]=False
    valid&=np.isfinite(reference)&(reference>0)
    cnt=int(valid.sum())
    ensure(cnt>0,f'WMFB_NO_VALID_BASELINE {rid}')
    abs_sum=float(np.abs(baseline[valid]-reference[valid]).sum())
    d={'baseline_abs_sum':abs_sum,'baseline_n':cnt}
    row={'record_id':rid,'baseline_mae_bpm':abs_sum/cnt,'baseline_labeled_samples':cnt}
    for kind,field in (('accel','accelerations'),('decel','decelerations')):
        ex=t07b.expert_events_samples(expert,field,n)
        # WMFB returns a 2xN matrix with seconds in [start; end] rows.
        pred=[]
        for start,end in events[field]:
            a=int(np.round(start*4));b=int(np.round(end*4))
            ensure(0<=a<b<=n+1,f'WMFB_EVENT_OUT_OF_BOUNDS {rid} {kind}')
            pred.append((min(a,n),min(b,n)))
        pred=[(a,b) for a,b in pred if b>a]
        ex=t07b.evaluable_events(ex,valid)
        pr=t07b.evaluable_events(pred,valid)
        tp,fp,fn=t07b.match_events(pr,ex)
        d.update({f'{kind}_tp':tp,f'{kind}_fp':fp,f'{kind}_fn':fn})
        row.update({f'{kind}_tp':tp,f'{kind}_fp':fp,f'{kind}_fn':fn,
                    f'{kind}_f1':t07b.f1(tp,fp,fn)})
    return row,d


def official_metrics(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    ensure(obj.get('status')=='PASS' and obj.get('train_records')==66,
           'OFFICIAL_EVALUATEAAM_NOT_PASS')
    metrics=obj.get('median_metrics')
    ensure(isinstance(metrics,dict) and 'MADI' in metrics and 'RMSD_bpm' in metrics,
           'OFFICIAL_METRICS_INCOMPLETE')
    ensure(metrics['MADI'] is not None and metrics['RMSD_bpm'] is not None,
           'OFFICIAL_PRIMARY_METRICS_UNAVAILABLE')
    return obj


def stage_finalize(cfg):
    start=time.perf_counter();names,t02=check_frozen(cfg)
    w,out=locations(cfg)
    ensure(w.is_dir() and out.is_dir(),'NOT_EXPORTED')
    ensure(not (out/'metrics.json').exists() and not (w/'wmfb_events.parquet').exists() and
           not (w/'wmfb_baseline.npz').exists(), 'STOP_FINAL_OUTPUT_EXISTS')
    metadata=pd.read_csv(out/'input_manifest.csv',dtype={'record_id':str})
    ensure(len(metadata)==552 and metadata.record_id.is_unique and set(metadata.record_id)==set(names),
           'INPUT_MANIFEST_552_MISMATCH')
    for record in metadata.itertuples():
        ensure(sha(w/'in'/f'{record.record_id}.mat')==record.input_sha256,
               f'INPUT_SHA_CHANGED {record.record_id}')
    official=official_metrics(out/'official_eval.json')
    ensure((out/'matlab_runtime.json').is_file(), 'MATLAB_RUN_NOT_COMPLETE')
    matlab_run=json.loads((out/'matlab_runtime.json').read_text())
    export_run=json.loads((out/'export_runtime.json').read_text())
    mod,cfg02,exp,train,test,details=t07b.preflight(cfg['root'],cfg['fhrma_repo'])
    ensure(details['train_labeled_candidates']==66,'FHRMA_TRAIN_GT_UNAVAILABLE')
    pertrain=[];data=[]
    for p in train:
        r,d=wmfb_expert_score(p.stem,out/'train_pred'/f'{p.stem}.mat',
                              exp[p.name.casefold()],mod,cfg02,p.parent)
        pertrain.append(r);data.append(d)
    metrics=t07b.bootstrap(data,2000,2026)
    primary={k:metrics[k]['estimate'] for k in ('baseline_mae_bpm','accel_f1','decel_f1')}
    gate=('PASS' if primary['baseline_mae_bpm'] is not None and primary['baseline_mae_bpm']<=5
          and primary['accel_f1'] is not None and primary['accel_f1']>=.60
          and primary['decel_f1'] is not None and primary['decel_f1']>=.60 else 'FAIL')
    old=json.loads((cfg['root']/'P21_CTG_Acidemia/outputs/t07_fhrma/metrics.json').read_text())
    oldtrain=old['splits']['train']['metrics']
    # Preserve exactly as stored (no T02 re-evaluation).
    comp=[]
    for method,series in [('T02_original',oldtrain),('WMFB_FHRMA',metrics)]:
        for k in ('baseline_mae_bpm','accel_f1','decel_f1'):
            x=series[k]
            comp.append({'method':method,'metric':k,'estimate':x['estimate'],
                         'ci95_low':x['ci95_cluster_record'][0] if x['ci95_cluster_record'] else None,
                         'ci95_high':x['ci95_cluster_record'][1] if x['ci95_cluster_record'] else None})
    errors=[];baseline_arrays={};evrows=[];counts={}
    for rid in names:
        meta=metadata.loc[metadata.record_id==rid].iloc[0]
        n=int(meta.n_stage1)
        path=w/'out'/f'{rid}.mat'
        if not path.is_file():
            err=out/'matlab_errors'/f'{rid}.txt'
            errors.append({'record_id':rid,'reason':err.read_text(encoding='utf-8')[:500] if err.is_file() else 'MATLAB_OUTPUT_MISSING'})
            continue
        try:
            b,events=matlab_output(path,n)
            rawdoc=loadmat(w/'in'/f'{rid}.mat',variable_names=['FHR0'])
            raw=np.asarray(rawdoc['FHR0'],dtype=np.float64).reshape(-1)
            ensure(len(raw)==len(b),'RAW_VS_BASELINE_LENGTH_MISMATCH')
            baseline_arrays[rid]=b.astype(np.float64)
            counts[rid]={'acc':0,'dec':0}
            for typ,field in (('acc','accelerations'),('dec','decelerations')):
                for a,z in events[field]:
                    # MATLAB 0-based event seconds -> raw sample half-open [a:z].
                    missing=event_missing_fraction(raw,a,z,fs=4)
                    evrows.append({'record_id':rid,'type':typ,'start_s':float(a),
                                   'end_s':float(z),'pct_missing_raw':100*missing,
                                   'flag_missing':bool(missing>0.50)})
                    counts[rid][typ]+=1
        except Exception as exc:
            baseline_arrays.pop(rid,None)
            errors.append({'record_id':rid,'reason':f'{type(exc).__name__}: {exc}'})
            evrows=[row for row in evrows if row['record_id']!=rid]
            counts.pop(rid,None)
    ensure(len(baseline_arrays)+len(errors)==552,'CTU_OUTPUT_COMPLETENESS_INCONSISTENT')
    # Final files only once after all validation; do not silently repair failures.
    df=pd.DataFrame(evrows,columns=['record_id','type','start_s','end_s','pct_missing_raw','flag_missing'])
    df.to_parquet(w/'wmfb_events.parquet',index=False)
    np.savez_compressed(w/'wmfb_baseline.npz',**baseline_arrays)
    (out/'matlab_failures.json').write_text(json.dumps(errors,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(comp[0]));wr.writeheader();wr.writerows(comp)
    with (out/'train_per_record.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(pertrain[0]));wr.writeheader();wr.writerows(pertrain)
    summary={
        'task':'P21-T07c','prereg_v1_6_sha256':PREREG,'gate_train':gate,
        'train_records_scored':len(pertrain),'train_metrics':metrics,
        'old_T02_train_metrics':oldtrain,
        'official_evaluateaam':official,
        'optimism_warning':'WMFB was developed using this dataset; training-set estimates may be optimistic.',
        'ctu_records_expected':552,'ctu_records_success':len(baseline_arrays),
        'ctu_records_failed':len(errors),'ctu_failures':errors,
        'ctu_acc_count':int((df.type=='acc').sum()),
        'ctu_dec_count':int((df.type=='dec').sum()),
        'ctu_flag_missing_count':int(df.flag_missing.sum()) if len(df) else 0,
        'ctu_event_counts':{},
        'no_ph_used':True,'toolbox_modified':False,'no_model_training':True,
        'event_flag_definition':'raw FHR0 == 0; > 50% zeros in [start,end) samples',
        'runtime_finalize_s':time.perf_counter()-start,
        'runtime_export_s':float(export_run['export_runtime_s']),
        'runtime_matlab_s':float(matlab_run['matlab_runtime_s']),
    }
    for typ in ('acc','dec'):
        vals=np.asarray([counts.get(rid,{}).get(typ,0) for rid in names if rid in counts],dtype=np.float64)
        summary['ctu_event_counts'][typ]={'median':float(np.median(vals)) if vals.size else None,
              'q1':float(np.percentile(vals,25)) if vals.size else None,
              'q3':float(np.percentile(vals,75)) if vals.size else None}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    manifest=[]
    for folder in (out,w):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.name!='OUTPUT_SHA256.txt':
                manifest.append((sha(p),p.relative_to(cfg['root']).as_posix()))
    (out/'OUTPUT_SHA256.txt').write_text(''.join(f'{h}  {p}\n' for h,p in manifest),encoding='utf-8')
    print('=== REPORT P21-T07c ===')
    print(f'status={"PASS" if gate=="PASS" and not errors else "FAIL_OR_PARTIAL"}; gate={gate}; CPU_ONLY=YES; no_training=YES')
    print(f'PREREG_v1.6_SHA256={PREREG}; FHRMA_commit={MATLAB_REPO_COMMIT[:8]}')
    print('T02_FROZEN_SHA=PASS; T07B_FROZEN_SHA=PASS; CTG_FEATURES_UNCHANGED=PASS')
    for typ,series in [('T02',oldtrain),('WMFB',metrics)]:
        a=[]
        for k in ('baseline_mae_bpm','accel_f1','decel_f1'):
            v=series[k];c=v['ci95_cluster_record']
            a.append(f'{k}={v["estimate"]:.4f}[{c[0]:.4f},{c[1]:.4f}]')
        print(typ+': '+'; '.join(a))
    print('official_MADI={}; official_RMSD_bpm={}; official_numeric_fields={}'.format(
        official['median_metrics'].get('MADI'),official['median_metrics'].get('RMSD_bpm'),
        ','.join(official['median_metrics'].keys())))
    print('WMFB_TRAIN_GATE='+gate+'; TRAIN_ONLY_MAY_BE_OPTIMISTIC=YES')
    print(f'CTU_RECORDS={len(baseline_arrays)}/552; MATLAB_FAILED={len(errors)}; acc={summary["ctu_acc_count"]}; dec={summary["ctu_dec_count"]}; flag_missing={summary["ctu_flag_missing_count"]}')
    for typ,stats in summary['ctu_event_counts'].items():
        print(f'{typ}_per_record_median_IQR={stats["median"]}[{stats["q1"]},{stats["q3"]}]')
    print('MATLAB_FAILURES='+('NONE' if not errors else ';'.join(e['record_id'] for e in errors)))
    print(f'wmfb_events.parquet_sha8={sha(w/"wmfb_events.parquet")[:8]}')
    print(f'python_script_sha8={sha(Path(__file__))[:8]}; matlab_script_sha8={sha(Path(__file__).with_name("t07c_wmfb_matlab.m"))[:8]}')
    print(f'wmfb_baseline.npz_sha8={sha(w/"wmfb_baseline.npz")[:8]}')
    print(f'metrics.json_sha8={sha(out/"metrics.json")[:8]}')
    print(f'OUTPUT_MANIFEST_SHA8={sha(out/"OUTPUT_SHA256.txt")[:8]}')
    total_runtime=float(export_run['export_runtime_s'])+float(matlab_run['matlab_runtime_s'])+time.perf_counter()-start
    print(f'runtime_total_s={total_runtime:.2f}; runtime_finalize_s={time.perf_counter()-start:.2f}; total_files_hashed={len(manifest)}')
    print('=== END REPORT P21-T07c ===')
    return 0 if gate=='PASS' and not errors else 3


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--config',default='P21_CTG_Acidemia/configs/t07c.yaml')
    p.add_argument('--stage',choices=['inspect','export','finalize'],required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args(argv)
    cfg=read_cfg(a.config)
    if a.stage!='export' and a.resume:raise RuntimeError('RESUME_ONLY_EXPORT')
    if a.stage=='inspect':return stage_inspect(cfg)
    if a.stage=='export':return stage_export(cfg,a.resume)
    return stage_finalize(cfg)

if __name__=='__main__':
    try:rc=main()
    except Exception as exc:
        print(f'T07C_ABORT={type(exc).__name__}: {exc}',file=sys.stderr)
        traceback.print_exc()
        rc=2
    print(f'P21_T07C_EXIT_CODE={rc}',flush=True)
    sys.exit(rc)

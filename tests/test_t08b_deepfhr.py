"""Synthetic-only T08b tests; never reads clinical outcomes or frozen result files."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from P21_CTG_Acidemia.src import t08b_deepfhr as t08


def fake_records(n=25, n_images=3):
    ids=np.asarray([f'{i:04d}' for i in range(n)])
    records=np.repeat(ids,n_images)
    y=np.repeat(np.arange(n)%2,n_images)
    return ids,records,y


def fake_split(ids,rep=0):
    entries=[]
    for fold in range(5):
        for i,rid in enumerate(ids):
            entries.append({'record_id':rid,'repeat':rep,'outer_fold':fold,
                            'inner_fold':-1 if i%5==fold else 0})
    return pd.DataFrame(entries)


def test_strict_record_separation_and_coverage():
    ids,records,y=fake_records(25,3)
    folds=t08.strict_folds(fake_split(ids),0,ids,records)
    assert len(folds)==5
    for tr,te in folds:
        assert not(set(records[tr])&set(records[te]))
    assert sum(len(te) for _,te in folds)==len(records)


def test_leaky_image_split_has_record_overlap():
    ids,records,y=fake_records(30,6)
    folds=t08.leaky_folds(y,0,records)
    assert len(folds)==5
    assert all(bool(set(records[tr])&set(records[te])) for tr,te in folds)


def test_validation_strict_record_separation():
    ids,records,y=fake_records(50,4)
    folds=t08.strict_folds(fake_split(ids),0,ids,records)
    tr,te=folds[0]
    fit,val=t08.validation_indices(tr,te,records,y,'S_img',0,0,0)
    assert not(set(records[fit])&set(records[val]))
    assert not(set(records[te])&set(records[fit]))
    assert not(set(records[te])&set(records[val]))


def test_image_split_validation_is_image_disjoint():
    ids,records,y=fake_records(40,4)
    folds=t08.leaky_folds(y,1,records)
    tr,te=folds[0]
    fit,val=t08.validation_indices(tr,te,records,y,'L_img',1,0,1)
    assert not set(fit)&set(val)
    assert not set(fit)&set(te)
    assert not set(val)&set(te)


def test_permutation_is_record_level():
    ids,records,y=fake_records(30,3)
    labels=pd.DataFrame({'record_id':ids,'y705':np.arange(len(ids))%2}).set_index('record_id')
    perm=t08.make_permuted_y(labels,records,42)
    assert len(perm)==len(records)
    assert all(len(np.unique(perm[records==rid]))==1 for rid in ids)
    assert int(perm.sum())==int(y.sum())


def test_expected_total_units_and_names():
    plan=list(t08.all_units())
    assert len(plan)==185 and len(set(plan))==185
    assert len([u for u in plan if u[0]=='NC_L_img'])==5
    assert len([u for u in plan if u[0]!='NC_L_img'])==180
    assert {(x[2],x[3]) for x in plan if x[0]!='NC_L_img'} == {(r,s) for r in range(3) for s in range(3)}
    assert plan[0]==('S_img','y705',0,0,0)
    assert len(set(str(t08.unit_file('/tmp/abc',k)) for k in plan))==185


def test_cwt_peak_frequency_known_sine():
    pywt=pytest.importorskip('pywt')
    cfg={'fs_hz':4,'frequencies_hz':[0.003,1.0],'scales':64}
    scales,freq=t08.scales_and_frequencies(cfg)
    t=np.arange(8192)/4.0
    sine=np.sin(2*np.pi*0.10*t)
    coef,actual=pywt.cwt(sine,scales,'morl',sampling_period=.25,method='fft')
    energy=np.mean(np.abs(coef[:,1500:-1500])**2,axis=1)
    peak=float(freq[np.argmax(energy)])
    assert abs(peak-0.10)<0.025
    assert np.allclose(actual,freq,atol=1e-12)


def test_segment_fraction_and_no_overlap():
    cfg={'fs_hz':4,'segment_minutes':10,'max_missing_fraction':0.5}
    x=np.full(6*2400,140.0)
    x[0:1440]=np.nan
    x[2400:2400+1200]=np.nan
    segments=list(t08.segment_fhr(x,cfg))
    assert len(segments)==6
    assert bool(segments[0][2]) is False
    assert bool(segments[1][2]) is True
    assert segments[0][1]==pytest.approx(0.6)
    assert segments[1][1]==pytest.approx(0.5)
    for j,frac,keep,seg in segments:
        assert seg.shape==(2400,) and np.isfinite(seg).all()


def test_weighted_auc_identical_to_replicated_samples():
    y=np.array([0,1,0,1,1,0])
    p=np.array([0.10,0.90,0.50,0.50,0.20,0.30])
    weights=np.array([2,4,0,1,3,2])
    assert t08.roc_weighted(y,p,weights)==pytest.approx(
        roc_auc_score(np.repeat(y,weights),np.repeat(p,weights)),abs=1e-13)


def test_bootstrap_uses_record_cluster_and_is_paired():
    ids,records,y=fake_records(40,2)
    p=np.where(y==1,0.7,0.3)
    imgids=np.asarray([f'{rid}-{i}' for rid in ids for i in range(2)])
    base=pd.DataFrame({'image_id':imgids,'record_id':records,'y':y,'p':p})
    left=base.copy()
    right=base.copy();right['p']=right['p']*0.5+0.25
    result,rep=t08.bootstrap_pair(left,right,n_resamples=80,seed=42)
    assert rep.shape==(80,6)
    assert result['point']['delta_image']==pytest.approx(0)
    assert result['point']['delta_record']==pytest.approx(0)
    assert result['ci95']['delta_image']==pytest.approx([0,0])


def test_atomic_file_refuses_overwrite(tmp_path):
    dst=tmp_path/'lock.json'
    t08.atomic_json(dst,{'x':2})
    with pytest.raises(ValueError,match='overwrite'):
        t08.atomic_json(dst,{'x':3})
    assert json.loads(dst.read_text())=={'x':2}


def test_manifest_parser_sha8_and_sha256sum():
    manifest=('abcdef12 data_proc/ctu-uhb-ctgdb/1.0.0/labels.parquet\n'
              + '1'*64+' *data_proc/ctu-uhb-ctgdb/1.0.0/split.csv\n'
              + '#comment\n')
    out=t08.manifest_sha_records(manifest)
    assert out['data_proc/ctu-uhb-ctgdb/1.0.0/labels.parquet']=='abcdef12'
    assert out['data_proc/ctu-uhb-ctgdb/1.0.0/split.csv']=='1'*64


def test_cpu_cnn_shape():
    if t08.torch is None:
        pytest.skip('PyTorch missing')
    model=t08.CWT2DCNN()
    x=t08.torch.rand(3,1,64,64)
    model.eval()
    with t08.torch.no_grad():
        y=model(x)
    assert tuple(y.shape)==(3,)
    assert sum(isinstance(k,t08.nn.Conv2d) for k in model.modules())==4


def test_input_qc_fixed_config():
    config=yaml_config()
    t08.fixed_config(config)
    config['dropout']=0.4
    with pytest.raises(ValueError,match='dropout'):
        t08.fixed_config(config)


def yaml_config():
    import yaml
    return yaml.safe_load((Path(__file__).parents[1]/'configs/t08b.yaml').read_text())


def test_manifest_csv_path_hash():
    got=t08.manifest_sha_records('data_proc/ctu-uhb-ctgdb/1.0.0/split.csv,deadbeef\n')
    assert got['data_proc/ctu-uhb-ctgdb/1.0.0/split.csv']=='deadbeef'


def test_resume_one_unit_preserves_predictions(tmp_path, monkeypatch):
    ids,records,y=fake_records(50,3)
    folds=t08.leaky_folds(y,0,records)
    images=np.zeros((len(records),1,64,64),dtype=np.float32)
    imageids=np.asarray([f'{rid}_{j}' for rid in ids for j in range(3)])
    cfg=yaml_config()
    source={'script_sha256':'a'*64,'config_sha256':'b'*64}
    hashes={'labels.parquet':'c'*64,'split.csv':'d'*64}
    calls=[]
    def dummy_train(images,y,outer_train,fit,val,test,seed,cfg):
        calls.append(1)
        return np.full(len(test),0.5,dtype=float),{
            'best_epoch':1,'runtime_s':.01,'train_auroc':0.8,
            'test_probability_sd':0.0,'fold_normalization':{'train_mean':0.0,'train_std':1.0}}
    monkeypatch.setattr(t08,'train_one',dummy_train)
    key=('L_img','y705',0,0,0)
    p,info,was_resume=t08.run_unit(key,tmp_path,images,imageids,records,y,folds,source,hashes,cfg)
    assert was_resume is False
    p2,info2,was_resume=t08.run_unit(key,tmp_path,images,imageids,records,y,folds,source,hashes,cfg)
    assert was_resume is True
    assert np.array_equal(p,p2) and len(calls)==1


def test_cpu_train_and_early_stopping_synthetic(monkeypatch):
    if t08.torch is None:
        pytest.skip('PyTorch missing')
    # Test the CNN/optimizer independently of PyWavelets's CWT implementation.
    # The separate CWT sine-frequency test is mandatory on PC-A.
    if t08.pywt is None:
        monkeypatch.setattr(t08,'pywt',object())
    x=np.random.default_rng(4).random((48,1,64,64),dtype=np.float32)
    y=np.tile(np.asarray([0,1],dtype=np.int64),24)
    fit=np.arange(0,32);val=np.arange(32,40);test=np.arange(40,48)
    cfg=yaml_config().copy()
    cfg['max_epochs']=2
    cfg['patience']=1
    p,info=t08.train_one(x,y,np.arange(0,40),fit,val,test,0,cfg)
    assert len(p)==len(test) and np.isfinite(p).all()
    assert all(0<=v<=1 for v in p)
    assert 1<=info['best_epoch']<=info['epoch_stop']<=2
    assert info['n_train_images']==32


def test_all_grid_2_protocol_2_label_9_combinations_5_folds():
    from collections import Counter
    grid=[u for u in t08.all_units() if u[0]!='NC_L_img']
    assert len(grid)==180
    counts=Counter((proto,label,repeat,seed) for proto,label,repeat,seed,fold in grid)
    assert len(counts)==2*2*3*3 and set(counts.values())=={5}
    assert set(fold for _,_,_,_,fold in grid)==set(range(5))


def test_zscore_uses_outer_train_only_and_no_test_statistics():
    x=np.zeros((10,1,2,2),dtype=np.float32)
    x[0:4]=np.asarray([1,3,5,7],dtype=np.float32)[:,None,None,None]
    x[4:8]=1000
    x[8:]=1e9
    outer=np.arange(8)
    fit=np.arange(0,4);val=np.arange(4,8);te=np.arange(8,10)
    f,v,t,info=t08.fold_train_zscore(x,outer,fit,val,te)
    expected=float(np.mean(x[outer],dtype=np.float64))
    assert info['train_mean']==pytest.approx(expected)
    assert info['n_normalization_train_images']==8
    z=np.concatenate((f,v))
    assert abs(z.mean())<1e-5 and abs(z.std()-1)<1e-5
    mutated=x.copy();mutated[te]=1e20
    _,_,_,info2=t08.fold_train_zscore(mutated,outer,fit,val,te)
    assert info2==info  # Changing test inputs cannot alter train normalization
    with pytest.raises(ValueError,match='TEST LEAKAGE'):
        t08.fold_train_zscore(x,np.arange(10),fit,val,te)


def test_zscore_strict_and_leaky_both_use_train_statistics():
    ids,records,y=fake_records(40,4)
    images=np.random.default_rng(6).normal(3,2,(len(records),1,4,4)).astype(np.float32)
    for protocol,folds in [('S_img',t08.strict_folds(fake_split(ids),0,ids,records)),
                           ('L_img',t08.leaky_folds(y,0,records))]:
        tr,te=folds[0]
        fit,val=t08.validation_indices(tr,te,records,y,protocol,0,0,0)
        xfit,xval,xtest,info=t08.fold_train_zscore(images,tr,fit,val,te)
        assert info['train_mean']==pytest.approx(float(images[tr].mean(dtype=np.float64)))
        assert info['train_std']==pytest.approx(float(images[tr].astype(float).std()))
        assert xfit.shape[0]==len(fit) and xval.shape[0]==len(val) and xtest.shape[0]==len(te)


def test_no_per_image_minmax_when_cwt_available():
    pywt=pytest.importorskip('pywt')
    cfg={'fs_hz':4,'frequencies_hz':[0.003,1.0],'scales':64}
    scales,_=t08.scales_and_frequencies(cfg)
    x=np.sin(np.linspace(0,70,2400))*20+140
    image=t08.cwt_image(x,scales,cfg)
    assert image.shape==(64,64) and np.isfinite(image).all()
    assert image.min()>=0 and image.max()>1.0
    # Invariance to constant FHR offset follows from segment mean subtraction.
    other=t08.cwt_image(x+30,scales,cfg)
    assert np.allclose(image,other,atol=1e-5)


def test_unit_file_distinguishes_repeat_and_seed():
    k0=('S_img','y705',0,1,2)
    k1=('S_img','y705',1,0,2)
    assert t08.unit_file('/tmp/abc',k0).name=='S_img_y705_s1_r0_f2.npz'
    assert t08.unit_file('/tmp/abc',k0)!=t08.unit_file('/tmp/abc',k1)


def test_assay_diagnostics_both_thresholds_are_frozen(tmp_path):
    # Test only metadata/report computation; no training or real clinical signals.
    u=list(t08.all_units())
    for i,key in enumerate(u):
        is_nc=key[0]=='NC_L_img'
        score=0.81 if (key[0]=='S_img' and key[1]=='y705') else 0.65
        if i==0: score=0.74
        p=np.array([0.2,0.4,0.5,0.7],dtype=np.float64)
        if i==1:p=np.array([0.4,0.4,0.4,0.4],dtype=np.float64)
        fn=t08.unit_file(tmp_path,key)
        t08.atomic_npz(fn,fit_info=np.asarray(json.dumps({'train_auroc':score,
                'test_probability_sd':float(p.std())})),p=p)
    d=t08.collect_assay_diagnostics(tmp_path)
    assert d['n_main_units']==180 and d['n_main_S_img_y705']==45
    assert d['S_img_y705_train_auroc_gt_0_75']==44
    assert d['near_constant_probability_count']==1
    assert d['pass'] is True
    assert d['verdict']=='assay sensitivity PASS'


def test_assay_diagnostic_counts_do_not_include_negative_control(tmp_path):
    for key in t08.all_units():
        p=np.full(4,0.5,dtype=np.float64)
        path=t08.unit_file(tmp_path,key)
        t08.atomic_npz(path,fit_info=np.asarray(json.dumps({'train_auroc':0.5,
               'test_probability_sd':float(p.std())})),p=p)
    d=t08.collect_assay_diagnostics(tmp_path)
    assert d['near_constant_probability_count']==180
    assert d['pass'] is False and d['verdict']=='assay sensitivity FAIL'


def test_gradient_clipping_invoked_at_locked_threshold(monkeypatch):
    if t08.torch is None:
        pytest.skip('PyTorch missing')
    if t08.pywt is None:
        monkeypatch.setattr(t08,'pywt',object())
    cfg=yaml_config()
    cfg['max_epochs']=1
    cfg['patience']=1
    calls=[]
    real=t08.nn.utils.clip_grad_norm_
    def checked(parameters,max_norm,*args,**kwargs):
        calls.append(max_norm)
        return real(parameters,max_norm,*args,**kwargs)
    monkeypatch.setattr(t08.nn.utils,'clip_grad_norm_',checked)
    x=np.random.default_rng(12).normal(0,1,(40,1,64,64)).astype(np.float32)
    y=np.tile([0,1],20).astype(np.int64)
    p,fit=t08.train_one(x,y,np.arange(32),np.arange(24),np.arange(24,32),
                        np.arange(32,40),0,cfg)
    assert calls and set(calls)=={1.0}
    assert np.isfinite(fit['train_auroc'])
    assert fit['test_probability_sd']==pytest.approx(float(p.std()))


def test_full_185_unit_summarization_9_pairs(monkeypatch,tmp_path):
    ids,records,y=fake_records(30,6)
    labels=pd.DataFrame({'record_id':ids,'y705':np.arange(30)%2,
                         'y715':(np.arange(30)%3==0).astype(int)}).set_index('record_id')
    split=pd.concat([fake_split(ids,rep=r) for r in range(3)],ignore_index=True)
    imageids=np.asarray([f'{rid}_s{j}' for rid in ids for j in range(6)])
    foldcache,permuted=t08.make_fold_cache(labels,split,records)
    out=tmp_path/'t08b_deepfhr'
    source={'script_sha256':'a'*64,'config_sha256':'b'*64}
    inputhashes={'labels.parquet':'c'*64,'split.csv':'d'*64}
    rgen=np.random.default_rng(12)
    for protocol,label,rep,seed,fold in t08.all_units():
        tr,te=foldcache[(protocol,label,rep)][fold]
        target=permuted if protocol=='NC_L_img' else labels.loc[records,label].to_numpy(dtype=np.int64)
        pred=rgen.uniform(0.01,0.99,len(te))
        fit,val=t08.validation_indices(np.asarray(tr),np.asarray(te),records,target,
                   'S_img' if protocol=='S_img' else 'L_img',rep,fold,seed)
        key=(protocol,label,rep,seed,fold)
        meta=t08._unit_meta(key,source,inputhashes,imageids[te].tolist(),fit,val,records)
        meta['n_train_test_record_overlap']=len(set(records[tr])&set(records[te]))
        info={'train_auroc':0.85,'test_probability_sd':float(pred.std()),
              'fold_normalization':{'train_mean':0.4,'train_std':0.2},
              'best_epoch':1,'runtime_s':0.01}
        t08.atomic_npz(t08.unit_file(out,key),image_id=imageids[te],record_id=records[te],
            y=target[te].astype(np.int8),p=pred,
            meta_json=np.asarray(json.dumps(meta,sort_keys=True)),
            fit_info=np.asarray(json.dumps(info,sort_keys=True)))
    cfg=yaml_config()
    cfg['bootstrap_resamples']=20
    fake_old={'metrics':{label:{'aggregate':{k:{'mean':0.5} for k in
                ('S_image_auroc','L_image_auroc','S_record_auroc','L_record_auroc',
                 'delta_image','delta_record')}} for label in ('y705','y715')}}
    monkeypatch.setattr(t08,'t08_reference',lambda cfg:fake_old)
    outputs={}
    monkeypatch.setattr(t08,'atomic_parquet',lambda path,df:outputs.setdefault(Path(path).name,df.copy()))
    metadata=pd.DataFrame({'record_id':ids,'n_images':[6]*30,'n_dropped':[0]*30})
    summary=t08.summarize(out,cfg,labels,records,imageids,foldcache,permuted,
                         source,inputhashes,metadata,1.0,2.0)
    assert len(summary['metrics']['y705']['by_repeat_seed'])==9
    assert len(summary['metrics']['y715']['by_repeat_seed'])==9
    for k in ('oof_S_img.parquet','oof_L_img.parquet'):
        df=outputs[k]
        assert len(df)==len(imageids)*2*9
        assert not df.duplicated(['label','repeat','seed','image_id']).any()
        assert np.isfinite(df.p).all()
    assert summary['assay_sensitivity']['n_main_units']==180
    assert summary['assay_sensitivity']['pass'] is True
    assert summary['images']['images_total']==len(imageids)

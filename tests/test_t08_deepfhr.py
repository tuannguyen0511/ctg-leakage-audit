"""Synthetic-only T08 tests; never reads clinical outcomes or frozen result files."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from P21_CTG_Acidemia.src import t08_deepfhr as t08


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
    assert len(plan)==65 and len(set(plan))==65
    assert len([u for u in plan if u[0]=='NC_L_img'])==5
    assert plan[0]==('S_img','y705',0,0)
    assert len(set(str(t08.unit_file('/tmp/abc',k)) for k in plan))==65


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
    return yaml.safe_load((Path(__file__).parents[1]/'configs/t08.yaml').read_text())


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
    def dummy_train(images,y,fit,val,test,seed,cfg):
        calls.append(1)
        return np.full(len(test),0.5,dtype=float),{'best_epoch':1,'runtime_s':.01}
    monkeypatch.setattr(t08,'train_one',dummy_train)
    key=('L_img','y705',0,0)
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
    p,info=t08.train_one(x,y,fit,val,test,0,cfg)
    assert len(p)==len(test) and np.isfinite(p).all()
    assert all(0<=v<=1 for v in p)
    assert 1<=info['best_epoch']<=info['epoch_stop']<=2
    assert info['n_train_images']==32

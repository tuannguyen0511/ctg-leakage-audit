"""T07c fixed-protocol tests, synthetic signals only; no external patient data."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib
import json
import struct
import numpy as np
import pytest
from scipy.io import savemat
from P21_CTG_Acidemia.src import t07c_wmfb as t, t07b_fhrma_validate as b


def test_fixed_prereg_and_bootstrap():
    assert len(t.PREREG)==64 and t.PREREG.startswith('df06f9dd')
    assert t.CONFIG_FIELDS and 'boot_replicates' in t.CONFIG_FIELDS
    assert t.MATLAB_REPO_COMMIT==b.EXPECTED_REPO_COMMIT


def test_cfg_schema_fixed(tmp_path):
    import yaml
    d=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/t07c.yaml').read_text())
    d['root']=str(tmp_path)
    f=tmp_path/'t07c.yaml';f.write_text(yaml.safe_dump(d))
    out=t.read_cfg(f)
    assert out['root']==tmp_path and out['boot_replicates']==2000
    d['boot_replicates']=1000;f.write_text(yaml.safe_dump(d))
    with pytest.raises(RuntimeError, match='FROZEN_CONFIG_MISMATCH'):t.read_cfg(f)


def test_stage1_excludes_only_stage2_and_preserves_values():
    record=SimpleNamespace(fs=4,channel_names=['UC','FHR'],
        signal=np.array([[30,45],[31,0],[32,np.nan],[33,234],[34,70]],dtype=float),
        clinical={'Pos. II.st.':'4'})
    arr,n=t.raw_stage1(record)
    assert n==4
    np.testing.assert_allclose(arr,[45,0,0,234])


def test_stage1_no_stage2_falls_back_to_end():
    rec=SimpleNamespace(fs=4,channel_names=['FHR'],signal=np.array([[50],[60],[70]]),
                        clinical={'Pos. II.st.':-1})
    sig,n=t.raw_stage1(rec)
    assert n==3 and sig.tolist()==[50,60,70]


def test_stage1_bad_cutoff_does_not_silent_clip():
    rec=SimpleNamespace(fs=4,channel_names=['FHR'],signal=np.array([[50],[60]]),
                        clinical={'Pos. II.st.':100})
    with pytest.raises(RuntimeError,match='INVALID_STAGE2_CUTOFF'):t.raw_stage1(rec)


def test_ctu_numerical_record_id_not_assumed_alpha():
    src=(Path(__file__).resolve().parents[1]/'src/t07c_wmfb_matlab.m').read_text()
    assert "setvartype(opts,'record_id','string')" in src


def test_events_two_by_n_seconds():
    arr=np.array([[10,50],[20,65]],dtype=float)
    assert np.allclose(t.as_2_by_n(arr,'acc'),[[10,20],[50,65]])
    assert t.as_2_by_n(np.empty((2,0)),'dec').shape==(0,2)
    with pytest.raises(ValueError,match='BAD_WMFB_EVENT_SHAPE'):
        t.as_2_by_n(np.ones((3,4)),'acc')


def test_events_invalid_endpoints_rejected():
    with pytest.raises(ValueError,match='BAD_WMFB_EVENT_ENDPOINTS'):
        t.as_2_by_n(np.array([[1.],[1.]]),'acc')
    with pytest.raises(ValueError,match='BAD_WMFB_EVENT_ENDPOINTS'):
        t.as_2_by_n(np.array([[1.],[np.nan]]),'dec')


def test_matlab_output_reads_baseline_and_sec_events(tmp_path):
    fn=tmp_path/'train01.mat'
    savemat(fn,{'baseline':np.full((1,400),125.),'n_stage1':[[400]],
                'accelerations':np.array([[10,50],[20,65]],dtype=float),
                'decelerations':np.empty((2,0))})
    baseline,events=t.matlab_output(fn,400)
    assert len(baseline)==400 and events['accelerations'].shape==(2,2)
    assert events['decelerations'].shape==(0,2)
    with pytest.raises(RuntimeError,match='MATLAB_LENGTH_META_MISMATCH'):
        t.matlab_output(fn,401)


def test_matlab_output_reject_nan(tmp_path):
    fn=tmp_path/'bad.mat'
    savemat(fn,{'baseline':[[120,np.nan]],'n_stage1':[[2]],
                'accelerations':np.empty((2,0)),'decelerations':np.empty((2,0))})
    with pytest.raises(RuntimeError,match='BASELINE_NAN'):
        t.matlab_output(fn,2)


def test_safe_mat_write_only_new(tmp_path):
    p=tmp_path/'x.mat'
    t.save_mat_new(p,{'FHR0':np.array([[1,0,3]])})
    assert t.sha(p)==hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError,match='STOP_EXISTS'):t.save_mat_new(p,{'FHR0':np.array([[2]])})


def test_official_requires_primary_metrics_and_66(tmp_path):
    p=tmp_path/'official.json'
    p.write_text(json.dumps({'status':'PASS','train_records':66,
          'median_metrics':{'MADI':0.5,'RMSD_bpm':4.2,'other_metric':1}}))
    assert t.official_metrics(p)['median_metrics']['MADI']==.5
    p.write_text(json.dumps({'status':'PASS','train_records':65,
          'median_metrics':{'MADI':0.5,'RMSD_bpm':4.2}}))
    with pytest.raises(RuntimeError,match='OFFICIAL_EVALUATEAAM_NOT_PASS'):t.official_metrics(p)


def test_wmfb_expert_score_uses_t07b_rules(tmp_path):
    cfg_path=Path(__file__).resolve().parents[1]/'configs/t02.yaml'
    import yaml
    cfg=yaml.safe_load(cfg_path.read_text())
    mod=importlib.import_module('P21_CTG_Acidemia.src.ctg_features')
    n=2400
    fhr=np.full(n,120,dtype=float);fhr[400:500]=140;fhr[1000:1100]=100
    fhr_path=tmp_path/'train01.fhr'
    with fhr_path.open('wb') as f:
        f.write(struct.pack('<I',42))
        for v in fhr:f.write(struct.pack('<HHBB',int(v)*4,0,30,0))
    mat=tmp_path/'wmfb.mat'
    savemat(mat,{'baseline':np.full(n,120),'n_stage1':[[n]],
      'accelerations':np.array([[100.],[125.]]),
      'decelerations':np.array([[250.],[275.]])})
    ann={'baseline':np.full(n,120),'accelerations':np.array([[100/60,125/60]]),
       'decelerations':np.array([[250/60,275/60]]),
       'unreliableSignal':np.empty((0,2)),'notToAnalyse':np.empty((0,2))}
    row,data=t.wmfb_expert_score('train01',mat,ann,mod,cfg,tmp_path)
    assert row['baseline_mae_bpm']==0
    assert (row['accel_tp'],row['decel_tp'])==(1,1)
    assert t07b_same(data)


def t07b_same(data):
    return b.aggregate([data])['baseline_mae_bpm']==0 and b.aggregate([data])['accel_f1']==1


def test_bootstrap_reference_2000_repeats():
    data=[{'baseline_abs_sum':1.,'baseline_n':2,
           'accel_tp':1,'accel_fp':0,'accel_fn':1,
           'decel_tp':1,'decel_fp':1,'decel_fn':0} for _ in range(66)]
    z=b.bootstrap(data,2000,2026)
    assert z['baseline_mae_bpm']['estimate']==.5
    assert z['accel_f1']['valid_replicates']==2000


def test_no_phenotype_file_parsing_in_stage_2():
    src=(Path(__file__).resolve().parents[1]/'src/t07c_wmfb.py').read_text()
    assert 'read_parquet(proc/' not in src
    assert "clinical.get('Pos. II.st.')" in src
    assert "fhr0[~np.isfinite(fhr0)]=0" in src


def test_matlab_calls_exact_toolbox_and_eval():
    src=(Path(__file__).resolve().parents[1]/'src/t07c_wmfb_matlab.m').read_text()
    assert "evaluateaam('aamwmfb(FHR)'" in src
    assert 'FHR=interpolFHR(FHR0)' in src
    assert '[baseline,accelerations,decelerations]=aamwmfb(FHR)' in src
    assert "-v7" in src


def test_refuse_existing_output_on_inspect(tmp_path,monkeypatch):
    cfg={'ctu_proc':tmp_path,'out':tmp_path/'o'}
    monkeypatch.setattr(t,'check_frozen',lambda cfg:(['1001'],{}))
    (tmp_path/'wmfb').mkdir()
    with pytest.raises(RuntimeError,match='STOP_TARGET_EXISTS'):
        t.stage_inspect(cfg)


def test_raw_missing_fraction_half_threshold():
    raw=np.array([100,0,0,100,100,100,100,100],dtype=float)
    assert t.event_missing_fraction(raw,0,1)==0.5
    assert not(t.event_missing_fraction(raw,0,1)>0.5)
    assert t.event_missing_fraction(raw,.25,.75)==1
    with pytest.raises(RuntimeError,match='CTU_EVENT_OUT_OF_BOUNDS'):
        t.event_missing_fraction(raw,-1,1)


def test_missing_fraction_raw_not_interpolated():
    raw=np.array([100,0,0,0,100,100,100,100],dtype=float)
    filled=np.interp(np.arange(len(raw)),[0,4,7],[100,100,100])
    assert t.event_missing_fraction(raw,0,1)==.75
    assert t.event_missing_fraction(filled,0,1)==0


def test_train_and_ctu_matlab_workflows_are_separate():
    s=(Path(__file__).resolve().parents[1]/'src/t07c_wmfb_matlab.m').read_text()
    assert 'preprocess(FHR1,FHR2,TOCO,item.unreliableSignal)' in s
    assert '[FHR,FHRraw,TOCO]=preprocess' in s
    assert 'FHR=interpolFHR(FHR0)' in s
    assert s.count('=aamwmfb(FHR);')>=2


def test_matlab_official_median_row_excluded():
    s=(Path(__file__).resolve().parents[1]/'src/t07c_wmfb_matlab.m').read_text()
    assert 'result(2:end)' in s
    assert 'numel(result)==67' in s

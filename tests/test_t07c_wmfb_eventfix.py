"""P21-T07c EVENTFIX. Synthetic/static checks; no patient data or MATLAB runtime."""
from pathlib import Path
import hashlib, re, ast, tempfile
import numpy as np
from scipy.io import savemat

SRC=Path(__file__).resolve().parents[1]/'src'
PROBE=SRC/'t07c_wmfb_probe_eventfix.m'
RUNNER=SRC/'t07c_wmfb_matlab_eventfix.m'
PY=SRC/'t07c_wmfb_finalize_eventfix.py'


def test_toolbox_native_3xn_is_checked_and_endpoint_only_is_saved():
    s=RUNNER.read_text(encoding='utf-8')
    assert 'size(ev,1)==3' in s
    assert 'bounds=ev(1:2,:)' in s
    assert 'all(ev(3,:)>=ev(1,:))' in s
    assert 'all(ev(3,:)<=ev(2,:))' in s
    assert 'function bounds=event_endpoints' in s
    assert s.count('event_endpoints(acc_raw,') == 2
    assert s.count('event_endpoints(dec_raw,') == 2
    assert s.count("save(dest,'baseline','accelerations','decelerations','n_stage1','-v7')") == 2


def test_empty_event_arrays_accepted_only_when_2x0_or_3x0():
    for s in [PROBE.read_text(encoding='utf-8'), RUNNER.read_text(encoding='utf-8')]:
        assert 'size(ev,2)==0' in s
        assert 'size(ev,1)==2 || size(ev,1)==3' in s
        assert 'isempty(ev)' in s


def test_probe_checks_raw_three_row_output_read_only():
    s=PROBE.read_text(encoding='utf-8')
    assert s.startswith('function t07c_wmfb_probe_eventfix()')
    assert 'aamwmfb(FHR)' in s
    assert 'size(ev,1)==3' in s
    assert 'T07C_EVENTFIX_PROBE_TRAIN_PASS' in s
    assert 'T07C_EVENTFIX_PROBE_CTU_PASS' in s
    assert not re.search(r'(?m)^\s*(?:save\(|writetable\(|fopen\(|mkdir\()',s)


def test_unmodified_preregistered_toolbox_methods_and_inputs():
    s=RUNNER.read_text(encoding='utf-8')
    assert "evaluateaam('aamwmfb(FHR)'" in s
    assert "FHR=interpolFHR(FHR0)" in s
    assert s.count(']=aamwmfb(FHR);')==2
    assert 'preprocess(FHR1,FHR2,TOCO,item.unreliableSignal)' in s
    assert 'addpath(genpath(repo))' in s
    assert "setvartype(opts,'record_id','string')" in s
    assert 'labels.parquet' not in s and 'y705' not in s and 'readtable(' in s


def test_fail_closed_and_new_failure_logs_not_overwritten():
    s=RUNNER.read_text(encoding='utf-8')
    assert 'matlab_runtime_eventfix.json' in s
    assert 'matlab_errors_eventfix' in s
    assert 'T07C:TrainRepeatedFailure' in s and 'T07C:CTURepeatedFailure' in s
    assert 'T07C:WMFBIncomplete' in s
    assert "if exist(path,'file')==2" in s
    assert 'matlab_runtime_recovery.json' not in s


def test_finalizer_uses_new_runtime_and_real_matlab_sha():
    t=PY.read_text(encoding='utf-8')
    ast.parse(t)
    assert 'matlab_runtime_eventfix.json' in t
    assert 'T07C_EVENTFIX_CTU_FAILURES' in t
    assert 'T07C_EVENTFIX_TRAIN_NOT_COMPLETE' in t
    assert 't07c_wmfb_matlab_eventfix.m' in t
    assert 'matlab_errors_eventfix' in t
    assert 'wmfb_expert_score' in t and 't07b.match_events' in t
    assert 'ctu_record' in t and 'event_serialization' in t


def test_finalizer_parses_endpoint_only_mat_file(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from P21_CTG_Acidemia.src import t07c_wmfb_finalize_eventfix as f
    p=tmp_path/'train01.mat'
    savemat(p, {'baseline':np.full((1,400),140.), 'n_stage1':[[400]],
                'accelerations':np.array([[10.,50.],[25.,70.]]),
                'decelerations':np.empty((2,0))})
    b,events=f.matlab_output(p,400)
    assert len(b)==400
    assert events['accelerations'].tolist()==[[10.,25.],[50.,70.]]
    assert events['decelerations'].shape==(0,2)


def test_finalizer_rejects_unconverted_three_row_mat_file(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from P21_CTG_Acidemia.src import t07c_wmfb_finalize_eventfix as f
    p=tmp_path/'train01.mat'
    savemat(p, {'baseline':np.full((1,400),140.), 'n_stage1':[[400]],
                'accelerations':np.array([[10.],[25.],[17.]]),
                'decelerations':np.empty((2,0))})
    import pytest
    with pytest.raises(ValueError,match='BAD_WMFB_EVENT_SHAPE'):
        f.matlab_output(p,400)


def test_frozen_train_metrics_and_no_training():
    s=PY.read_text(encoding='utf-8')
    assert "'boot_replicates':2000" in s
    assert "'prereg_v1_6_sha256':PREREG" in s
    assert 't07b.bootstrap(data,2000,2026)' in s
    assert 'T07B_SCRIPT_SHA256' in s
    assert 'no_model_training' in s

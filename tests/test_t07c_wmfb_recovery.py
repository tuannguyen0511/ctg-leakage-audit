"""Static checks of fail-closed recovery path; no patient data loaded."""
from pathlib import Path
import ast
import re

SRC=Path(__file__).resolve().parents[1]/'src'
REC=SRC/'t07c_wmfb_matlab_recover.m'
PROBE=SRC/'t07c_wmfb_probe.m'
PYREC=SRC/'t07c_wmfb_finalize_recover.py'

def test_matlab_asserts_have_message_text():
    text=REC.read_text()
    assert len(re.findall(r"'T07C:[A-Za-z0-9_]+', 'T07C check failed: ", text))==28
    assert not re.search(r"'T07C:[A-Za-z0-9_]+'\s*\)",text)

def test_no_changes_to_frozen_toolbox_commands():
    text=REC.read_text()
    assert "evaluateaam('aamwmfb(FHR)'" in text
    assert text.count('[baseline,accelerations,decelerations]=aamwmfb(FHR);')==2
    assert 'FHR=interpolFHR(FHR0)' in text
    assert "addpath(genpath(repo))" in text

def test_errors_do_not_overwrite_existing_evidence():
    text=REC.read_text()
    assert "matlab_errors_recovery" in text
    assert "if exist(path,'file')==2" in text
    assert 'T07C_PRIOR_ERROR_PRESERVED' in text
    assert "fullfile(out,'matlab_errors',[rid '.txt'])" not in text

def test_runtime_and_incomplete_fail_closed():
    text=REC.read_text()
    assert "matlab_runtime_recovery.json" in text
    assert "matlab_runtime.json'" not in text
    assert 'T07C:WMFBIncomplete' in text
    assert 'T07C:TrainRepeatedFailure' in text
    assert 'T07C:CTURepeatedFailure' in text

def test_probe_is_read_only_and_evaluates_train_and_ctu():
    text=PROBE.read_text()
    assert 'T07C_PROBE_TRAIN_PASS' in text
    assert 'T07C_PROBE_CTU_PASS' in text
    assert 'T07C_PROBE_PASS' in text
    assert 'aamwmfb(FHR)' in text
    assert not re.search(r'(?m)^\s*(?:save\(|writetable\(|fopen\(|mkdir\()',text)

def test_python_finalize_reads_new_runtime_and_checks_full_train():
    x=PYREC.read_text()
    ast.parse(x)
    assert "matlab_runtime_recovery.json" in x
    assert "T07C_RECOVERY_TRAIN_NOT_COMPLETE" in x
    assert "T07C_RECOVERY_TRAIN_FAILURES" in x
    assert "T07C_RECOVERY_CTU_ACCOUNTING_INCOMPLETE" in x
    assert "def wmfb_expert_score" in x
    assert "def stage_finalize" in x
    assert 'with_name("t07c_wmfb_matlab_recover.m")' in x
    assert "T07B_SCRIPT_SHA256" in x

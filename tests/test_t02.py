from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common.io_wfdb import CTGRecord
from common.splits import build_nested_splits, verify_nested_splits
from P21_CTG_Acidemia.src.ctg_features import build_dl_1hz, clean_signals, compute_features
from P21_CTG_Acidemia.src import t02_build_inputs


CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "t02.yaml").read_text(encoding="utf-8"))


def test_accelerations_and_prolonged_deceleration():
    fs = 4
    fhr = np.full(1800 * fs, 140.0)
    fhr[200 * fs:230 * fs] = 160.0
    fhr[500 * fs:530 * fs] = 160.0
    fhr[900 * fs:1050 * fs] = 110.0
    uc = np.full_like(fhr, 30.0)
    fhr, uc = clean_signals(fhr, uc, CFG)
    features = compute_features(fhr, uc, CFG)
    assert features["baseline"] == 140
    assert features["n_accel"] == 2
    assert features["n_decel"] == 1
    assert features["n_prolonged_decel"] == 1
    assert features["decel_area"] > 0


def test_stv_constant_fhr_is_zero():
    fhr = np.full(30 * 60 * 4, 140.0)
    uc = np.full_like(fhr, 30.0)
    features = compute_features(fhr, uc, CFG)
    assert features["stv_ms"] == 0.0
    assert features["n_valid_min"] == 30


def test_short_right_aligned_dl_and_empty_fhr():
    fhr = np.full(20 * 4, 140.0)
    uc = np.full(20 * 4, 30.0)
    X = build_dl_1hz(fhr, uc, 30, CFG)
    assert X.shape == (3, 1800)
    assert (X[1, :1780] == 1).all()
    assert (X[1, -20:] == 0).all()
    assert (X[0] == 140).all()
    assert (X[2] == 30).all()
    empty = build_dl_1hz(np.full(80, np.nan), uc, 30, CFG)
    assert (empty[0] == 0).all()
    assert (empty[1] == 1).all()
    assert (empty[2] == 0).all()


def test_nested_split_no_overlap_and_once_as_test():
    ids = np.array([f"fake_{i:03d}" for i in range(140)])
    y = np.r_[np.ones(40, dtype=int), np.zeros(100, dtype=int)]
    splits = build_nested_splits(ids, y, repeats=10, outer_folds=5, inner_folds=5, seed=2026)
    assert verify_nested_splits(splits, ids, repeats=10, outer_folds=5, inner_folds=5)
    assert len(splits) == 140 * 10 * 5
    for repeat in range(10):
        test_rows = splits[(splits["repeat"] == repeat) & (splits["inner_fold"] == -1)]
        assert len(test_rows) == 140
        assert test_rows.record_id.value_counts().eq(1).all()


def test_balance_by_repeat_for_six_and_eight_positives():
    ids = np.array([f"fake_{i:03d}" for i in range(50)])
    y = np.r_[np.ones(40, dtype=int), np.zeros(10, dtype=int)]
    full = build_nested_splits(ids, y, repeats=10, outer_folds=5, inner_folds=5, seed=2026)
    reduced = build_nested_splits(ids[10:], y[10:], repeats=10, outer_folds=5, inner_folds=5, seed=2026)
    for split, these_ids, these_y, expected in (
        (full, ids, y, 8), (reduced, ids[10:], y[10:], 6)
    ):
        selected = pd.DataFrame({"record_id": these_ids, "y705": these_y})
        assert verify_nested_splits(split, these_ids)
        stats = t02_build_inputs._outer_test_stats(split, selected)
        assert t02_build_inputs._balanced_by_repeat(stats, 10, 5)
        assert stats.positives.min() == expected == stats.positives.max()


def test_full_synthetic_audit_and_overwrite_guard(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "raw"
    derived = tmp_path / "derived"
    raw.mkdir()
    derived.mkdir()
    ids = [f"fake_{i:03d}" for i in range(50)]
    (raw / "RECORDS").write_text("\n".join(ids) + "\n", encoding="utf-8")
    ph = [7.0] * 40 + [7.3] * 10
    pd.DataFrame({"record_name": ids, "pH": ph}).to_parquet(derived / "clinical.parquet", index=False)
    pd.DataFrame({"record_name": ids, "fs": [4] * 50}).to_parquet(derived / "signal_qc.parquet", index=False)
    cfg = dict(CFG)
    cfg["raw_dir"] = str(raw)
    cfg["derived_dir"] = str(derived)
    cfg["output_root"] = str(tmp_path / "outputs")
    cfg_path = tmp_path / "t02.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def fake_read_record(directory, name):
        samples = np.column_stack((np.full(1024, 140.0), np.full(1024, 30.0)))
        if int(name[-3:]) < 10:
            samples[:768, 0] = 0.0
        return CTGRecord(samples, 4.0, ["FHR", "UC"], ["bpm", "nd"], {"Pos. II.st.": -1})

    monkeypatch.setattr(t02_build_inputs, "read_record", fake_read_record)
    monkeypatch.setattr(
        t02_build_inputs.subprocess, "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0})(),
    )
    assert t02_build_inputs.audit(cfg_path) == 0
    report = capsys.readouterr().out
    assert "status: PASS" in report
    assert "outer_test_y705_balance=PASS" in report
    assert "outer_test_y705_noexcl_balance=PASS" in report
    assert "no_record_overlap_noexcl=PASS" in report
    assert "prereg_version: 1.1" in report
    assert len(report.splitlines()) <= 30
    with np.load(derived / "dl_w30_1hz.npz", allow_pickle=False) as data:
        assert data["X"].shape == (50, 3, 1800)
        assert np.isfinite(data["X"]).all()
    assert len(pd.read_csv(derived / "split.csv")) == 40 * 10 * 5
    assert len(pd.read_csv(derived / "split_noexcl.csv")) == 50 * 10 * 5
    assert "split_noexcl.csv" in report
    assert t02_build_inputs.audit(cfg_path) == 1
    assert "no files overwritten" in capsys.readouterr().out

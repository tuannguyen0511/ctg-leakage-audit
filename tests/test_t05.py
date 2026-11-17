"""P21-T05 synthetic contract tests. No source dataset or GPU is required."""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

# Local static-test environments may lack WFDB; all production runs require it.
try:
    import wfdb  # noqa: F401
except ImportError:
    sys.modules["wfdb"] = types.SimpleNamespace(rdrecord=lambda *_a, **_kw: None)

from P21_CTG_Acidemia.src import t05_leakage as m


def test_sliding_windows_exact_count():
    signal = np.full(25 * 60 * 4, 140.0)
    ranges, dropped = m.stage1_segments(signal, np.zeros_like(signal))
    assert len(ranges) == 4
    assert dropped == 0
    assert [a for a, _, _ in ranges] == [0, 5 * 60 * 4, 10 * 60 * 4, 15 * 60 * 4]


def test_short_trace_and_partial_tail_not_used():
    signal = np.full(9 * 60 * 4, 140.0)
    assert m.stage1_segments(signal, signal) == ([], 0)
    signal = np.full((25 * 60 + 20) * 4, 140.0)
    assert len(m.stage1_segments(signal, signal)[0]) == 4


def test_segment_missingness_50_percent_boundary():
    signal = np.full(10 * 60 * 4, 140.0)
    signal[:len(signal) // 2] = np.nan
    assert len(m.stage1_segments(signal, signal)[0]) == 1
    signal[len(signal) // 2] = np.nan
    accepted, dropped = m.stage1_segments(signal, signal)
    assert len(accepted) == 0 and dropped == 1


def test_s_seg_record_grouped_outer_split():
    records = np.repeat([f"rec_{i:02d}" for i in range(20)], 3)
    train = np.flatnonzero(np.isin(records, [f"rec_{i:02d}" for i in range(15)]))
    test = np.flatnonzero(np.isin(records, [f"rec_{i:02d}" for i in range(15, 20)]))
    assert m.grouped_segment_split(records, train, test)


def test_l1_random_segment_split_creates_record_leakage():
    records = np.repeat([f"rec_{i:02d}" for i in range(40)], 4)
    labels = np.repeat([0] * 30 + [1] * 10, 4)
    folds = m.make_folds(labels, repeat=0)
    assert any(not m.grouped_segment_split(records, train, test) for train, test in folds)
    for train, test in folds:
        assert not set(train) & set(test)


def test_grouped_validation_never_mixes_records():
    records = np.repeat([f"rec_{i:02d}" for i in range(40)], 4)
    labels = np.repeat([0] * 30 + [1] * 10, 4)
    tr, val = m.group_validation_indices(records, labels, 0, 0, 0)
    assert m.grouped_segment_split(records, tr, val)
    assert len(tr) + len(val) == len(records)


def test_smote_k5_balances_while_preserving_original_rows():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 11))
    y = np.r_[[0] * 22, [1] * 8]
    augmented, yy, n = m.smote_all(x, y, k=5, seed=42)
    assert n == 14 and augmented.shape == (44, 11)
    assert int(yy.sum()) == 22
    np.testing.assert_array_equal(augmented[:30], x)
    np.testing.assert_array_equal(augmented, m.smote_all(x, y, 5, 42)[0])


def test_oversample_duplicate_rows_are_original_indices():
    y = np.r_[[0] * 22, [1] * 8]
    idx, yy, count = m.random_oversample(np.arange(30), y, 42)
    assert count == 14 and yy.sum() == 22
    assert set(idx[30:]).issubset(set(range(22, 30)))


def test_auroc_with_tied_probabilities_matches_reference():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.2, 0.7, 0.2, 0.7, 0.5, 0.5])
    assert m.auc_fast(y, p) == pytest.approx(roc_auc_score(y, p))


def test_bootstrap_known_positive_delta_and_ci():
    rng = np.random.default_rng(77)
    y = np.r_[[0] * 80, [1] * 20]
    stronger = np.where(y == 1, 0.9, 0.1)
    weaker = rng.uniform(0, 1, len(y))
    r = m.bootstrap_auc_delta(y, stronger, weaker, B=300, seed=42)
    assert r["delta"] > 0.3
    assert 0 < r["p_raw"] <= 1
    assert r["ci95"][0] > 0


def test_cluster_bootstrap_retains_record_as_sampling_unit():
    rng = np.random.default_rng(5)
    ids = np.repeat([f"r{i}" for i in range(518)], 2)
    y = np.repeat(np.r_[[1] * 30, [0] * 488], 2)
    p_s = rng.uniform(0, 1, len(y))
    p_l = np.where(y == 1, 0.9, 0.1)
    r = m.bootstrap_auc_delta(y, p_l, p_s, groups=ids, B=200)
    assert r["delta"] > 0.3 and np.isfinite(r["ci95"]).all()


def test_holm_monotonic_eight_comparisons():
    p = [0.04, 0.02, 0.001, 0.5, 0.03, 0.07, 1.0, 0.1]
    adjusted = m.holm(p)
    assert adjusted[2] == pytest.approx(0.008)
    order = np.argsort(p)
    assert np.all(np.diff(np.asarray(adjusted)[order]) >= -1e-12)
    assert len(adjusted) == 8


def test_youden_threshold_selected_only_on_supplied_fold():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold = m.youden_cutoff(y, p)
    se, sp = m.se_sp(y, p, threshold)
    assert se == sp == 1.0


def test_l4_sensitivity_bootstrap_is_paired_by_record():
    y = np.r_[[1] * 30, [0] * 488]
    strict = np.zeros(518)
    leaky = np.zeros(518)
    leaky[:30] = 1.0
    r = m.bootstrap_sensitivity_delta(y, strict, leaky, B=100, seed=42)
    assert r["delta"] == 1.0 and r["ci95"] == [1.0, 1.0]


def test_negative_control_uses_one_score_per_segment():
    frame = pd.DataFrame({"model": ["M1"] * 518, "repeat": [0] * 518,
                          "row_id": [f"r{i}_seg0" for i in range(518)],
                          "record_id": [f"r{i}" for i in range(518)],
                          "y": [1] * 30 + [0] * 488,
                          "p": np.linspace(0, 1, 518)})
    assert np.isfinite(m.negative_control(frame))


def test_atomic_artifact_refuses_to_overwrite(tmp_path):
    path = tmp_path / "unit.bin"
    m.atomic_new(path, lambda file: file.write(b"frozen"))
    with pytest.raises(ValueError):
        m.atomic_new(path, lambda file: file.write(b"modified"))
    assert path.read_bytes() == b"frozen"


def test_t04_frozen_sha_parser_ignores_metrics_hashes(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text("62a0da0e outputs/t04_dl/A_primary/oof.parquet\n"
                    "af48a9c9 outputs/t04_dl/A_primary/metrics.json\n")
    assert m.read_frozen_shas(lock) == "62a0da0e"


def test_h2_eight_comparisons_with_synthetic_repeated_oof(tmp_path, monkeypatch):
    """Exercise L1/L2/L3/L4 aggregation and Holm without training."""
    rng = np.random.default_rng(99)
    ids = np.asarray([f"r{i:04d}" for i in range(518)])
    y = np.r_[[1] * 30, [0] * 488]
    frozen_rows = []
    for repeat in range(3):
        for fold, (_, hold) in enumerate(m.make_folds(y, repeat)):
            for k in hold:
                frozen_rows.append({"record_id": ids[k], "repeat": repeat,
                                    "outer_fold": fold, "y": int(y[k]),
                                    "p_M1": 0.3 + 0.15 * y[k] + rng.uniform(-0.15, 0.15),
                                    **{f"p_M2_s{s}": 0.3 + 0.2 * y[k] +
                                       rng.uniform(-0.15, 0.15) for s in range(3)}})
    data = {"s_oof": pd.DataFrame(frozen_rows), "y": y}
    oofs = {}
    for protocol in m.PROTOCOLS:
        rows = []
        for model in m.MODELS:
            for seed in (range(3) if model == "M2" else (0,)):
                for repeat in range(3):
                    for k, rid in enumerate(ids):
                        nsegments = 2 if protocol in ("S_seg", "L1") else 1
                        for seg in range(nsegments):
                            row_id = f"{rid}__{seg}" if protocol in ("S_seg", "L1") else (
                                f"real::{rid}" if protocol == "L2" else rid)
                            record_id = rid
                            score = float(0.3 + 0.25 * y[k] + rng.uniform(-0.1, 0.1))
                            rows.append({"protocol": protocol, "model": model, "seed": seed,
                                         "repeat": repeat, "outer_fold": k % 5,
                                         "row_id": row_id, "record_id": record_id,
                                         "y": int(y[k]), "p": score})
                    if protocol == "L2":
                        for synthetic_idx in range(458):
                            tag = f"smote::{synthetic_idx:05d}"
                            rows.append({"protocol": protocol, "model": model, "seed": seed,
                                         "repeat": repeat, "outer_fold": synthetic_idx % 5,
                                         "row_id": tag, "record_id": tag,
                                         "y": 1, "p": float(rng.uniform(0.4, 0.9))})
        oofs[protocol] = pd.DataFrame(rows)
    cfg = {"bootstrap_resamples": 20, "bootstrap_seed": 42}
    result, l4 = m.h2_results(data, oofs, None, cfg)
    assert len(result) == 8
    assert result.p_holm.notna().all()
    assert len(l4) == 518 * 3 * 2
    monkeypatch.setattr(m, "sha", lambda _path: "a" * 64)
    report_cfg = {"project_dir": str(tmp_path), "negative_control_bounds": [0.4, 0.6]}
    meta = pd.DataFrame({"n_segments": [2] * 518,
                         "n_dropped_missing": [0] * 518,
                         "n_segments_total": [2] * 518})
    m.write_report(tmp_path, report_cfg, data, meta, result, 0.5,
                   {"unit_s": 1.0, "projected_m2_h": 0.05}, 100.0)
    report = (tmp_path / "reports" / "P21-T05.md").read_text()
    assert len(report.splitlines()) <= 30
    assert "H2 CONFIRMATORY: 8 tests" in report

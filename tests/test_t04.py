"""Synthetic, patient-free checks for the immutable P21-T04 protocol."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.bootstrap import discrimination_ci
from common.metrics import all_metrics, discrimination
from P21_CTG_Acidemia.src import t04_dl as t04


def test_locked_budget_full():
    decision = t04.budget_estimate(1.0, 1.0)
    assert decision["rule"] == "FULL_10_ALL"
    assert all(n == 10 for n in decision["repeats"].values())


def test_locked_budget_reduced():
    decision = t04.budget_estimate(100.0, 100.0)
    assert decision["rule"] == "REDUCED_A_B_10_C_D_E_3"
    assert list(decision["repeats"].values()) == [10, 10, 3, 3, 3]


def test_budget_hard_stop():
    decision = t04.budget_estimate(500.0, 500.0)
    assert decision["rule"] == "STOP_A_B_OVER_48H"
    assert decision["repeats"] is None


def test_budget_invalid():
    with pytest.raises(ValueError):
        t04.budget_estimate(0, 8)
    with pytest.raises(ValueError):
        t04.budget_estimate(np.nan, 8)


def test_expected_unit_counts_and_uniqueness():
    complete = {key: 10 for key in t04.CONFIGS}
    reduced = {key: (10 if key in ("A_primary", "B_noexcl") else 3)
               for key in t04.CONFIGS}
    full_units = list(t04.expected_units(complete))
    reduced_units = list(t04.expected_units(reduced))
    assert len(full_units) == 1555
    assert len(reduced_units) == 925
    assert len(set(full_units)) == len(full_units)
    assert len(set(reduced_units)) == len(reduced_units)
    assert sum(key[1] == "M2UC" for key in full_units) == 50
    assert sum(key[1] == "NC_M2" for key in full_units) == 5


def test_fixed_normalization_ignores_uc_in_main_model():
    x = np.zeros((2, 3, 12), dtype=np.float32)
    x[:, 0] = 160
    x[:, 1, :4] = 1
    x[:, 2] = 25
    two = t04.fixed_normalize(x, with_uc=False)
    three = t04.fixed_normalize(x, with_uc=True)
    assert two.shape == (2, 2, 12)
    assert three.shape == (2, 3, 12)
    np.testing.assert_array_equal(two[:, 0], 1.0)
    np.testing.assert_array_equal(two[:, 1], x[:, 1])
    np.testing.assert_array_equal(three[:, 2], 0.25)


def test_nonbinary_mask_rejected():
    x = np.zeros((2, 3, 12), dtype=np.float32)
    x[:, 1] = 0.5
    with pytest.raises(ValueError):
        t04.fixed_normalize(x)


def _synthetic_split():
    ids = np.array([f"synthetic_{i:02d}" for i in range(20)])
    result = []
    for repeat in range(10):
        for outer in range(5):
            for ix, rid in enumerate(ids):
                inner = -1 if ix % 5 == outer else ix % 5
                result.append((rid, repeat, outer, inner))
    return pd.DataFrame(result, columns=["record_id", "repeat", "outer_fold", "inner_fold"]), ids


def test_splits_no_leakage():
    frame, ids = _synthetic_split()
    t04.check_split(frame, ids)
    bad = frame.copy()
    bad.loc[0, "inner_fold"] = 100
    with pytest.raises(AssertionError):
        t04.check_split(bad, ids)


def test_train_validation_excludes_outer_test():
    ids = np.array([f"train_{i}" for i in range(40)])
    y = np.array([0] * 30 + [1] * 10)
    tr, va = t04.train_val_indices(ids, y, np.array(["outer_test"]), 0, 0, 0)
    assert len(set(ids[tr]) & set(ids[va])) == 0
    assert set(ids[tr]) | set(ids[va]) == set(ids)
    with pytest.raises(AssertionError):
        t04.train_val_indices(ids, y, np.array(["train_0"]), 0, 0, 0)


def test_known_metrics_and_record_bootstrap():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert discrimination(y, p) == (1.0, 1.0)
    metrics = all_metrics(y, p)
    assert metrics["brier"] < 0.1
    bounds = discrimination_ci(y, p, n_resamples=200, seed=42)
    assert bounds["auroc_ci95"] == [1.0, 1.0]


def test_atomic_unit_no_overwrite_and_schema(tmp_path):
    path = tmp_path / "A_primary" / "units" / "M2_s0_r00_f0.npz"
    ids = np.array(["synthetic_01", "synthetic_02"])
    y = np.array([0, 1], dtype=np.int8)
    p = np.array([0.25, 0.75])
    expected = dict(task=t04.TASK, prereg_sha256=t04.PREREG_SHA,
                    script_sha256="script", config_sha256="config",
                    config="A_primary", model="M2", seed=0, repeat=0, outer_fold=0)
    meta = {**expected,
            "test_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "no_outer_test_leakage": True}
    t04.atomic_new(path, lambda f: np.savez_compressed(
        f, record_id=ids, y=y, p=p, meta_json=np.asarray(json.dumps(meta))))
    loaded, _ = t04.read_unit(path, expected, ids, y)
    np.testing.assert_array_equal(loaded, p)
    with pytest.raises(FileExistsError):
        t04.atomic_new(path, lambda f: f.write(b"not allowed"))
    with pytest.raises(AssertionError):
        t04.read_unit(path, expected, ids[::-1], y)


def test_metric_three_seed_aggregation():
    rows = []
    for repeat in range(3):
        for ix in range(20):
            y = int(ix >= 10)
            p = 0.8 if y else 0.2
            rows.append({"record_id": f"r{ix}", "repeat": repeat,
                         "outer_fold": ix % 5, "y": y,
                         "p_M2_s0": p, "p_M2_s1": p + 0.01,
                         "p_M2_s2": p - 0.01})
    score = t04._model_metrics(pd.DataFrame(rows),
                              {"bootstrap_resamples": 100, "bootstrap_seed": 42}, "M2")
    assert score["seed0_repeat_auroc_mean"] == 1.0
    assert score["max_seed_auroc_difference"] == 0
    assert score["mean_probability_over_repeats_and_seeds"]["auroc"] == 1.0


def test_architectures_cpu_shape_when_torch_available():
    torch = pytest.importorskip("torch")
    t04.require_torch()
    with torch.no_grad():
        assert t04.make_model("M2", 2)(torch.randn(2, 2, 64)).shape == (2,)
        assert t04.make_model("M2UC", 3)(torch.randn(2, 3, 64)).shape == (2,)
        assert t04.make_model("M3", 2)(torch.randn(2, 2, 64)).shape == (2,)


def test_cpu_reproducibility_when_torch_available():
    pytest.importorskip("torch")
    # Two epochs on synthetic signals; no patient data or outer-test labels used.
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(40, 2, 64)).astype(np.float32)
    train_ids = np.asarray([f"train_{i}" for i in range(40)])
    y_train = np.array([0] * 30 + [1] * 10)
    x_test = rng.normal(size=(4, 2, 64)).astype(np.float32)
    cfg = {"batch_size": 32, "learning_rate": 0.001,
           "weight_decay": 0.0001, "max_epochs": 2, "patience": 10}
    one, _ = t04.fit_predict("M2", 0, x_train, y_train, train_ids,
                             np.array(["outer_test"]), x_test, 0, 0, cfg)
    two, _ = t04.fit_predict("M2", 0, x_train, y_train, train_ids,
                             np.array(["outer_test"]), x_test, 0, 0, cfg)
    np.testing.assert_allclose(one, two, rtol=0, atol=1e-6)

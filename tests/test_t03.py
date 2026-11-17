from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from common.bootstrap import discrimination_ci
from common.metrics import (
    all_metrics, calibration_intercept, calibration_slope,
    discrimination, sensitivity_at_specificity,
)
from common.splits import build_nested_splits
from P21_CTG_Acidemia.src.t03_baselines import (
    CONFIGS, FEATURES, _one_config, _read_tables, grid, locked_config, make_model,
    report, run, tune_and_predict,
)


CFG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "configs" / "t03.yaml").read_text(encoding="utf-8")
)


def test_known_auroc_and_auprc():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    auc, ap = discrimination(y, p)
    assert auc == pytest.approx(0.75)
    assert ap == pytest.approx(5.0 / 6.0)
    assert all_metrics(y, p)["brier"] == pytest.approx(np.mean((y - p) ** 2))


def test_calibration_intercept_and_slope():
    y = np.array([0] * 5 + [1] * 5)
    assert calibration_intercept(y, np.full(10, 0.2)) == pytest.approx(np.log(4), abs=1e-9)
    # Exact grouped empirical risks 0.2, 0.5, 0.8 identify slope ~= 1.
    p = np.repeat([0.2, 0.5, 0.8], 10)
    y = np.r_[np.r_[np.ones(2), np.zeros(8)],
              np.r_[np.ones(5), np.zeros(5)],
              np.r_[np.ones(8), np.zeros(2)]].astype(int)
    assert calibration_intercept(y, p) == pytest.approx(0.0, abs=1e-8)
    assert calibration_slope(y, p) == pytest.approx(1.0, abs=1e-4)
    assert np.isnan(calibration_slope(np.array([0, 1]), np.array([0.5, 0.5])))


def test_se_at_sp85_is_interpolated_not_fitted():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    assert sensitivity_at_specificity(y, p, 0.85) == pytest.approx(1.0)


def test_bootstrap_record_level_reproducible():
    y = np.array([0] * 16 + [1] * 8)
    p = np.linspace(0.02, 0.98, len(y))
    first = discrimination_ci(y, p, n_resamples=100, seed=42)
    second = discrimination_ci(y, p, n_resamples=100, seed=42)
    assert first == second
    assert first["bootstrap_requested"] == 100
    assert first["bootstrap_unit"] == "record"
    assert first["auroc_ci95"] == pytest.approx([1.0, 1.0])


def test_locked_grid_and_no_outer_test_leakage():
    locked_config(CFG)
    assert list(CONFIGS) == ["A_primary", "B_noexcl", "C_w60", "D_y715", "E_y720"]
    assert len(FEATURES) == 11 and "n_valid_min" not in FEATURES
    assert len(grid("M0", CFG)) == 4 and len(grid("M1", CFG)) == 12
    assert "class_weight" in make_model("M1", grid("M1", CFG)[0]).get_params()

    rng = np.random.default_rng(2026)
    x = rng.normal(size=(60, 11))
    y = np.tile([0, 1], 30)
    inner = np.repeat(np.arange(5), 12)
    ids = np.array([f"fake_{i}" for i in range(60)])
    test_ids = np.array(["heldout_1", "heldout_2"])
    x_test = rng.normal(size=(2, 11))
    cfg = dict(CFG)
    cfg["m0_c"] = [0.1]  # Isolated synthetic speed check, not the locked run.
    p1, params, score = tune_and_predict("M0", x, y, inner, x_test, cfg, ids, test_ids)
    p2, _, _ = tune_and_predict("M0", x, y, inner, x_test, cfg, ids, test_ids)
    assert np.array_equal(p1, p2)
    assert params == {"C": 0.1}
    assert 0 <= score <= 1
    with pytest.raises(AssertionError, match="Outer-test record"):
        tune_and_predict("M0", x, y, inner, x_test, cfg, ids, np.array([ids[0]]))


def test_m1_native_nan_and_reproducible_probabilities():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 11))
    x[::4, 3] = np.nan
    y = np.tile([0, 1], 40)
    params = {"learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 10}
    model_a = make_model("M1", params)
    model_b = make_model("M1", params)
    model_a.fit(x, y)
    model_b.fit(x, y)
    assert np.array_equal(model_a.predict_proba(x), model_b.predict_proba(x))
    assert np.isfinite(model_a.predict_proba(x)).all()


def test_t02_read_alignment_and_split_membership(tmp_path):
    ids = np.array([f"fake_{i:03d}" for i in range(552)])
    y705 = np.r_[np.ones(40, dtype=int), np.zeros(512, dtype=int)]
    excluded = np.zeros(552, dtype=bool)
    excluded[:10] = True
    excluded[40:64] = True
    pd.DataFrame({
        "record_id": ids, "y705": y705,
        "y715": np.r_[np.ones(113, dtype=int), np.zeros(439, dtype=int)],
        "y720": np.r_[np.ones(194, dtype=int), np.zeros(358, dtype=int)],
        "excluded": excluded,
    }).to_parquet(tmp_path / "labels.parquet", index=False)
    feat = pd.DataFrame({"record_id": ids, **{col: np.ones(552) for col in FEATURES}})
    feat["n_valid_min"] = 30
    feat.to_parquet(tmp_path / "features_w30.parquet", index=False)
    feat.to_parquet(tmp_path / "features_w60.parquet", index=False)
    cv = dict(repeats=10, outer_folds=5, inner_folds=5, seed=2026)
    build_nested_splits(ids[~excluded], y705[~excluded], **cv).to_csv(tmp_path / "split.csv", index=False)
    build_nested_splits(ids, y705, **cv).to_csv(tmp_path / "split_noexcl.csv", index=False)
    cfg = dict(CFG)
    cfg["input_dir"] = str(tmp_path)
    labels, frames, splits = _read_tables(cfg)
    assert len(labels) == 552 and len(frames["W30"].columns) == 11
    assert len(splits["split.csv"]) == 518 * 50
    assert len(splits["split_noexcl.csv"]) == 552 * 50


def test_output_directory_protection_is_declared():
    assert CFG["run_name"] == "t03_baselines"
    assert CFG["prereg_sha256"].startswith("df1d7b0c")


def test_report_fits_thirty_lines():
    metric = {
        "auroc_repeat_mean": 0.7, "auroc_repeat_sd": 0.03,
        "auprc_repeat_mean": 0.4, "auprc_repeat_sd": 0.05,
        "auroc_ci95": [0.62, 0.78], "auprc_ci95": [0.3, 0.5],
        "mean_probability_metrics": {
            "auroc": 0.71, "auprc_average_precision": 0.42, "brier": 0.13,
            "calibration_intercept_offset_logit": 0.01, "calibration_slope": 1.1,
            "sensitivity_at_specificity_0_85": 0.5, "prevalence": 0.06,
        },
    }
    params = pd.DataFrame([
        {"model": "M0", "params_json": '{"C": 0.01}'},
        {"model": "M1", "params_json": '{"learning_rate": 0.03}'},
    ])
    results = {name: {
        "runtime_s": 3.0, "negative_auroc": 0.50, "params": params,
    } for name in CONFIGS}
    metrics = {name: {"M0": metric, "M1": metric} for name in CONFIGS}
    output = report("PASS", __import__("time").perf_counter(),
                    {"pytest": True, "no_nan": True}, results, metrics, [])
    assert len(output.splitlines()) <= 30
    assert output.count("AUPRC_repeat=") == 10


def test_oof_assembly_and_negative_control_smoke(monkeypatch):
    ids = np.array([f"fake_{i:03d}" for i in range(100)])
    y = np.tile([0, 1], 50)
    labels = pd.DataFrame({"record_id": ids, "y705": y, "excluded": False}).set_index("record_id")
    frame = pd.DataFrame(np.ones((100, 11)), index=ids, columns=FEATURES)
    split = build_nested_splits(ids, y, repeats=10, outer_folds=5, inner_folds=5, seed=2026)

    def fake_tune(model, x_train, y_train, inner, x_test, cfg, train_ids, test_ids):
        assert not (set(train_ids) & set(test_ids))
        return np.full(len(test_ids), float(y_train.mean())), {"C": 0.1}, 0.5

    monkeypatch.setattr("P21_CTG_Acidemia.src.t03_baselines.tune_and_predict", fake_tune)
    result = _one_config("A_primary", CONFIGS["A_primary"], labels,
                         {"W30": frame}, {"split.csv": split}, CFG)
    assert result["oof_unique"] and result["no_nan"] and result["reproducible"]
    assert len(result["oof"]) == 100 * 10
    assert len(result["params"]) == 10 * 5 * 2
    assert result["negative_auroc"] == pytest.approx(0.5)


def test_existing_output_aborts_before_data_or_fit(tmp_path, capsys):
    cfg = dict(CFG)
    cfg["output_root"] = str(tmp_path)
    target = tmp_path / "t03_baselines"
    target.mkdir()
    guard = target / "keep.txt"
    guard.write_text("preserve", encoding="utf-8")
    cfg_path = tmp_path / "t03.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert run(cfg_path) == 1
    assert guard.read_text(encoding="utf-8") == "preserve"
    assert "no_overwrite=FAIL" in capsys.readouterr().out

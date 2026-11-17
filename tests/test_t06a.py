"""Synthetic T06a tests; do not require clinical data or model retraining."""
import numpy as np
import pandas as pd
import pytest

from P21_CTG_Acidemia.src import t06a_stats as t06


def test_paired_bootstrap_known_positive_delta():
    y = np.tile([0, 1], 30)
    perfect = y.astype(float)
    reversed_score = 1.0 - perfect
    r = t06.paired_bootstrap(y, perfect, reversed_score, B=200, seed=42,
                             descriptive=True)
    assert r["delta"] == pytest.approx(1.0)
    assert r["ci95"] == pytest.approx([1.0, 1.0])
    assert r["ci90"] == pytest.approx([1.0, 1.0])
    assert r["p_raw"] == pytest.approx(1 / 200)
    assert r["delta_auprc"] > 0
    assert all(np.isfinite(v) for v in r["ci95_brier"])


def test_paired_bootstrap_identical_scores():
    y = np.tile([0, 1], 30)
    p = np.linspace(0.1, 0.9, len(y))
    r = t06.paired_bootstrap(y, p, p, B=200, seed=42)
    assert r["delta"] == 0.0
    assert r["ci95"] == [0.0, 0.0]
    assert r["p_raw"] == 1.0


@pytest.mark.parametrize("raw,expected", [
    ([0.01, 0.03], [0.02, 0.03]),
    ([0.03, 0.01], [0.03, 0.02]),
    ([0.03, 0.03], [0.06, 0.06]),
    ([0.001, 0.9], [0.002, 0.9]),
])
def test_holm_known_p_values(raw, expected):
    assert t06.holm(raw) == pytest.approx(expected)


def test_tost_three_outcomes():
    assert t06.tost([-0.02, 0.02]) == "equivalent"
    assert t06.tost([0.06, 0.12]) == "non_equivalent"
    assert t06.tost([0.01, 0.07]) == "inconclusive"


def test_frozen_sha_parser_t04_report_format():
    hashes = ["62a0da0e", "03800e23", "bb803487", "d51b7474", "caf03970"]
    lines = [f"{name}_sha8: manifest=12345678; oof={hash}; metrics=11111111"
             for name, hash in zip(t06.NAMES, hashes)]
    assert list(t06.parse_frozen_sha("\n".join(lines)).values()) == hashes


def test_frozen_sha_parser_rejects_missing_config():
    with pytest.raises(ValueError, match="missing OOF"):
        t06.parse_frozen_sha("A_primary oof=62a0da0e")


def test_frozen_sha_parser_exact_pc_a_format_with_metrics_entries():
    """Regression: the user's lock has an OOF and metrics SHA per config."""
    oof = ["62a0da0e", "03800e23", "bb803487", "d51b7474", "caf03970"]
    metrics = ["af48a9c9", "63bbbdd1", "38303c08", "f9e24f84", "2a823c75"]
    lines = [f"{digest} outputs/t04_dl/{name}/oof.parquet"
             for name, digest in zip(t06.NAMES, oof)]
    lines += [f"{digest} outputs/t04_dl/{name}/metrics.json"
              for name, digest in zip(t06.NAMES, metrics)]
    assert t06.parse_frozen_sha("\n".join(lines)) == dict(zip(t06.NAMES, oof))
    assert t06.parse_frozen_sha("\n".join(reversed(lines))) == dict(zip(t06.NAMES, oof))


def test_frozen_sha_parser_never_accepts_metrics_instead_of_oof():
    lines = [f"{i:08x} outputs/t04_dl/{name}/metrics.json"
             for i, name in enumerate(t06.NAMES, 1)]
    with pytest.raises(ValueError, match="missing OOF"):
        t06.parse_frozen_sha("\n".join(lines))


def test_privacy_count_under_ten():
    assert t06.positive_label([0, 0, 1]) == "<10"
    assert t06.positive_label([1] * 10) == "10"


def test_e1_percent_score_auroc_unchanged_by_positive_scaling(monkeypatch):
    """Regression: T02 fhr_nan_pct_w30 is a 0–100 percentage, not a probability."""
    y = np.array([0] * 20 + [1] * 20, dtype=int)
    percent = np.linspace(0, 95, len(y))
    from sklearn.metrics import roc_auc_score

    original_ci = t06.discrimination_ci
    monkeypatch.setattr(
        t06, "discrimination_ci",
        lambda y_, p_, n_resamples=10000, seed=42:
        original_ci(y_, p_, n_resamples=50, seed=seed),
    )
    result = t06.exploratory_auc("E1", "A_primary", "fhr_nan_pct_w30", y, percent)
    assert result["estimate"] == pytest.approx(roc_auc_score(y, percent))
    assert result["ci95_low"] == pytest.approx(1.0)
    assert result["ci95_high"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside"):
        t06.exploratory_auc("E1", "A_primary", "fhr_nan_pct_w30", y,
                           np.full_like(percent, 101.0))


def test_read_oof_averages_ten_repeats_and_three_seeds(monkeypatch):
    rows_t03, rows_t04 = [], []
    for rid in range(518):
        for repeat in range(10):
            common = {"record_id": f"rec_{rid:04d}", "repeat": repeat,
                      "outer_fold": rid % 5, "y": rid % 2}
            rows_t03.append({**common, "p_M0": 0.2, "p_M1": 0.4})
            rows_t04.append({
                **common, "p_M2_s0": 0.3, "p_M2_s1": 0.4, "p_M2_s2": 0.5,
                "p_M3_s0": 0.6, "p_M3_s1": 0.7, "p_M3_s2": 0.8,
                "p_M2UC_s0": 0.45,
            })
    t03 = pd.DataFrame(rows_t03)
    t04 = pd.DataFrame(rows_t04)

    def fake_parquet(path):
        return t03.copy() if "t03_baselines" in str(path) else t04.copy()

    monkeypatch.setattr(pd, "read_parquet", fake_parquet)
    joined, means = t06.read_oof({"project_dir": "D:/test"}, "A_primary")
    assert len(joined) == 5180
    assert len(means) == 518
    assert means.p_M1.iloc[0] == pytest.approx(0.4)
    assert means.p_M2.iloc[0] == pytest.approx(0.4)
    assert means.p_M3.iloc[0] == pytest.approx(0.7)


def test_read_oof_detects_fold_mismatch(monkeypatch):
    rows = [{"record_id": str(i), "repeat": r, "outer_fold": i % 5, "y": i % 2,
             "p_M0": 0.3, "p_M1": 0.4}
            for i in range(518) for r in range(10)]
    t03 = pd.DataFrame(rows)
    t04 = t03[["record_id", "repeat", "outer_fold", "y"]].copy()
    t04.loc[0, "outer_fold"] = (int(t04.loc[0, "outer_fold"]) + 1) % 5
    for model in ("M2", "M3"):
        for seed in range(3):
            t04[f"p_{model}_s{seed}"] = 0.5
    t04["p_M2UC_s0"] = 0.5
    monkeypatch.setattr(pd, "read_parquet",
                        lambda path: t03.copy() if "t03_baselines" in str(path)
                        else t04.copy())
    with pytest.raises(ValueError, match="fold/label disagreement"):
        t06.read_oof({"project_dir": "D:/test"}, "A_primary")


def test_end_to_end_synthetic_output_and_refusal_to_overwrite(tmp_path, monkeypatch):
    """Exercise frozen checks, joins, E1-E4, and atomic output using synthetic data."""
    root = tmp_path / "project"
    source = tmp_path / "t02"
    source.mkdir()
    ids = [f"r{i:04d}" for i in range(552)]
    eligible = set(ids[:518])
    labels = pd.DataFrame({
        "record_id": ids,
        "excluded": [name not in eligible for name in ids],
        "y705": [int(i % 11 == 0) for i in range(552)],
        "y715": [int(i % 8 == 0) for i in range(552)],
        "y720": [int(i % 7 == 0) for i in range(552)],
        "fhr_nan_pct_w30": [100 * i / 552 for i in range(552)],
    })
    features = pd.DataFrame({
        "record_id": ids,
        **{feature: np.linspace(0.05, 0.95, 552) + j * 0.01
           for j, feature in enumerate(t06.t03.FEATURES)},
    })
    labels.to_parquet(source / "labels.parquet", index=False)
    features.to_parquet(source / "features_w30.parquet", index=False)
    split_rows = []
    for repeat in range(10):
        for fold in range(5):
            for i in range(518):
                split_rows.append({
                    "record_id": ids[i], "repeat": repeat, "outer_fold": fold,
                    "inner_fold": -1 if i % 5 == fold else (i // 5 + fold + repeat) % 5,
                })
    pd.DataFrame(split_rows).to_csv(source / "split.csv", index=False)

    locks = []
    for name in t06.NAMES:
        allowed = range(552) if name == "B_noexcl" else range(518)
        target = "y715" if name == "D_y715" else "y720" if name == "E_y720" else "y705"
        base, dl = [], []
        for repeat in range(10):
            for i in allowed:
                common = {
                    "record_id": ids[i], "repeat": repeat, "outer_fold": i % 5,
                    "y": int(labels.loc[i, target]),
                }
                x = i / 552
                base.append({**common, "p_M0": 0.2 + 0.3 * x,
                             "p_M1": 0.2 + 0.4 * x})
                deep = {
                    **common,
                    **{f"p_M2_s{s}": 0.15 + 0.52 * x + 0.06 * int(common["y"]) + s * 0.01
                       for s in range(3)},
                    **{f"p_M3_s{s}": 0.12 + 0.40 * x + 0.25 * int(common["y"]) + s * 0.01
                       for s in range(3)},
                }
                if name == "A_primary":
                    deep["p_M2UC_s0"] = 0.4
                dl.append(deep)
        base_path = root / "outputs" / "t03_baselines" / name
        dl_path = root / "outputs" / "t04_dl" / name
        base_path.mkdir(parents=True)
        dl_path.mkdir(parents=True)
        pd.DataFrame(base).to_parquet(base_path / "oof.parquet", index=False)
        pd.DataFrame(dl).to_parquet(dl_path / "oof.parquet", index=False)
        locks.append(f"{name}: oof={t06.sha(dl_path / 'oof.parquet')}")
    frozen = root / "reports" / "T04_FROZEN_SHA.txt"
    frozen.parent.mkdir()
    frozen.write_text("\n".join(locks) + "\n", encoding="utf-8")

    original_paired = t06.paired_bootstrap
    original_spearman = t06.spearman_bootstrap
    original_ci = t06.discrimination_ci
    monkeypatch.setattr(t06, "paired_bootstrap",
                        lambda y, a, b, B=10000, seed=42, descriptive=False:
                        original_paired(y, a, b, B=40, seed=seed, descriptive=descriptive))
    monkeypatch.setattr(t06, "spearman_bootstrap",
                        lambda a, b, B=10000, seed=42:
                        original_spearman(a, b, B=40, seed=seed))
    monkeypatch.setattr(t06, "discrimination_ci",
                        lambda y, p, n_resamples=10000, seed=42:
                        original_ci(y, p, n_resamples=40, seed=seed))
    monkeypatch.setattr(t06.subprocess, "run",
                        lambda *args, **kwargs: type("Proc", (), {"returncode": 0})())
    monkeypatch.setattr(
        t06.t03, "tune_and_predict",
        lambda model, x_train, y_train, inner, x_test, c, train_ids, test_ids:
        (np.clip(0.2 + 0.5 * x_test[:, 0], 0, 1),
         {"learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 10}, 0.6),
    )
    config = {
        "task": "P21-T06a", "run_name": "t06a_stats", "prereg_version": "1.3",
        "prereg_sha256": t06.PREREG,
        "project_dir": str(root), "input_dir": str(source),
        "frozen_hash_file": str(frozen),
        "config_names": list(t06.NAMES),
        "cv_repeats": 10, "cv_outer_folds": 5, "cv_inner_folds": 5,
        "bootstrap_resamples": 10000, "bootstrap_seed": 42,
        "alpha": 0.05, "equivalence_margin": 0.05,
        "models_h1": ["M2", "M3"],
        "exploratory_feature": "fhr_nan_pct_w30",
        "e3_feature_columns": list(t06.E3_FEATURES),
        "m1_learning_rate": [0.03, 0.1], "m1_max_depth": [2, 3],
        "m1_min_samples_leaf": [10, 20, 40], "cpu_threads": 4,
    }
    t06.run(config)
    output = root / "outputs" / "t06a_stats"
    assert all((output / name).is_file() for name in (*t06.FILES, "report.txt"))
    assert len((output / "report.txt").read_text().splitlines()) <= 30
    assert len(pd.read_csv(output / "h1_tests.csv")) == 2
    assert len(pd.read_csv(output / "sensitivity.csv")) == 8
    assert len(pd.read_csv(output / "exploratory.csv")) == 17
    assert len(pd.read_parquet(output / "e3_oof.parquet")) == 5180
    with pytest.raises(FileExistsError, match="output directory exists"):
        t06.run(config)

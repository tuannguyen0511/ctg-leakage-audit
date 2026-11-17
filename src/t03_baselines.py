"""P21-T03: one locked nested-CV run for feature-based M0 and M1.

No DL, clinical covariates, threshold fitting, or outer-test model selection.
The T02 split file assigns each record's outer and inner folds explicitly.
"""

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.bootstrap import discrimination_ci
from common.metrics import all_metrics, discrimination
from common.splits import verify_nested_splits


TASK = "P21-T03"
PREREG_SHA = "df1d7b0c3b4b0a27bc267ce6b97b05808f66e88e8530e989efa337071b079c89"
FEATURES = (
    "baseline", "stv_ms", "ltv", "pct_min_var", "n_accel", "n_decel",
    "decel_area", "n_prolonged_decel", "n_contractions", "lf_power", "hf_power",
)
CONFIGS = {
    "A_primary": {"features": "W30", "split": "split.csv", "target": "y705"},
    "B_noexcl": {"features": "W30", "split": "split_noexcl.csv", "target": "y705"},
    "C_w60": {"features": "W60", "split": "split.csv", "target": "y705"},
    "D_y715": {"features": "W30", "split": "split.csv", "target": "y715"},
    "E_y720": {"features": "W30", "split": "split.csv", "target": "y720"},
}
OUTPUT_FILES = ("oof.parquet", "params.csv", "metrics.json")


def sha8(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:8]


def locked_config(cfg):
    """Fail closed if a model, feature, grid, seed, or protocol setting drifts."""
    checks = (
        cfg["run_name"] == "t03_baselines",
        cfg["prereg_version"] == "1.1",
        cfg["prereg_sha256"] == PREREG_SHA,
        cfg["configs"] == CONFIGS,
        tuple(cfg["feature_columns"]) == FEATURES,
        list(map(float, cfg["m0_c"])) == [0.01, 0.1, 1.0, 10.0],
        list(map(float, cfg["m1_learning_rate"])) == [0.03, 0.1],
        list(map(int, cfg["m1_max_depth"])) == [2, 3],
        list(map(int, cfg["m1_min_samples_leaf"])) == [10, 20, 40],
        [int(cfg[k]) for k in ("cv_repeats", "cv_outer_folds", "cv_inner_folds")] == [10, 5, 5],
        int(cfg["bootstrap_resamples"]) == 10000,
        int(cfg["bootstrap_seed"]) == 42,
        int(cfg["negative_control_seed"]) == 42,
        list(map(float, cfg["negative_control_auroc_bounds"])) == [0.40, 0.60],
        float(cfg["specificity"]) == 0.85,
        1 <= int(cfg["n_jobs"]) <= 4,
    )
    if not all(checks):
        raise ValueError("T03 YAML differs from the locked Director specification")


def grid(model_name, cfg):
    if model_name == "M0":
        return [{"C": float(c)} for c in cfg["m0_c"]]
    if model_name == "M1":
        return [
            {"learning_rate": float(lr), "max_depth": int(depth), "min_samples_leaf": int(leaf)}
            for lr, depth, leaf in product(
                cfg["m1_learning_rate"], cfg["m1_max_depth"], cfg["m1_min_samples_leaf"]
            )
        ]
    raise ValueError("Only M0 and M1 are preregistered")


def make_model(model_name, params):
    if model_name == "M0":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=params["C"], penalty="l2", solver="lbfgs", max_iter=2000,
                class_weight="balanced",
            )),
        ])
    if model_name == "M1":
        return HistGradientBoostingClassifier(
            max_iter=200, l2_regularization=1.0, early_stopping=False,
            class_weight="balanced", random_state=42, **params,
        )
    raise ValueError("Only M0 and M1 are preregistered")


def tune_and_predict(model_name, x_train, y_train, inner_fold, x_test, cfg,
                     train_ids, test_ids):
    """Tune only on inner CV of outer-train; refit on outer-train."""
    train_ids = np.asarray(train_ids, dtype=str)
    test_ids = np.asarray(test_ids, dtype=str)
    y_train = np.asarray(y_train, dtype=np.int64)
    inner_fold = np.asarray(inner_fold, dtype=np.int64)
    if set(train_ids) & set(test_ids):
        raise AssertionError("Outer-test record entered fit or tuning")
    if len(train_ids) != len(y_train) or len(inner_fold) != len(train_ids):
        raise AssertionError("Inner-fold indices and training records differ")
    if set(inner_fold) != set(range(int(cfg["cv_inner_folds"]))):
        raise AssertionError("Inner CV must include all five validation folds")
    folds = []
    for k in range(int(cfg["cv_inner_folds"])):
        fit_mask, val_mask = inner_fold != k, inner_fold == k
        if set(train_ids[fit_mask]) & set(test_ids) or set(train_ids[val_mask]) & set(test_ids):
            raise AssertionError("Leakage between outer-test and inner CV")
        if len(np.unique(y_train[fit_mask])) != 2 or len(np.unique(y_train[val_mask])) != 2:
            raise ValueError("Inner fold contains only one class; no parameter selected")
        folds.append((np.flatnonzero(fit_mask), np.flatnonzero(val_mask)))

    best_params = None
    best_auroc = -np.inf
    for params in grid(model_name, cfg):
        inner_aurocs = []
        for fit_idx, val_idx in folds:
            estimator = make_model(model_name, params)
            estimator.fit(x_train[fit_idx], y_train[fit_idx])
            p_val = estimator.predict_proba(x_train[val_idx])[:, 1]
            if not np.isfinite(p_val).all():
                raise ValueError("Nonfinite inner-fold probabilities")
            inner_aurocs.append(discrimination(y_train[val_idx], p_val)[0])
        mean_auroc = float(np.mean(inner_aurocs))
        # Deterministic tie-breaking: retain the first candidate in YAML grid order.
        if mean_auroc > best_auroc:
            best_params, best_auroc = dict(params), mean_auroc
    if best_params is None:
        raise ValueError("Inner CV did not select parameters")

    final_estimator = make_model(model_name, best_params)
    final_estimator.fit(x_train, y_train)
    p_test = final_estimator.predict_proba(x_test)[:, 1].astype(np.float64)
    if not np.isfinite(p_test).all() or ((p_test < 0) | (p_test > 1)).any():
        raise ValueError("Nonfinite or invalid outer-test probabilities")
    return p_test, best_params, best_auroc


def _read_tables(cfg):
    root = Path(cfg["input_dir"])
    inputs = ["labels.parquet", "features_w30.parquet", "features_w60.parquet", "split.csv", "split_noexcl.csv"]
    absent = [name for name in inputs if not (root / name).is_file()]
    if absent:
        raise FileNotFoundError("Missing T02 inputs: " + ", ".join(absent))
    labels = pd.read_parquet(root / "labels.parquet")
    if len(labels) != 552 or labels.record_id.duplicated().any():
        raise ValueError("Expected 552 unique T02 labels")
    for col in ("y705", "y715", "y720", "excluded"):
        if labels[col].isna().any():
            raise ValueError("Missing target or QC flags")
    for col in ("y705", "y715", "y720"):
        if not labels[col].isin([0, 1]).all():
            raise ValueError("Labels must be binary")
    labels = labels.set_index("record_id", drop=True)

    features = {}
    for win, file_name in (("W30", "features_w30.parquet"), ("W60", "features_w60.parquet")):
        data = pd.read_parquet(root / file_name)
        if data.record_id.duplicated().any() or set(data.record_id) != set(labels.index):
            raise ValueError("Feature-to-label record alignment failed")
        if not set(FEATURES).issubset(data.columns):
            raise ValueError("T02 features missing one of the 11 locked columns")
        frame = data.set_index("record_id").loc[labels.index, list(FEATURES)]
        features[win] = frame.astype(np.float64)
        if np.isinf(features[win].to_numpy()).any():
            raise ValueError("Features contain an infinite value")

    splits = {}
    for file_name in ("split.csv", "split_noexcl.csv"):
        frame = pd.read_csv(root / file_name, dtype={"record_id": str})
        if list(frame.columns) != ["record_id", "repeat", "outer_fold", "inner_fold"]:
            raise ValueError("T02 split column format changed")
        expected = labels.index if file_name == "split_noexcl.csv" else labels.index[~labels.excluded.astype(bool)]
        if not verify_nested_splits(
            frame, expected.to_numpy(), repeats=10, outer_folds=5, inner_folds=5
        ):
            raise ValueError("T02 split identity or train/test check failed")
        splits[file_name] = frame
    if int(labels.excluded.sum()) != 34:
        raise ValueError("T02 QC exclusion count changed")
    return labels, features, splits


def _one_config(config_name, spec, labels, features, splits, cfg):
    started = time.perf_counter()
    targets = labels[spec["target"]].astype(int)
    frame = features[spec["features"]]
    split = splits[spec["split"]]
    allowed = labels.index if spec["split"] == "split_noexcl.csv" else labels.index[~labels.excluded.astype(bool)]
    allowed_set = set(allowed)
    output_rows = []
    param_rows = []
    negative_scores = pd.Series(np.nan, index=allowed, dtype=float) if config_name == "A_primary" else None
    control_rng = np.random.default_rng(int(cfg["negative_control_seed"]))
    reproducible = True
    leakage_free = True

    for repeat in range(10):
        for outer_fold in range(5):
            block = split[(split.repeat == repeat) & (split.outer_fold == outer_fold)]
            train = block.loc[block.inner_fold >= 0]
            test = block.loc[block.inner_fold == -1]
            train_ids = train.record_id.to_numpy(dtype=str)
            test_ids = test.record_id.to_numpy(dtype=str)
            if not len(train_ids) or not len(test_ids) or set(train_ids) & set(test_ids):
                raise AssertionError("Empty or overlapping outer fold")
            if set(train_ids) | set(test_ids) != allowed_set:
                raise AssertionError("Outer split does not cover all allowed records")
            x_train = frame.loc[train_ids, list(FEATURES)].to_numpy(dtype=np.float64)
            x_test = frame.loc[test_ids, list(FEATURES)].to_numpy(dtype=np.float64)
            y_train = targets.loc[train_ids].to_numpy(dtype=np.int64)
            y_test = targets.loc[test_ids].to_numpy(dtype=np.int64)
            inner = train.inner_fold.to_numpy(dtype=np.int64)

            predictions = {}
            for model_name in ("M0", "M1"):
                pred, selected_params, inner_auroc = tune_and_predict(
                    model_name, x_train, y_train, inner, x_test, cfg, train_ids, test_ids
                )
                predictions[model_name] = pred
                param_rows.append({
                    "repeat": repeat, "outer_fold": outer_fold, "model": model_name,
                    "params_json": json.dumps(selected_params, sort_keys=True),
                    "mean_inner_auroc": inner_auroc,
                })
            for ix, record_id in enumerate(test_ids):
                output_rows.append({
                    "record_id": record_id, "repeat": repeat, "outer_fold": outer_fold,
                    "y": int(y_test[ix]), "p_M0": float(predictions["M0"][ix]),
                    "p_M1": float(predictions["M1"][ix]),
                })

            if config_name == "A_primary" and repeat == 0:
                # Shuffle only outer-train labels; evaluate on untouched outer-test labels.
                y_permuted = control_rng.permutation(y_train)
                neg_pred, _, _ = tune_and_predict(
                    "M1", x_train, y_permuted, inner, x_test, cfg, train_ids, test_ids
                )
                negative_scores.loc[test_ids] = neg_pred

            if config_name == "A_primary" and repeat == 0 and outer_fold == 0:
                # Recompute both tuned estimators with identical data and seeds.
                for model_name in ("M0", "M1"):
                    repeated_pred, _, _ = tune_and_predict(
                        model_name, x_train, y_train, inner, x_test, cfg, train_ids, test_ids
                    )
                    reproducible &= np.array_equal(predictions[model_name], repeated_pred)
        print(f"{config_name}: finished repeat {repeat + 1}/10", flush=True)

    oof = pd.DataFrame(output_rows).sort_values(["repeat", "record_id"]).reset_index(drop=True)
    params = pd.DataFrame(param_rows).sort_values(["repeat", "outer_fold", "model"]).reset_index(drop=True)
    expected = len(allowed) * 10
    oof_unique = (
        len(oof) == expected and
        oof.groupby(["record_id", "repeat"]).size().eq(1).all() and
        set(oof.record_id) == allowed_set and
        oof.groupby("record_id").y.nunique().eq(1).all()
    )
    no_nan = bool(np.isfinite(oof[["p_M0", "p_M1"]].to_numpy()).all())
    negative_auroc = None
    if negative_scores is not None:
        if negative_scores.isna().any():
            raise AssertionError("Negative control omitted an outer-test record")
        negative_auroc = discrimination(targets.loc[allowed], negative_scores.loc[allowed])[0]
    return {
        "oof": oof, "params": params, "runtime_s": time.perf_counter() - started,
        "oof_unique": bool(oof_unique), "no_nan": no_nan,
        "leakage_free": bool(leakage_free), "reproducible": bool(reproducible),
        "negative_auroc": negative_auroc,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _metrics_for_config(oof, cfg):
    results = {}
    for model in ("M0", "M1"):
        col = f"p_{model}"
        per_repeat = []
        for repeat in range(10):
            rows = oof.loc[oof.repeat == repeat]
            auroc, auprc = discrimination(rows.y, rows[col])
            per_repeat.append({"repeat": repeat, "auroc": auroc, "auprc": auprc})
        means = oof.groupby("record_id", sort=True).agg(
            y=("y", "first"), p=(col, "mean"), n=(col, "size")
        )
        if not means.n.eq(10).all():
            raise AssertionError("Mean probability missing a repeated prediction")
        averaged = all_metrics(means.y.to_numpy(), means.p.to_numpy())
        ci = discrimination_ci(
            means.y.to_numpy(), means.p.to_numpy(),
            n_resamples=int(cfg["bootstrap_resamples"]), seed=int(cfg["bootstrap_seed"]),
        )
        auc_values = np.array([row["auroc"] for row in per_repeat])
        ap_values = np.array([row["auprc"] for row in per_repeat])
        results[model] = {
            "repeated_oof": per_repeat,
            "auroc_repeat_mean": float(auc_values.mean()),
            "auroc_repeat_sd": float(auc_values.std(ddof=1)),
            "auprc_repeat_mean": float(ap_values.mean()),
            "auprc_repeat_sd": float(ap_values.std(ddof=1)),
            "mean_probability_metrics": averaged,
            **ci,
            "prereg_sha256": PREREG_SHA,
        }
    return _json_safe(results)


def _most_selected(params):
    parts = []
    for model in ("M0", "M1"):
        ranked = Counter(params.loc[params.model == model, "params_json"])
        chosen, count = sorted(ranked.items(), key=lambda x: (-x[1], x[0]))[0]
        parts.append(f"{model}={chosen} ({count}/50)")
    return "; ".join(parts)


def _model_report(name, model, metrics):
    r = metrics[model]
    m = r["mean_probability_metrics"]
    auc_l, auc_u = r["auroc_ci95"]
    ap_l, ap_u = r["auprc_ci95"]
    def f(x):
        return "NA" if x is None else f"{x:.4f}"
    return (
        f"{name}_{model}: AUROC_repeat={f(r['auroc_repeat_mean'])}+/-{f(r['auroc_repeat_sd'])}; "
        f"AUPRC_repeat={f(r['auprc_repeat_mean'])}+/-{f(r['auprc_repeat_sd'])}; "
        f"AUROC_avg={f(m['auroc'])}[{f(auc_l)},{f(auc_u)}]; "
        f"AUPRC_avg={f(m['auprc_average_precision'])}[{f(ap_l)},{f(ap_u)}]; "
        f"Brier={f(m['brier'])}; cal_int={f(m['calibration_intercept_offset_logit'])}; "
        f"cal_slope={f(m['calibration_slope'])}; Se@Sp85={f(m['sensitivity_at_specificity_0_85'])}; "
        f"prevalence={f(m['prevalence'])}"
    )


def report(status, started, checks, results, metrics_by_config, warnings, hashes=None):
    lines = [
        f"=== REPORT {TASK} ===", f"status: {status}",
        "pc: PC-A CPU; OMP_NUM_THREADS=4; n_jobs=1",
        f"runtime_total_s: {time.perf_counter() - started:.2f}",
        f"script_sha256_8: {sha8(Path(__file__))}",
        f"prereg_version: 1.1; prereg_sha256_director_declared: {PREREG_SHA}",
        "T02_inputs: N=552; primary eligible=518; noexcl=552; fixed 11 features (stv_ms column)",
    ]
    for name in CONFIGS:
        if name not in results:
            continue
        item = results[name]
        hs = (hashes or {}).get(name, {})
        files = "; ".join(f"{key}={value}" for key, value in hs.items()) or "none"
        lines.append(f"{name}: runtime_s={item['runtime_s']:.2f}; SHA256_8: {files}")
        if name in metrics_by_config:
            for model in ("M0", "M1"):
                lines.append(_model_report(name, model, metrics_by_config[name]))
    if "A_primary" in results:
        lines.append("A_primary_most_selected: " + _most_selected(results["A_primary"]["params"]))
        lines.append(f"negative_control_A_M1_repeat0_AUROC: {results['A_primary']['negative_auroc']:.4f}")
    lines.extend([
        "acceptance: " + "; ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in checks.items()),
        "warnings: " + ("; ".join(warnings) if warnings else "none"),
        "=== END ===",
    ])
    if len(lines) > 30:
        raise AssertionError(f"REPORT exceeds 30 lines: {len(lines)}")
    return "\n".join(lines)


def run(config_path):
    started = time.perf_counter()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    locked_config(cfg)
    out = Path(cfg["output_root"]) / cfg["run_name"]
    checks = {
        "pytest": False, "oof_once_per_repeat": False, "no_nan": False,
        "no_outer_test_leakage": False, "negative_control_0_40_to_0_60": False,
        "fold_reproducibility": False, "no_overwrite": False,
    }
    results = {}
    metrics_by_config = {}
    warnings = []
    checks["no_overwrite"] = not out.exists()
    if not checks["no_overwrite"]:
        warnings.append("target output directory already exists; nothing overwritten")
        print(report("FAIL", started, checks, results, metrics_by_config, warnings))
        return 1
    project = Path(__file__).resolve().parents[1]
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(project / "tests" / "test_t03.py")],
        cwd=str(project.parent), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    checks["pytest"] = tests.returncode == 0
    if not checks["pytest"]:
        warnings.append("T03 synthetic tests failed; no outputs written")
        print(report("FAIL", started, checks, results, metrics_by_config, warnings))
        return 1

    labels, features, splits = _read_tables(cfg)
    print("=== P21-T03 START ===", flush=True)
    for name, spec in CONFIGS.items():
        print(f"{name}: running strict nested CV (10x5x5)", flush=True)
        item = _one_config(name, spec, labels, features, splits, cfg)
        results[name] = item
        if name == "A_primary":
            lower, upper = cfg["negative_control_auroc_bounds"]
            checks["negative_control_0_40_to_0_60"] = bool(lower <= item["negative_auroc"] <= upper)
            checks["fold_reproducibility"] = bool(item["reproducible"])
            if not checks["negative_control_0_40_to_0_60"]:
                warnings.append("negative control outside preregistered bounds; no outputs written")
                print(report("FAIL", started, checks, results, metrics_by_config, warnings))
                return 1
            if not checks["fold_reproducibility"]:
                warnings.append("repeated fold predictions differ; no outputs written")
                print(report("FAIL", started, checks, results, metrics_by_config, warnings))
                return 1

    checks["oof_once_per_repeat"] = all(item["oof_unique"] for item in results.values())
    checks["no_nan"] = all(item["no_nan"] for item in results.values())
    checks["no_outer_test_leakage"] = all(item["leakage_free"] for item in results.values())
    if not all(checks.values()):
        warnings.append("acceptance failed; no outputs written")
        print(report("FAIL", started, checks, results, metrics_by_config, warnings))
        return 1

    for name, item in results.items():
        print(f"{name}: calculating record-level 10000x bootstrap CIs", flush=True)
        metric_started = time.perf_counter()
        metrics_by_config[name] = _metrics_for_config(item["oof"], cfg)
        item["runtime_s"] += time.perf_counter() - metric_started

    if out.exists():
        checks["no_overwrite"] = False
        warnings.append("target output appeared during processing; no outputs written")
        print(report("FAIL", started, checks, results, metrics_by_config, warnings))
        return 1

    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    # Stage every output and publish one final directory only after all checks.
    with tempfile.TemporaryDirectory(prefix="t03_stage_", dir=parent) as staging_root:
        staged = Path(staging_root) / out.name
        staged.mkdir()
        for name, item in results.items():
            folder = staged / name
            folder.mkdir()
            item["oof"].to_parquet(folder / "oof.parquet", index=False)
            item["params"].to_csv(folder / "params.csv", index=False)
            (folder / "metrics.json").write_text(
                json.dumps(metrics_by_config[name], indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            hashes[name] = {key: sha8(folder / key) for key in OUTPUT_FILES}
        if out.exists():
            raise FileExistsError("Output directory appeared during staged write")
        staged.rename(out)

    print(report("PASS", started, checks, results, metrics_by_config, warnings, hashes=hashes))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("P21_CTG_Acidemia/configs/t03.yaml"))
    args = parser.parse_args()
    try:
        sys.exit(run(args.config))
    except Exception as exc:
        # Only aggregate error information, never signal/record identifiers.
        print(f"=== REPORT {TASK} ===")
        print("status: FAIL")
        print(f"error_type: {type(exc).__name__}")
        print(f"error: {exc}")
        print("warnings: no successful final run")
        print("=== END ===")
        sys.exit(1)

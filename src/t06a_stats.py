"""P21-T06a: H1 confirmatory inference and E1-E4 exploratory analysis.

Reads frozen T03/T04 OOF. Only E3 fits one additional M1 variant.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
import yaml
from common.bootstrap import discrimination_ci
from common.metrics import discrimination
from P21_CTG_Acidemia.src import t03_baselines as t03

PREREG = "40573556ecf4664c430430eb99eaaa20d1d0958aa32647cd30b7144434d048eb"
NAMES = ("A_primary", "B_noexcl", "C_w60", "D_y715", "E_y720")
N = {name: (552 if name == "B_noexcl" else 518) for name in NAMES}
E3_FEATURES = (*t03.FEATURES, "fhr_nan_pct_w30")
FILES = ("h1_tests.csv", "sensitivity.csv", "exploratory.csv", "e3_oof.parquet", "summary.json")


def check(condition, error):
    if not condition:
        raise ValueError(error)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def config_check(c):
    fixed = (
        c["task"] == "P21-T06a",
        c["run_name"] == "t06a_stats",
        str(c["prereg_version"]) == "1.3",
        c["prereg_sha256"] == PREREG,
        tuple(c["config_names"]) == NAMES,
        tuple(int(c[k]) for k in ("cv_repeats", "cv_outer_folds", "cv_inner_folds")) == (10, 5, 5),
        (int(c["bootstrap_resamples"]), int(c["bootstrap_seed"])) == (10000, 42),
        (float(c["alpha"]), float(c["equivalence_margin"])) == (0.05, 0.05),
        tuple(c["models_h1"]) == ("M2", "M3"),
        c["exploratory_feature"] == "fhr_nan_pct_w30",
        tuple(c["e3_feature_columns"]) == E3_FEATURES,
        list(map(float, c["m1_learning_rate"])) == [0.03, 0.1],
        list(map(int, c["m1_max_depth"])) == [2, 3],
        list(map(int, c["m1_min_samples_leaf"])) == [10, 20, 40],
        int(c["cpu_threads"]) == 4,
    )
    check(all(fixed), "Locked T06a config differs from directive")


def parse_frozen_sha(contents):
    """Select only OOF hashes; ignore metrics/log/manifest hashes per config."""
    result = {}
    digest_re = re.compile(r"(?<![0-9a-f])(?:[0-9a-f]{64}|[0-9a-f]{8})(?![0-9a-f])", re.I)
    for line in contents.splitlines():
        named = [name for name in NAMES if re.search(
            r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", line, re.I)]
        if len(named) != 1:
            continue
        name = named[0]
        tagged = re.search(
            r"oof(?:\.parquet|_sha256(?:_8)?|_sha8)?\s*[:=]\s*([0-9a-f]{8,64})(?![0-9a-f])",
            line, re.I)
        # The frozen file also contains metrics.json hashes for every config.
        # A config name alone never establishes that a hash belongs to OOF.
        if not tagged and not re.search(r"[/\\]oof\.parquet\b", line, re.I):
            continue
        matches = [tagged.group(1).lower()] if tagged else [
            m.group(0).lower() for m in digest_re.finditer(line)]
        check(len(matches) == 1 and len(matches[0]) in (8, 64),
              f"Ambiguous/missing OOF SHA {name}: {line}")
        if name in result:
            check(result[name] == matches[0], f"Conflicting hashes for {name}")
        result[name] = matches[0]
    check(set(result) == set(NAMES), f"Frozen SHA file missing OOF hashes for {set(NAMES)-set(result)}")
    return result


def verify_sha(c):
    lock_file = Path(c["frozen_hash_file"])
    expected = parse_frozen_sha(lock_file.read_text(encoding="utf-8-sig"))
    found = {}
    for name in NAMES:
        path = Path(c["project_dir"]) / "outputs" / "t04_dl" / name / "oof.parquet"
        actual = sha(path)
        check(actual.startswith(expected[name]),
              f"FROZEN_OOF_SHA_MISMATCH: {name} locked={expected[name]}, got={actual[:8]}")
        found[name] = actual
    return found


def validate_oof(df, name, columns):
    check({"record_id", "repeat", "outer_fold", "y", *columns}.issubset(df.columns),
          f"{name} OOF columns missing")
    check(len(df) == N[name] * 10 and df.record_id.astype(str).nunique() == N[name],
          f"{name} unexpected OOF record/row count")
    check(df.groupby(["record_id", "repeat"]).size().eq(1).all(),
          f"{name} record-repeat missing/duplicated")
    check(set(df.repeat.astype(int)) == set(range(10)), f"{name} incomplete repeats")
    check(df.outer_fold.astype(int).between(0, 4).all(), f"{name} invalid outer_fold")
    check(df.groupby("record_id").y.nunique().eq(1).all() and df.y.isin((0, 1)).all(),
          f"{name} invalid labels")
    p = df[columns].to_numpy(dtype=float)
    check(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), f"{name} invalid OOF scores")


def read_oof(c, name):
    root = Path(c["project_dir"]) / "outputs"
    base = pd.read_parquet(root / "t03_baselines" / name / "oof.parquet")
    dl = pd.read_parquet(root / "t04_dl" / name / "oof.parquet")
    dl_cols = [f"p_{m}_s{s}" for m in ("M2", "M3") for s in range(3)]
    if name == "A_primary":
        dl_cols += ["p_M2UC_s0"]
    validate_oof(base, name, ["p_M0", "p_M1"])
    validate_oof(dl, name, dl_cols)
    base, dl = base.copy(), dl.copy()
    base.record_id = base.record_id.astype(str)
    dl.record_id = dl.record_id.astype(str)
    joined = base.merge(dl[["record_id", "repeat", "outer_fold", "y", *dl_cols]],
                        on=["record_id", "repeat"], validate="one_to_one",
                        suffixes=("_t03", "_t04"))
    check(len(joined) == len(base) == len(dl), f"{name} T03/T04 coverage mismatch")
    check((joined.outer_fold_t03 == joined.outer_fold_t04).all() and
          (joined.y_t03 == joined.y_t04).all(), f"{name} fold/label disagreement")
    joined = joined.rename(columns={"outer_fold_t03": "outer_fold", "y_t03": "y"})
    joined = joined.drop(columns=["outer_fold_t04", "y_t04"])
    means = joined.groupby("record_id", sort=True).agg(
        y=("y", "first"), p_M0=("p_M0", "mean"), p_M1=("p_M1", "mean"),
        **{col: (col, "mean") for col in dl_cols})
    for model in ("M2", "M3"):
        means[f"p_{model}"] = means[[f"p_{model}_s{s}" for s in range(3)]].mean(axis=1)
    return joined, means


def paired_bootstrap(y, a, b, B=10000, seed=42, descriptive=False):
    """Resample identical record indices for both scores. Two-sided sign p."""
    y = np.asarray(y, dtype=np.int64)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    check(len(y) == len(a) == len(b) and len(y) >= 2 and
          np.array_equal(np.unique(y), [0, 1]), "Invalid bootstrap class/shape")
    check(np.isfinite(a).all() and np.isfinite(b).all(), "Bootstrap nonfinite scores")
    delta = float(discrimination(y, a)[0] - discrimination(y, b)[0])
    draws = np.empty((B, 3 if descriptive else 1), dtype=float)
    rng = np.random.default_rng(seed)
    valid = 0
    for _ in range(B):
        ix = rng.integers(0, len(y), len(y))
        yy = y[ix]
        if yy.min() == yy.max():
            continue
        aa, bb = a[ix], b[ix]
        draws[valid, 0] = roc_auc_score(yy, aa) - roc_auc_score(yy, bb)
        if descriptive:
            draws[valid, 1] = average_precision_score(yy, aa) - average_precision_score(yy, bb)
            draws[valid, 2] = np.mean((aa - yy)**2) - np.mean((bb - yy)**2)
        valid += 1
    check(valid > 0, "Bootstrap no valid resamples")
    draws = draws[:valid]
    p_raw = float(min(1.0, max(1 / B,
                 2 * min(np.mean(draws[:, 0] <= 0), np.mean(draws[:, 0] >= 0)))))
    result = {
        "delta": delta, "ci95": np.percentile(draws[:, 0], [2.5, 97.5]).tolist(),
        "ci90": np.percentile(draws[:, 0], [5, 95]).tolist(),
        "p_raw": p_raw, "bootstrap_valid": valid,
    }
    if descriptive:
        result.update({
            "delta_auprc": float(average_precision_score(y, a) - average_precision_score(y, b)),
            "ci95_auprc": np.percentile(draws[:, 1], [2.5, 97.5]).tolist(),
            "delta_brier": float(np.mean((a - y)**2) - np.mean((b - y)**2)),
            "ci95_brier": np.percentile(draws[:, 2], [2.5, 97.5]).tolist(),
        })
    return result


def holm(p_values):
    """Holm step-down adjustment, with monotonicity in sorted p-values."""
    p = np.asarray(p_values, dtype=float)
    check(p.ndim == 1 and len(p) > 0 and np.isfinite(p).all()
          and ((p >= 0) & (p <= 1)).all(), "Invalid Holm p")
    ix = np.argsort(p, kind="stable")
    output = np.empty_like(p)
    running = 0.0
    for rank, original in enumerate(ix):
        running = max(running, min(1.0, float(p[original]) * (len(p) - rank)))
        output[original] = running
    return output.tolist()


def tost(ci90, margin=0.05):
    lo, hi = ci90
    if lo > -margin and hi < margin:
        return "equivalent"
    if lo > margin or hi < -margin:
        return "non_equivalent"
    return "inconclusive"


def h1_primary(means):
    y = means.y.to_numpy(dtype=int)
    m1 = means.p_M1.to_numpy(dtype=float)
    auc_m1 = float(roc_auc_score(y, m1))
    rows = []
    for model in ("M2", "M3"):
        r = paired_bootstrap(y, means[f"p_{model}"], m1, descriptive=True)
        seed_auc = [float(roc_auc_score(y, means[f"p_{model}_s{s}"])) for s in range(3)]
        seed_delta = [v - auc_m1 for v in seed_auc]
        spread = float(max(seed_auc) - min(seed_auc))
        same_sign = (all(v > 0 for v in seed_delta) if r["delta"] > 0
                     else all(v < 0 for v in seed_delta))
        rows.append({
            "config": "A_primary", "comparison": f"{model}-M1",
            "delta_auroc": r["delta"], "ci95_low": r["ci95"][0],
            "ci95_high": r["ci95"][1], "ci90_low": r["ci90"][0],
            "ci90_high": r["ci90"][1], "p_raw": r["p_raw"],
            "delta_auprc": r["delta_auprc"],
            "ci95_auprc_low": r["ci95_auprc"][0],
            "ci95_auprc_high": r["ci95_auprc"][1],
            "delta_brier": r["delta_brier"],
            "ci95_brier_low": r["ci95_brier"][0],
            "ci95_brier_high": r["ci95_brier"][1],
            "max_seed_auroc_spread": spread,
            "seed_deltas": json.dumps(seed_delta),
            "gate_seed_spread": bool(abs(r["delta"]) > spread),
            "gate_seed_sign": bool(same_sign),
            "tost": tost(r["ci90"]),
        })
    for row, p_holm in zip(rows, holm([row["p_raw"] for row in rows])):
        row["p_holm"] = p_holm
        row["gate_holm"] = bool(p_holm < 0.05)
        row["dl_beats_m1"] = bool(row["delta_auroc"] > 0 and row["gate_holm"]
                                  and row["gate_seed_spread"] and row["gate_seed_sign"])
    return rows, auc_m1 - float(roc_auc_score(y, means.p_M0))


def sensitivity(name, means):
    rows = []
    for model in ("M2", "M3"):
        r = paired_bootstrap(means.y, means[f"p_{model}"], means.p_M1)
        rows.append({
            "config": name, "comparison": f"{model}-M1",
            "delta_auroc": r["delta"], "ci95_low": r["ci95"][0],
            "ci95_high": r["ci95"][1], "p_raw": r["p_raw"],
            "analysis_type": "sensitivity, no Holm",
        })
    return rows


def positive_label(y):
    number = int(np.sum(y))
    return "<10" if number < 10 else str(number)


def exploratory_auc(analysis, config, group, y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    check(np.isfinite(p).all(), "Nonfinite exploratory prediction")
    if analysis == "E1":
        # T02 stores 100 * missingness fraction (percentage, 0..100).
        # Division by 100 preserves ordering, ties, and AUROC/bootstrap CI,
        # while meeting the frozen common.metrics probability-range contract.
        check(((p >= 0) & (p <= 100)).all(), "E1 FHR missingness outside [0,100]%")
        p = p / 100.0
    auc = float(discrimination(y, p)[0])
    ci = discrimination_ci(y, p, n_resamples=10000, seed=42)["auroc_ci95"]
    return {"analysis": analysis, "config": config, "group": group, "n": len(y),
            "positives": positive_label(y), "measure": "AUROC",
            "estimate": auc, "ci95_low": ci[0], "ci95_high": ci[1]}


def exploratory_delta(analysis, config, group, y, a, b):
    r = paired_bootstrap(y, a, b)
    return {"analysis": analysis, "config": config, "group": group, "n": len(y),
            "positives": positive_label(y), "measure": "delta_AUROC",
            "estimate": r["delta"], "ci95_low": r["ci95"][0], "ci95_high": r["ci95"][1]}


def exploratory_e1_e2(data, labels):
    rows = []
    for config in ("A_primary", "B_noexcl"):
        mean = data[config]["means"]
        missingness = labels.loc[mean.index, "fhr_nan_pct_w30"].to_numpy(dtype=float)
        rows.append(exploratory_auc("E1", config, "fhr_nan_pct_w30",
                                    mean.y, missingness))
    primary = data["A_primary"]["means"]
    missingness = labels.loc[primary.index, "fhr_nan_pct_w30"]
    median = float(missingness.median())
    for group, mask in (("le_median", missingness <= median),
                        ("gt_median", missingness > median)):
        subset = primary.loc[mask.to_numpy()]
        y = subset.y.to_numpy(dtype=int)
        for model in ("M1", "M3"):
            rows.append(exploratory_auc("E2", "A_primary", f"{group}_{model}",
                                        y, subset[f"p_{model}"]))
        rows.append(exploratory_delta("E2", "A_primary", f"{group}_M3-M1",
                                      y, subset.p_M3, subset.p_M1))
    return rows, median


def exploratory_e3(joined, primary, labels, features, splits, c):
    """E3: repeat T03 M1 inner tuning, with its 11 features plus missingness."""
    check(tuple(c["e3_feature_columns"]) == E3_FEATURES, "Unexpected E3 feature")
    features = features.copy()
    features.record_id = features.record_id.astype(str)
    check(not features.record_id.duplicated().any(), "Duplicate features record_id")
    frame = features.set_index("record_id").loc[primary.index, list(t03.FEATURES)].copy()
    frame["fhr_nan_pct_w30"] = labels.loc[primary.index, "fhr_nan_pct_w30"]
    frame = frame.astype(float)
    check(not np.isinf(frame.to_numpy()).any(), "Infinite feature E3")
    check(len(splits) == 10 * 5 * len(primary) and set(splits.record_id) == set(primary.index),
          "T02 split missing A records")
    results, chosen = [], []
    for repeat in range(10):
        for fold in range(5):
            block = splits[(splits.repeat == repeat) & (splits.outer_fold == fold)]
            train = block.loc[block.inner_fold >= 0]
            test = block.loc[block.inner_fold == -1]
            train_ids = train.record_id.to_numpy(dtype=str)
            test_ids = test.record_id.to_numpy(dtype=str)
            check(len(train_ids) > 0 and len(test_ids) > 0, "E3 empty fold")
            check(not set(train_ids) & set(test_ids), "E3 outer-test leakage")
            check(set(train_ids) | set(test_ids) == set(primary.index), "E3 incomplete fold")
            x_train = frame.loc[train_ids, E3_FEATURES].to_numpy(dtype=float)
            x_test = frame.loc[test_ids, E3_FEATURES].to_numpy(dtype=float)
            y_train = labels.loc[train_ids, "y705"].to_numpy(dtype=int)
            predictions, best_params, cv_auc = t03.tune_and_predict(
                "M1", x_train, y_train, train.inner_fold.to_numpy(dtype=int),
                x_test, c, train_ids, test_ids)
            chosen.append({"repeat": repeat, "outer_fold": fold,
                           "params": best_params, "inner_auroc": float(cv_auc)})
            for rid, score in zip(test_ids, predictions):
                results.append({"record_id": rid, "repeat": repeat, "outer_fold": fold,
                                "y": int(labels.at[rid, "y705"]),
                                "p_M1_plus_missingness": float(score)})
        print(f"E3_M1_plus_fhr_nan_pct: repeat={repeat+1}/10 COMPLETE", flush=True)
    oof = pd.DataFrame(results).sort_values(["repeat", "record_id"]).reset_index(drop=True)
    validate_oof(oof, "A_primary", ["p_M1_plus_missingness"])
    compare = oof.merge(joined[["record_id", "repeat", "outer_fold", "y"]],
                        on=["record_id", "repeat"], validate="one_to_one",
                        suffixes=("_e3", "_frozen"))
    check(len(compare) == len(oof) and (compare.outer_fold_e3 == compare.outer_fold_frozen).all()
          and (compare.y_e3 == compare.y_frozen).all(),
          "E3 OOF assignment diverges from frozen T03/T04")
    predicted = oof.groupby("record_id", sort=True).p_M1_plus_missingness.mean()
    check(predicted.index.equals(primary.index), "E3 averaged score alignment failed")
    rows = [exploratory_auc("E3", "A_primary", "M1_plus_fhr_nan_pct_w30",
                            primary.y, predicted)]
    for ref in ("M1", "M3"):
        rows.append(exploratory_delta("E3", "A_primary", f"M1_plus_missingness-{ref}",
                                      primary.y, predicted, primary[f"p_{ref}"]))
    return oof, rows, chosen


def spearman_bootstrap(a, b, B=10000, seed=42):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    check(len(a) == len(b) and len(a) > 1 and np.isfinite(a).all()
          and np.isfinite(b).all(), "E4 nonfinite or mismatched inputs")
    point = float(spearmanr(a, b).statistic)
    check(np.isfinite(point), "E4 undefined Spearman")
    rng, draws = np.random.default_rng(seed), []
    for _ in range(B):
        ix = rng.integers(0, len(a), len(a))
        if np.unique(a[ix]).size < 2 or np.unique(b[ix]).size < 2:
            continue
        rho = float(spearmanr(a[ix], b[ix]).statistic)
        if np.isfinite(rho):
            draws.append(rho)
    check(len(draws) > 0, "E4 no valid bootstrap samples")
    return point, np.percentile(draws, [2.5, 97.5]).tolist()


def exploratory_e4(primary, labels):
    missingness = labels.loc[primary.index, "fhr_nan_pct_w30"].to_numpy(dtype=float)
    y = labels.loc[primary.index, "y705"].to_numpy(dtype=int)
    check(np.array_equal(y, primary.y.to_numpy(dtype=int)), "E4 label mismatch")
    rows = []
    for group, mask in (("all", np.ones(len(y), dtype=bool)), ("y705_eq_0", y == 0)):
        for model in ("M1", "M2", "M3"):
            rho, ci = spearman_bootstrap(missingness[mask],
                                         primary[f"p_{model}"].to_numpy(dtype=float)[mask])
            rows.append({"analysis": "E4", "config": "A_primary",
                         "group": f"{group}_{model}", "n": int(mask.sum()),
                         "positives": positive_label(y[mask]), "measure": "Spearman_rho",
                         "estimate": rho, "ci95_low": ci[0], "ci95_high": ci[1]})
    return rows


def delta_text(row):
    return f"{row['delta_auroc']:+.4f}[{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}]"


def make_report(h1, sens, exploratory, descriptive, hashes, elapsed):
    lines = [
        "=== REPORT P21-T06a ===",
        f"status: PASS; PC-A CPU; runtime_s={elapsed:.2f}; GPU=unused",
        f"PREREG_v1.3_SHA256={PREREG}",
        "acceptance: T04_sha5=PASS; pytest=PASS; A=518; B=552; p_and_CI_no_NaN=PASS",
        "PART 1 H1 CONFIRMATORY A_primary; paired bootstrap 10000/seed42; Holm family 2",
        f"M1-M0 (descriptive; untested): delta_AUROC={descriptive:+.4f}",
        "H1: comparison | delta[CI95] | p_raw,p_Holm | Holm/seedspread/sign | TOST | DL_beats_M1",
    ]
    for row in h1:
        gates = "/".join("PASS" if row[key] else "FAIL"
                         for key in ("gate_holm", "gate_seed_spread", "gate_seed_sign"))
        lines.append(
            f"{row['comparison']} | {delta_text(row)} | {row['p_raw']:.4g},{row['p_holm']:.4g}"
            f" | {gates} | {row['tost']} | {'YES' if row['dl_beats_m1'] else 'NO'}")
        lines.append(
            f"  descriptive delta_AUPRC={row['delta_auprc']:+.4f}"
            f"[{row['ci95_auprc_low']:+.4f},{row['ci95_auprc_high']:+.4f}]; "
            f"delta_Brier={row['delta_brier']:+.4f}"
            f"[{row['ci95_brier_low']:+.4f},{row['ci95_brier_high']:+.4f}]; "
            f"seed_spread={row['max_seed_auroc_spread']:.4f}; seed_deltas={row['seed_deltas']}")
    lines.append("SENSITIVITY: raw p, no Holm")
    for name in NAMES[1:]:
        subset = [row for row in sens if row["config"] == name]
        lines.append(name + ": " + "; ".join(
            f"{row['comparison']}={delta_text(row)} p_raw={row['p_raw']:.4g}"
            for row in subset))
    lines.append("PART 2 EXPLORATORY, POST HOC (E1-E4); not confirmatory")
    for phase in ("E1", "E2", "E3", "E4"):
        subset = [row for row in exploratory if row["analysis"] == phase]
        lines.append(phase + ": " + "; ".join(
            f"{row['config']}/{row['group']} n={row['n']} positive={row['positives']} "
            f"{row['measure']}={row['estimate']:+.4f}"
            f"[{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}]"
            for row in subset))
    lines.append("SHA256_8 script=" + sha(__file__)[:8] + "; " +
                 "; ".join(f"{name}={hashes[name][:8]}" for name in FILES))
    lines.append("=== END ===")
    check(len(lines) <= 30, "REPORT exceeds 30 lines")
    return "\n".join(lines) + "\n"


def run(c):
    started = time.perf_counter()
    config_check(c)
    project = Path(c["project_dir"])
    destination = project / "outputs" / "t06a_stats"
    if destination.exists():
        raise FileExistsError(f"STOP: output directory exists: {destination}")
    frozen = verify_sha(c)  # Must precede all training and calculations.
    test_path = Path(__file__).resolve().parents[1] / "tests" / "test_t06a.py"
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", str(test_path)], check=False)
    check(test.returncode == 0, f"T06a pytest failed, exit={test.returncode}")
    data = {}
    for name in NAMES:
        joined, means = read_oof(c, name)
        data[name] = {"joined": joined, "means": means}
        print(f"FROZEN_T04_SHA_{name}={frozen[name][:8]} PASS; unique_records={len(means)}",
              flush=True)
    input_dir = Path(c["input_dir"])
    labels = pd.read_parquet(input_dir / "labels.parquet")
    features = pd.read_parquet(input_dir / "features_w30.parquet")
    splits = pd.read_csv(input_dir / "split.csv", dtype={"record_id": str})
    labels.record_id = labels.record_id.astype(str)
    features.record_id = features.record_id.astype(str)
    check(len(labels) == 552 and not labels.record_id.duplicated().any(),
          "T02 labels must have 552 unique rows")
    check({"record_id", "excluded", "y705", "y715", "y720", "fhr_nan_pct_w30"}
          .issubset(labels.columns), "Required T02 labels missing")
    check(set(t03.FEATURES).issubset(features.columns), "11 T03 features missing")
    check(list(splits.columns) == ["record_id", "repeat", "outer_fold", "inner_fold"],
          "Split format changed")
    labels = labels.set_index("record_id")
    check(np.isfinite(labels.fhr_nan_pct_w30.to_numpy(dtype=float)).all(),
          "Missingness fraction nonfinite")
    check(set(labels.index) == set(data["B_noexcl"]["means"].index),
          "B OOF not all T02 records")
    check(set(labels.index[~labels.excluded.astype(bool)]) ==
          set(data["A_primary"]["means"].index), "A QC eligibility mismatch")
    for name in NAMES:
        means = data[name]["means"]
        target = {"D_y715": "y715", "E_y720": "y720"}.get(name, "y705")
        check(np.array_equal(labels.loc[means.index, target].to_numpy(dtype=int),
                             means.y.to_numpy(dtype=int)),
              f"{name} frozen labels mismatch")
        if name != "B_noexcl":
            check(set(means.index) == set(data["A_primary"]["means"].index),
                  f"{name} primary eligible record mismatch")

    primary = data["A_primary"]["means"]
    h1, descriptive = h1_primary(primary)
    sens = [row for name in NAMES[1:] for row in sensitivity(name, data[name]["means"])]
    exploratory, median = exploratory_e1_e2(data, labels)
    print("H1, sensitivities, E1/E2 COMPLETE. Starting ONLY E3 extra M1 variant.", flush=True)
    e3_oof, rows_e3, params = exploratory_e3(
        data["A_primary"]["joined"], primary, labels, features, splits, c)
    exploratory.extend(rows_e3)
    exploratory.extend(exploratory_e4(primary, labels))
    check(len(h1) == 2 and len(sens) == 8 and len(exploratory) == 17,
          "Unexpected analysis row counts")
    for rows in (h1, sens, exploratory):
        for row in rows:
            for key, value in row.items():
                if key in ("p_raw", "p_holm", "delta_auroc", "estimate") or "ci95_" in key or "ci90_" in key:
                    check(np.isfinite(float(value)),
                          f"Nonfinite p or CI {row.get('comparison', row.get('group'))}.{key}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"STOP: directory became occupied: {destination}")
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".t06a_stage_") as temp:
        stage = Path(temp)
        pd.DataFrame(h1).to_csv(stage / "h1_tests.csv", index=False)
        pd.DataFrame(sens).to_csv(stage / "sensitivity.csv", index=False)
        pd.DataFrame(exploratory).to_csv(stage / "exploratory.csv", index=False)
        e3_oof.to_parquet(stage / "e3_oof.parquet", index=False)
        hashes = {name: sha(stage / name) for name in FILES[:-1]}
        elapsed = time.perf_counter() - started
        summary = {
            "status": "PASS", "task": "P21-T06a", "runtime_s": elapsed,
            "prereg_version": "1.3", "prereg_sha256_director_declared": PREREG,
            "frozen_t04_sha256": frozen, "record_n": N,
            "bootstrap": {"unit": "paired_record", "n_resamples": 10000, "seed": 42},
            "h1": h1, "m1_minus_m0_descriptive": descriptive,
            "sensitivity_no_holm": sens, "exploratory_post_hoc": exploratory,
            "e2_missingness_median": median, "e3_inner_cv_params": params,
            "sha256_script": sha(__file__), "sha256_files_except_summary": hashes,
            "acceptance": {
                "frozen_sha": True, "pytest": True, "nA518": True, "nB552": True,
                "no_nan_p_ci": True, "retrain_original_M0_M1_M2_M3": False,
            },
        }
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        hashes["summary.json"] = sha(stage / "summary.json")
        report = make_report(h1, sens, exploratory, descriptive, hashes, elapsed)
        (stage / "report.txt").write_text(report, encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"STOP before publish: {destination}")
        stage.rename(destination)
    print(report, end="", flush=True)
    print("P21_T06A_EXIT_CODE=0", flush=True)


def main():
    parser = argparse.ArgumentParser(description="P21-T06a frozen OOF statistical analysis")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs" / "t06a.yaml")
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--frozen-sha", type=Path)
    args = parser.parse_args()
    c = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for cli, setting in (("project_dir", "project_dir"), ("input_dir", "input_dir"),
                         ("frozen_sha", "frozen_hash_file")):
        if getattr(args, cli) is not None:
            c[setting] = str(getattr(args, cli))
    try:
        run(c)
    except Exception as exc:
        print(f"P21_T06A_STOP: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()


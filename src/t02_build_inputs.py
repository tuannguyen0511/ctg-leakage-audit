"""P21-T02: fixed preregistered inputs. No training or outcome association."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import yaml

from common.io_wfdb import read_record
from common.report import render_report, safe_count
from common.splits import build_nested_splits, verify_nested_splits
from P21_CTG_Acidemia.src.ctg_features import FEATURE_NAMES, build_dl_1hz, clean_signals, compute_features


TASK = "P21-T02"
LOCKED_PREREG_SHA = "df1d7b0c3b4b0a27bc267ce6b97b05808f66e88e8530e989efa337071b079c89"
OUTPUT_NAMES = (
    "labels.parquet", "features_w30.parquet", "features_w60.parquet",
    "dl_w30_1hz.npz", "dl_w60_1hz.npz", "split.csv", "split_noexcl.csv",
)


def _sha8(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:8]


def _number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _stat(values):
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return "not available"
    q1, median, q3 = np.percentile(series, [25, 50, 75])
    return f"{median:.3f}[{q1:.3f},{q3:.3f}]"


def _feature_lines(frame):
    parts = [
        f"{name}:NaN={safe_count(frame[name].isna().sum())},med[IQR]={_stat(frame[name])}"
        for name in FEATURE_NAMES
    ]
    return "; ".join(parts[:6]), "; ".join(parts[6:])


def _label_rates(labels, include):
    denominator = int(include.sum())
    parts = []
    for label in ("y705", "y715", "y720"):
        count = int(labels.loc[include, label].sum())
        if denominator == 0:
            pct = "not available"
        elif 1 <= count <= 9:
            pct = f"<={900 / denominator:.2f}%"
        else:
            pct = f"{100 * count / denominator:.2f}%"
        parts.append(f"{label}={safe_count(count)}({pct})")
    return "; ".join(parts)


def _outer_test_stats(splits, selected):
    test_rows = splits.loc[splits.inner_fold == -1].merge(
        selected, on="record_id", validate="many_to_one"
    )
    return test_rows.groupby(["repeat", "outer_fold"], sort=True).agg(
        n=("record_id", "size"), positives=("y705", "sum")
    )


def _balanced_by_repeat(stats, repeats, outer_folds):
    if len(stats) != repeats * outer_folds:
        return False
    spread = stats.groupby(level="repeat")["positives"].agg(lambda x: x.max() - x.min())
    return bool(len(spread) == repeats and spread.le(1).all())


def audit(config_path):
    started = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_dir = Path(__file__).resolve().parents[1]
    raw_dir = Path(cfg["raw_dir"])
    derived_dir = Path(cfg["derived_dir"])
    output_dir = Path(cfg["output_root"]) / cfg["run_name"]
    checks = {
        "pytest": False, "outer_test_y705_balance": False,
        "outer_test_y705_noexcl_balance": False,
        "no_record_overlap": False, "no_record_overlap_noexcl": False, "dl_shapes": False,
        "dl_no_nan": False, "no_overwrite": False,
    }
    summary = {
        "status": "FAIL", "pc": "PC-A CPU; OMP_NUM_THREADS=4",
        "runtime_s": "not finished", "script_sha256_8": _sha8(Path(__file__)),
        "prereg_version": "1.1",
        "prereg_sha256_director_declared": cfg["prereg_sha256"],
    }
    warnings = []

    if cfg["run_name"] != "t02_inputs" or cfg["prereg_sha256"] != LOCKED_PREREG_SHA:
        warnings.append("RUN_NAME or Director prereg SHA mismatch")
    if tuple(cfg["windows_minutes"]) != (30, 60) or cfg["fs_hz"] != 4 or cfg["dl_fs_hz"] != 1:
        warnings.append("fixed window or sampling configuration mismatch")
    present = [name for name in OUTPUT_NAMES if (derived_dir / name).exists()]
    checks["no_overwrite"] = not output_dir.exists() and not present
    if not checks["no_overwrite"]:
        warnings.append("output already exists; no files overwritten")
    if not raw_dir.is_dir() or not (raw_dir / "RECORDS").is_file():
        warnings.append("raw directory or RECORDS missing")
    for input_name in ("clinical.parquet", "signal_qc.parquet"):
        if not (derived_dir / input_name).is_file():
            warnings.append(f"T01 input missing: {input_name}")
    if warnings:
        summary["runtime_s"] = f"{time.perf_counter() - started:.2f}"
        print(render_report(TASK, summary, checks, "; ".join(warnings)))
        return 1

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(project_dir / "tests" / "test_t02.py")],
        cwd=str(project_dir.parent), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False,
    )
    checks["pytest"] = tests.returncode == 0
    if not checks["pytest"]:
        warnings.append("T02 synthetic tests failed; no outputs written")
        summary["runtime_s"] = f"{time.perf_counter() - started:.2f}"
        print(render_report(TASK, summary, checks, "; ".join(warnings)))
        return 1

    np.random.seed(int(cfg["cv_seed"]))
    clinical = pd.read_parquet(derived_dir / "clinical.parquet")
    signal_qc = pd.read_parquet(derived_dir / "signal_qc.parquet")
    record_ids = [line.strip() for line in (raw_dir / "RECORDS").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if (
        len(record_ids) != len(set(record_ids))
        or set(record_ids) != set(clinical.record_name)
        or set(record_ids) != set(signal_qc.record_name)
        or clinical.record_name.duplicated().any()
        or signal_qc.record_name.duplicated().any()
    ):
        raise ValueError("RECORDS and T01 derived record IDs do not match")
    clinical = clinical.set_index("record_name").loc[record_ids]

    labels_rows = []
    feature_rows = {30: [], 60: []}
    dl_rows = {30: [], 60: []}
    short_counts = {30: 0, 60: 0}
    all_fhr_missing = {30: 0, 60: 0}
    for name in record_ids:
        record = read_record(raw_dir, name)
        if record.fs != cfg["fs_hz"]:
            raise ValueError("Nonuniform fs in record")
        channels = {channel.upper(): i for i, channel in enumerate(record.channel_names)}
        if "FHR" not in channels or "UC" not in channels:
            raise ValueError("Record does not have FHR and UC")
        fhr, uc = clean_signals(record.signal[:, channels["FHR"]], record.signal[:, channels["UC"]], cfg)
        position = _number(record.clinical.get("Pos. II.st."))
        no_stage2 = not (np.isfinite(position) and position >= 0)
        end = len(fhr) if no_stage2 else int(position)
        if end < 0 or end > len(fhr):
            raise ValueError("Stage-2 starting position outside record")
        ph = _number(clinical.loc[name, "pH"])
        if not np.isfinite(ph):
            raise ValueError("Missing numerical pH")
        nan_percentages = {}
        for minutes in cfg["windows_minutes"]:
            target_samples = int(minutes * 60 * cfg["fs_hz"])
            start = max(0, end - target_samples)
            fhr_win, uc_win = fhr[start:end], uc[start:end]
            if len(fhr_win) < target_samples:
                short_counts[minutes] += 1
            if not np.isfinite(fhr_win).any():
                all_fhr_missing[minutes] += 1
            nan_percentages[minutes] = float(100 * np.mean(~np.isfinite(fhr_win))) if len(fhr_win) else 100.0
            feature_rows[minutes].append({"record_id": name, **compute_features(fhr_win, uc_win, cfg)})
            dl_rows[minutes].append(build_dl_1hz(fhr_win, uc_win, minutes, cfg))
        labels_rows.append({
            "record_id": name,
            "y705": int(ph < 7.05), "y715": int(ph < 7.15), "y720": int(ph < 7.20),
            "no_stage2": no_stage2,
            "fhr_nan_pct_w30": nan_percentages[30],
            "fhr_nan_pct_w60": nan_percentages[60],
            "excluded": nan_percentages[30] > cfg["exclusion_fhr_nan_pct_w30"],
        })

    labels = pd.DataFrame(labels_rows)
    features = {minutes: pd.DataFrame(rows) for minutes, rows in feature_rows.items()}
    X = {minutes: np.stack(rows).astype(np.float32) for minutes, rows in dl_rows.items()}
    checks["dl_shapes"] = all(X[m].shape == (len(record_ids), 3, m * 60) for m in (30, 60))
    checks["dl_no_nan"] = all(np.isfinite(X[m]).all() for m in (30, 60))

    include = ~labels.excluded
    selected = labels.loc[include, ["record_id", "y705"]]
    cv_args = {
        "repeats": int(cfg["cv_repeats"]),
        "outer_folds": int(cfg["cv_outer_folds"]),
        "inner_folds": int(cfg["cv_inner_folds"]),
        "seed": int(cfg["cv_seed"]),
    }
    splits = build_nested_splits(
        selected.record_id.to_numpy(), selected.y705.to_numpy(),
        **cv_args,
    )
    splits_noexcl = build_nested_splits(
        labels.record_id.to_numpy(), labels.y705.to_numpy(),
        **cv_args,
    )
    checks["no_record_overlap"] = verify_nested_splits(
        splits, selected.record_id.to_numpy(),
        repeats=cv_args["repeats"], outer_folds=cv_args["outer_folds"], inner_folds=cv_args["inner_folds"],
    )
    checks["no_record_overlap_noexcl"] = verify_nested_splits(
        splits_noexcl, labels.record_id.to_numpy(),
        repeats=cv_args["repeats"], outer_folds=cv_args["outer_folds"], inner_folds=cv_args["inner_folds"],
    )
    fold_stats = _outer_test_stats(splits, selected)
    fold_stats_noexcl = _outer_test_stats(splits_noexcl, labels[["record_id", "y705"]])
    checks["outer_test_y705_balance"] = _balanced_by_repeat(
        fold_stats, cv_args["repeats"], cv_args["outer_folds"]
    )
    checks["outer_test_y705_noexcl_balance"] = _balanced_by_repeat(
        fold_stats_noexcl, cv_args["repeats"], cv_args["outer_folds"]
    )

    feature_lines = _feature_lines(features[30])
    mask_pct = 100 * X[30][:, 1, :].mean(axis=1)
    summary.update({
        "data": f"CTU-UHB 1.0.0; N={len(record_ids)}",
        "excluded": safe_count(labels.excluded.sum()),
        "excluded_by_y705": (
            f"y705=1:{safe_count((labels.excluded & labels.y705.eq(1)).sum())}; "
            f"y705=0:{safe_count((labels.excluded & labels.y705.eq(0)).sum())}"
        ),
        "no_stage2": safe_count(labels.no_stage2.sum()),
        "short_w30_w60": f"{safe_count(short_counts[30])}/{safe_count(short_counts[60])}",
        "all_FHR_missing_w30_w60": f"{safe_count(all_fhr_missing[30])}/{safe_count(all_fhr_missing[60])}",
        "positive_after_exclusion": _label_rates(labels, include),
        "features_w30_a": feature_lines[0],
        "features_w30_b": feature_lines[1],
        "dl_w30_mask_pct_median_IQR": _stat(mask_pct),
        "dl_shapes_w30_w60": f"{tuple(X[30].shape)}/{tuple(X[60].shape)}",
        "split_format": "long, both files; inner_fold=-1 outer-test, 0..4 outer-train; split excludes QC, split_noexcl includes all",
        "outer_test_y705_min_max": f"{int(fold_stats.positives.min())}/{int(fold_stats.positives.max())}",
        "outer_test_y705_noexcl_min_max": f"{int(fold_stats_noexcl.positives.min())}/{int(fold_stats_noexcl.positives.max())}",
        "outer_test_size_min_max": f"{int(fold_stats.n.min())}/{int(fold_stats.n.max())}",
        "split_overlap_check": (
            "PASS" if checks["no_record_overlap"] and checks["no_record_overlap_noexcl"] else "FAIL"
        ),
    })

    if not all(checks.values()):
        warnings.append("acceptance criterion failed; no final outputs written")
        summary["outputs"] = "none"
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        labels.to_parquet(derived_dir / "labels.parquet", index=False)
        features[30].to_parquet(derived_dir / "features_w30.parquet", index=False)
        features[60].to_parquet(derived_dir / "features_w60.parquet", index=False)
        for minutes in (30, 60):
            np.savez_compressed(
                derived_dir / f"dl_w{minutes}_1hz.npz",
                X=X[minutes], record_id=np.asarray(record_ids, dtype=str),
            )
        splits.to_csv(derived_dir / "split.csv", index=False)
        splits_noexcl.to_csv(derived_dir / "split_noexcl.csv", index=False)
        summary["outputs"] = "; ".join(OUTPUT_NAMES)
        summary["output_sha256_8"] = "; ".join(f"{name}={_sha8(derived_dir / name)}" for name in OUTPUT_NAMES)

    summary["status"] = "PASS" if all(checks.values()) else "FAIL"
    summary["runtime_s"] = f"{time.perf_counter() - started:.2f}"
    print(render_report(TASK, summary, checks, "; ".join(warnings) or "none"))
    if output_dir.is_dir():
        (output_dir / "summary.json").write_text(
            json.dumps({"summary": summary, "acceptance": checks, "warnings": warnings}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("P21_CTG_Acidemia/configs/t02.yaml"))
    args = parser.parse_args()
    sys.exit(audit(args.config))

"""P21-T01: read-only audit of the CTU-UHB intrapartum CTG database."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import yaml

from common.io_wfdb import read_record
from common.report import render_report, safe_count


TASK = "P21-T01"
WINDOWS = (30, 60)


def numeric(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan


def percent_zero(signal, end, fs, minutes):
    start = max(0, end - int(round(minutes * 60 * fs)))
    segment = signal[start:end]
    return (float(np.mean(segment == 0) * 100) if segment.size else np.nan, len(segment))


def qstats(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return "not available"
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return f"{median:.3f} [{q1:.3f}, {q3:.3f}]"


def threshold_summary(ph, threshold, inclusive):
    label = "<=" if inclusive else "<"
    available = int(ph.notna().sum())
    if not available:
        return f"{label}{threshold:.2f}: not available"
    count = int((ph <= threshold).sum() if inclusive else (ph < threshold).sum())
    percentage = f"<={100 * 9 / available:.2f}%" if 1 <= count <= 9 else f"{100 * count / available:.2f}%"
    return f"{label}{threshold:.2f}: n={safe_count(count)}, pct={percentage}"


def make_manifest(raw_dir):
    files = sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and (path.suffix.lower() in {".hea", ".dat"} or path.name == "RECORDS")
    )
    rows = []
    for path in files:
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        rows.append({"file": path.relative_to(raw_dir).as_posix(), "size": path.stat().st_size, "sha256": sha.hexdigest()})
    return pd.DataFrame(rows, columns=["file", "size", "sha256"])


def audit(config_path):
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["run_name"] != "t01_audit" or tuple(config["stage1_windows_minutes"]) != WINDOWS:
        raise ValueError("T01 requires run_name=t01_audit and windows [30,60]")
    np.random.seed(int(config["seed"]))
    raw_dir = Path(config["raw_dir"])
    derived_dir = Path(config["derived_dir"])
    output_dir = Path(config["output_root"]) / config["run_name"]
    project_dir = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:8]

    summary = {
        "status": "FAIL",
        "pc": "PC-A CPU; OMP_NUM_THREADS=4; seed=42",
        "runtime_s": "not finished",
        "script_sha256_8": source_hash,
    }
    checks = {
        "pytest": False,
        "records_read_equal_RECORDS": False,
        "fs_uniform": False,
        "pH_coverage_ge_95pct": False,
        "manifest_complete": False,
        "aggregate_report_only": True,
    }
    warnings = []

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(project_dir / "tests")],
        cwd=str(project_dir.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    checks["pytest"] = test.returncode == 0
    if not checks["pytest"]:
        warnings.append("synthetic pytest failed; no dataset files written")

    if output_dir.exists():
        warnings.append("output RUN_NAME already exists; no files overwritten")
    if not raw_dir.is_dir() or not (raw_dir / "RECORDS").is_file():
        warnings.append("raw data directory or RECORDS missing")
    existing_derived = [name for name in ("manifest_sha256.csv", "clinical.parquet", "signal_qc.parquet") if (derived_dir / name).exists()]
    if existing_derived:
        warnings.append("derived output already exists; no files overwritten")

    if warnings:
        summary["runtime_s"] = f"{time.perf_counter() - started:.2f}"
        print(render_report(TASK, summary, checks, "; ".join(warnings)))
        return 1

    records = [line.strip() for line in (raw_dir / "RECORDS").read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    expected = {"RECORDS"} | {f"{name}{suffix}" for name in records for suffix in (".hea", ".dat")}
    manifest = make_manifest(raw_dir)
    manifest_names = set(manifest["file"].tolist())
    checks["manifest_complete"] = expected.issubset(manifest_names) and len(manifest) == len(manifest_names) and len(records) == len(set(records))

    clinical_rows = []
    qc_rows = []
    error_types = Counter()
    channel_sets = set()
    for name in records:
        try:
            record = read_record(raw_dir, name)
            if record.signal.ndim != 2 or len(record.channel_names) != 2:
                raise ValueError("Unexpected channel dimensions")
            channel_index = {channel.upper(): i for i, channel in enumerate(record.channel_names)}
            if "FHR" not in channel_index or "UC" not in channel_index:
                raise ValueError("FHR or UC channel missing")
            fhr = record.signal[:, channel_index["FHR"]]
            uc = record.signal[:, channel_index["UC"]]
            fs = record.fs
            n_samples = len(fhr)
            position = numeric(record.clinical.get("Pos. II.st."))
            stage_length = numeric(record.clinical.get("II.stage"))
            has_stage2 = bool(np.isfinite(position) and position >= 0)
            if has_stage2 and position > n_samples:
                raise ValueError("Stage 2 start is outside signal")
            stage1_end = int(position) if has_stage2 else n_samples
            row = {
                "record_name": name,
                "fs": fs,
                "n_samples": n_samples,
                "duration_minutes": n_samples / (fs * 60),
                "fhr_zero_pct": float(np.mean(fhr == 0) * 100),
                "uc_zero_pct": float(np.mean(uc == 0) * 100),
                "has_stage2": has_stage2,
                "stage2_minutes": stage_length if has_stage2 and stage_length >= 0 else np.nan,
            }
            for minutes in WINDOWS:
                for label, signal in (("fhr", fhr), ("uc", uc)):
                    missing_pct, sample_count = percent_zero(signal, stage1_end, fs, minutes)
                    row[f"{label}_stage1_last{minutes}_zero_pct"] = missing_pct
                    row[f"{label}_stage1_last{minutes}_n_samples"] = sample_count
            qc_rows.append(row)
            clinical_rows.append({"record_name": name, **record.clinical})
            channel_sets.add(tuple(zip(record.channel_names, record.units)))
        except Exception as exc:
            error_types[type(exc).__name__] += 1

    checks["records_read_equal_RECORDS"] = len(clinical_rows) == len(records)
    clinical = pd.DataFrame(clinical_rows)
    qc = pd.DataFrame(qc_rows)
    fs_values = sorted(qc["fs"].unique().tolist()) if not qc.empty else []
    checks["fs_uniform"] = len(fs_values) == 1 and checks["records_read_equal_RECORDS"]

    # Mixed text/numeric clinical columns are stored as strings rather than lost.
    for field in clinical.columns.difference(["record_name"]):
        nonmissing = clinical[field].dropna()
        if not nonmissing.empty and not nonmissing.map(lambda x: isinstance(x, (int, float, np.integer, np.floating))).all():
            clinical[field] = clinical[field].astype("string")

    header_fields = sorted(field for field in clinical.columns if field != "record_name")
    ph_key = next((field for field in header_fields if field.lower() == "ph"), None)
    ph = pd.to_numeric(clinical[ph_key], errors="coerce") if ph_key else pd.Series(dtype="float64")
    checks["pH_coverage_ge_95pct"] = bool(len(records) and int(ph.notna().sum()) / len(records) >= 0.95)
    bd_key = next((field for field in header_fields if field.lower() == "bdecf"), None)
    bd = pd.to_numeric(clinical[bd_key], errors="coerce") if bd_key else pd.Series(dtype="float64")

    field_missing = "; ".join(f"{field}={safe_count(clinical[field].isna().sum())}" for field in header_fields)
    group_id_fields = [field for field in header_fields if re.search(r"patient|mother|maternal.?id|subject|woman|participant", field, re.IGNORECASE)]
    repeat_hint_fields = [field for field in group_id_fields if clinical[field].dropna().duplicated().any()]

    summary.update({
        "data": f"CTU-UHB 1.0.0; records_read={safe_count(len(clinical))}/{safe_count(len(records))}; manifest_files={safe_count(len(manifest))}; expected_record_files={safe_count(len(expected))}",
        "fs_hz": ",".join(f"{value:g}" for value in fs_values) or "not available",
        "channels_units": " | ".join(",".join(f"{channel}({unit})" for channel, unit in channels) for channels in sorted(channel_sets)) or "not available",
        "duration_min_median_max_minutes": (f"{qc.duration_minutes.min():.2f}/{qc.duration_minutes.median():.2f}/{qc.duration_minutes.max():.2f}" if not qc.empty else "not available"),
        "stage2_records": safe_count(qc.has_stage2.sum()) if not qc.empty else "0",
        "header_fields_missing_counts": field_missing or "not available",
        "pH_missing": safe_count(len(records) - ph.notna().sum()),
        "pH_median_IQR": qstats(ph),
        "pH_min_max": f"{ph.min():.3f}/{ph.max():.3f}" if ph.notna().any() else "not available",
        "pH_thresholds": "; ".join(threshold_summary(ph, threshold, threshold != 7.05) for threshold in (7.05, 7.10, 7.15, 7.20)),
        "BDecf_ge_12": safe_count((bd >= 12).sum()) if bd_key else "not available",
        "FHR_stage1_last60_zero_pct_median_IQR": qstats(qc["fhr_stage1_last60_zero_pct"]) if not qc.empty else "not available",
        "FHR_stage1_last60_gt20_gt50": (f"{safe_count((qc.fhr_stage1_last60_zero_pct > 20).sum())}/{safe_count((qc.fhr_stage1_last60_zero_pct > 50).sum())}" if not qc.empty else "not available"),
        "maternal_repeat_identifier_hint": "YES: " + ",".join(repeat_hint_fields) if repeat_hint_fields else "NO: none identified; dbID denotes record, not mother",
        "read_errors_by_type": ",".join(f"{key}={safe_count(value)}" for key, value in sorted(error_types.items())) or "none",
    })

    if not checks["manifest_complete"]:
        warnings.append("manifest missing or duplicate files/records")
    if not checks["records_read_equal_RECORDS"]:
        warnings.append("one or more records could not be read; derived tables not saved")
    if not checks["pH_coverage_ge_95pct"]:
        warnings.append("pH coverage below threshold")

    if checks["records_read_equal_RECORDS"] and checks["manifest_complete"]:
        output_dir.mkdir(parents=True, exist_ok=False)
        derived_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(derived_dir / "manifest_sha256.csv", index=False)
        clinical.to_parquet(derived_dir / "clinical.parquet", index=False)
        qc.to_parquet(derived_dir / "signal_qc.parquet", index=False)
        summary["outputs"] = "manifest_sha256.csv; clinical.parquet; signal_qc.parquet; outputs/t01_audit/audit_summary.json"
    else:
        summary["outputs"] = "none (read or manifest failure)"

    summary["status"] = "PASS" if all(checks.values()) else "FAIL"
    summary["runtime_s"] = f"{time.perf_counter() - started:.2f}"
    print(render_report(TASK, summary, checks, "; ".join(warnings) or "none"))
    if output_dir.is_dir():
        (output_dir / "audit_summary.json").write_text(json.dumps({"summary": summary, "acceptance": checks, "warnings": warnings}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("P21_CTG_Acidemia/configs/t01.yaml"))
    args = parser.parse_args()
    sys.exit(audit(args.config))

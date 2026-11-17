"""P21-T07b interval metadata audit. Read-only; prints no annotation values.

Use this to identify the cause of an INVALID_INTERVALS abort before any
scientific output is created. Never edits the FHRMA checkout or T02 features.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import loadmat

EXPECTED_REPO_COMMIT = "407a279ea55458b0512fd709646c5646dc2ef669"
FIELDS = ("accelerations", "decelerations", "unreliableSignal", "notToAnalyse")


def metadata_for_field(value, field):
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return {"type_non_numeric": 1}, 0
    if a.size == 0:
        return {}, 0
    if a.ndim == 1 and a.size == 2:
        a = a.reshape(1, 2)
    elif a.ndim == 2 and a.shape[1] == 2:
        pass
    elif a.ndim == 2 and a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T
    else:
        return {"bad_shape": 1}, 0
    flags = Counter()
    for lo, hi in a:
        if not (np.isfinite(lo) and np.isfinite(hi)):
            flags["nonfinite"] += 1
        elif lo < 0 or hi < 0:
            flags["negative_endpoint"] += 1
        elif hi < lo:
            flags["reversed_endpoint"] += 1
        elif hi == lo:
            flags["zero_duration_mask" if field in ("unreliableSignal", "notToAnalyse") else "zero_duration_event"] += 1
    return dict(flags), len(a)


def audit_mat(file: Path, max_details=30):
    obj = loadmat(file, variable_names=["data"], simplify_cells=True)
    if "data" not in obj:
        raise RuntimeError("EXPERT_SCHEMA_ERROR: data missing")
    d = obj["data"]
    if isinstance(d, dict):
        rows = [d]
    elif isinstance(d, np.ndarray):
        rows = d.ravel().tolist()
    elif isinstance(d, list):
        rows = d
    else:
        raise RuntimeError("EXPERT_SCHEMA_ERROR: unsupported data container")
    counts = Counter()
    details = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            counts["nonstruct_record"] += 1
            continue
        filename = str(row.get("filename", "UNKNOWN")).replace("\\", "/").split("/")[-1]
        for field in FIELDS:
            if field not in row:
                counts["missing_field"] += 1
                details.append((index, filename, field, "missing_field", 1))
                continue
            flags, _ = metadata_for_field(row[field], field)
            for reason, n in sorted(flags.items()):
                counts[reason] += n
                details.append((index, filename, field, reason, n))
    hard = sum(v for k, v in counts.items() if k != "zero_duration_mask")
    print("=== P21-T07b ANNOTATION METADATA AUDIT ===")
    print("source=FHRMAdataset/analyses/expertAnalyses.mat; values_not_printed=YES")
    print(f"records_in_mat={len(rows)}")
    for key, n in sorted(counts.items()):
        print(f"{key}={n}")
    for i, filename, field, reason, n in details[:max_details]:
        print(f"ROW_META index={i}; filename={filename}; field={field}; reason={reason}; count={n}")
    if len(details) > max_details:
        print(f"ROW_META_TRUNCATED={len(details)-max_details}")
    status = "ZERO_DURATION_MASK_ONLY_OR_CLEAN" if hard == 0 else "UNSUPPORTED_ANNOTATIONS"
    print(f"T07B_ANNOTATION_AUDIT={status}")
    print("=== END AUDIT ===")
    return 0 if hard == 0 else 2


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="D:/Tuan/AI_training/data_raw/fhrma")
    ns = p.parse_args(argv)
    repo = Path(ns.repo)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, check=True, text=True).stdout.strip()
    if head != EXPECTED_REPO_COMMIT:
        raise RuntimeError(f"FHRMA_COMMIT_CHANGED: {head}")
    file = repo / "FHRMAdataset/analyses/expertAnalyses.mat"
    if not file.is_file():
        raise FileNotFoundError(file)
    return audit_mat(file)


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:
        print("T07B_ANNOTATION_AUDIT_ABORT=" + type(e).__name__ + ": " + str(e))
        code = 2
    raise SystemExit(code)

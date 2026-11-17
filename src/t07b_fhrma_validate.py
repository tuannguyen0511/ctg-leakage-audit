"""P21-T07b. Read-only external validation of the frozen T02 CTG features.

Original T02 `ctg_features.py` and `t02.yaml` are never written.
No waveform or expert annotation values are written to outputs.
Data unavailable is distinct from algorithm failure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
import yaml

PREREG = "85161286492355667fefb0d55d1643cedc5b2736d8234cac1914023c05744c58"
EXPECTED_FEATURE_SHA = "d81bd7924184a4479171fd83f1fd2ce1875843aba2987f52f70460aee02fabf6"
EXPECTED_CFG_SHA = "2c0b3d2e97011db6cbc5fe1fdcc8c6d93ea8ff77ecf2385ab244a512f6b4694d"
EXPECTED_REPO_COMMIT = "407a279ea55458b0512fd709646c5646dc2ef669"
FS = 4
N_BOOT = 2000
BOOT_SEED = 2026
FHRMA_END_MASK_SECONDS = 60
MATLAB_INVALID_EVENT_FRAC = 1 / 3


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for part in iter(lambda: fh.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def require_sha(path: Path, expected: str):
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen file missing: {path}")
    got = sha(path)
    if got != expected:
        raise RuntimeError(f"FROZEN_SHA_MISMATCH: {path}: expected {expected}, got {got}")
    return got


def load_frozen_features(root):
    # The package imports only the original functions, not modified replacements.
    root = Path(root)
    p = root / "P21_CTG_Acidemia/src/ctg_features.py"
    c = root / "P21_CTG_Acidemia/configs/t02.yaml"
    require_sha(p, EXPECTED_FEATURE_SHA)
    require_sha(c, EXPECTED_CFG_SHA)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    m = importlib.import_module("P21_CTG_Acidemia.src.ctg_features")
    cfg = yaml.safe_load(c.read_text(encoding="utf-8"))
    if int(cfg["fs_hz"]) != FS:
        raise RuntimeError("INVALID_T02_FS: expected 4 Hz")
    return m, cfg


def read_fhr(path):
    # MATLAB fhropen.m reads uint32 timestamp, followed by six-byte records.
    raw = np.fromfile(path, dtype=np.uint8)
    if len(raw) < 10 or (len(raw) - 4) % 6:
        raise ValueError(f"BAD_FHR_SIZE: {path.name}, bytes={len(raw)}")
    # Header timestamp intentionally not reported.
    payload = raw[4:].reshape(-1, 6)
    # Little-endian equivalent of MATLAB 'uint16' on Windows PC-A.
    u16 = np.frombuffer(raw[4:].tobytes(), dtype="<u2").reshape(-1, 3)
    fhr1 = u16[:, 0].astype(np.float64) / 4
    fhr2 = u16[:, 1].astype(np.float64) / 4
    toco = payload[:, 4].astype(np.float64) / 2
    return fhr1, fhr2, toco


def intervals(value, name, *, record_id="UNKNOWN", diagnostics=None):
    if value is None:
        return np.zeros((0, 2), dtype=np.float64)
    a = np.asarray(value, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if a.ndim == 1 and a.size == 2:
        a = a.reshape(1, 2)
    elif a.ndim == 2 and a.shape[1] == 2:
        pass
    elif a.ndim == 2 and a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T
    else:
        raise ValueError(f"BAD_INTERVAL_SHAPE {name}: shape={a.shape}")
    # FHRMA exclusion masks may contain zero-duration markers. Such rows select
    # exactly zero samples under MATLAB's half-open interpretation and therefore
    # cannot change any mask. Do not apply this exception to expert events.
    # Never repair/reverse, clip, or silently drop other invalid annotations.
    finite = np.isfinite(a).all(axis=1)
    nonnegative = (a >= 0).all(axis=1)
    reversed_rows = a[:, 1] < a[:, 0]
    zero_duration = a[:, 1] == a[:, 0]
    exclusion_only = name in ("unreliableSignal", "notToAnalyse")
    invalid = (~finite) | (~nonnegative) | reversed_rows
    if not exclusion_only:
        invalid |= zero_duration
    if invalid.any():
        bad = np.flatnonzero(invalid)
        kinds = []
        if (~finite).any():
            kinds.append("nonfinite")
        if (~nonnegative).any():
            kinds.append("negative_endpoint")
        if reversed_rows.any():
            kinds.append("reversed_endpoint")
        if (zero_duration & invalid).any():
            kinds.append("zero_duration_event")
        raise ValueError(
            f"INVALID_INTERVALS {name}: record={record_id}; "
            f"bad_row_indices={bad[:20].tolist()}; bad_count={bad.size}; "
            f"reason={','.join(kinds)}"
        )
    if exclusion_only and zero_duration.any():
        removed = int(zero_duration.sum())
        if diagnostics is not None:
            diagnostics[name] += removed
        # Zero-length masks have no effect on sample eligibility.
        a = a[~zero_duration, :]
    return a


def read_annotations(path, *, diagnostics=None):
    try:
        doc = loadmat(path, variable_names=["data"], simplify_cells=True)
    except NotImplementedError as exc:
        raise RuntimeError("MAT_V73_SCHEMA_UNSUPPORTED: inspect with h5py before parsing") from exc
    if "data" not in doc:
        raise ValueError("EXPERT_SCHEMA_ERROR: missing MATLAB variable 'data'")
    d = doc["data"]
    if isinstance(d, dict):
        rows = [d]
    elif isinstance(d, np.ndarray):
        rows = d.ravel().tolist()
    elif isinstance(d, list):
        rows = d
    else:
        raise ValueError(f"EXPERT_SCHEMA_ERROR: unexpected data type {type(d).__name__}")
    out = {}
    required = ("filename", "trainingData", "baseline", "accelerations", "decelerations",
                "unreliableSignal", "notToAnalyse")
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"EXPERT_SCHEMA_ERROR: non-struct entry index={i}")
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"EXPERT_SCHEMA_ERROR index={i}: missing fields {missing}")
        name = str(row["filename"]).replace("\\", "/").split("/")[-1].strip().casefold()
        if not name or not name.endswith(".fhr"):
            raise ValueError(f"EXPERT_SCHEMA_ERROR index={i}: filename not .fhr")
        if name in out:
            raise ValueError(f"DUPLICATE_EXPERT_FILENAME: {name}")
        flag_a = np.asarray(row["trainingData"]).reshape(-1)
        if flag_a.size != 1 or int(flag_a[0]) not in (0, 1):
            raise ValueError(f"TRAINING_FLAG_UNKNOWN: index={i}")
        flag = int(flag_a[0])
        baseline = np.asarray(row["baseline"], dtype=np.float64).reshape(-1)
        data = dict(row)
        data["trainingData"] = flag
        data["baseline"] = baseline
        for field in ("accelerations", "decelerations", "unreliableSignal", "notToAnalyse"):
            data[field] = intervals(row[field], field, record_id=name, diagnostics=diagnostics)
        out[name] = data
    return out


def get_inventory(repo):
    base = Path(repo) / "FHRMAdataset"
    train = sorted((base / "traindata").glob("*.fhr"))
    test = sorted((base / "testdata").glob("*.fhr"))
    if (len(train), len(test)) != (66, 90):
        raise RuntimeError(f"FHRMA_FILE_COUNTS_CHANGED: train={len(train)}, test={len(test)}")
    seen = set()
    for p in train + test:
        k = p.name.casefold()
        if k in seen:
            raise ValueError(f"TRAIN_TEST_DUPLICATE_FILE_NAME: {k}")
        seen.add(k)
    return train, test


def preflight(root, repo, check_sources=True, test_expert_mat=None):
    repo = Path(repo)
    if check_sources:
        mod, cfg = load_frozen_features(root)
    else:
        mod, cfg = None, None
    if not (repo / ".git").is_dir():
        raise RuntimeError("NOT_A_GIT_CLONE: FHRMA")
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    head = result.stdout.strip()
    if head != EXPECTED_REPO_COMMIT:
        raise RuntimeError(f"FHRMA_COMMIT_CHANGED: {head}")
    tr, te = get_inventory(repo)
    p = repo / "FHRMAdataset/analyses/expertAnalyses.mat"
    if not p.is_file():
        raise FileNotFoundError(f"MISSING_EXPERT_MAT: {p}")
    mask_diagnostics = Counter()
    exp = read_annotations(p, diagnostics=mask_diagnostics)
    test_ground_truth_sha = None
    if test_expert_mat is not None:
        extra_path = Path(test_expert_mat)
        if not extra_path.is_file():
            raise FileNotFoundError(f"TEST_EXPERT_MAT_MISSING: {extra_path}")
        test_ground_truth_sha = sha(extra_path)
        supplement = read_annotations(extra_path, diagnostics=mask_diagnostics)
        approved_test_names = {f.name.casefold() for f in te}
        for key, entry in supplement.items():
            if key not in approved_test_names or entry["trainingData"] != 0:
                raise ValueError(f"TEST_GROUND_TRUTH_SPLIT_VIOLATION: {key}")
            existing = exp.get(key)
            if (existing is not None and existing["baseline"].size > 0
                    and np.isfinite(existing["baseline"]).any()):
                raise RuntimeError(f"TEST_ANNOTATION_CONFLICT: {key}")
            exp[key] = entry
    details = {"train_files": len(tr), "test_files": len(te),
               "expert_mat_sha256": sha(p), "external_test_mat_sha256": test_ground_truth_sha, "expert_entries": len(exp),
               "train_annotation_entries": 0, "test_annotation_entries": 0,
               "train_labeled_candidates": 0, "test_labeled_candidates": 0,
               "train_missing_entries": 0, "test_missing_entries": 0,
               "zero_duration_exclusion_rows_ignored": int(sum(mask_diagnostics.values())),
               "zero_duration_unreliableSignal_rows": int(mask_diagnostics["unreliableSignal"]),
               "zero_duration_notToAnalyse_rows": int(mask_diagnostics["notToAnalyse"])}
    for split, paths in (("train", tr), ("test", te)):
        for path in paths:
            ann = exp.get(path.name.casefold())
            if ann is None:
                details[f"{split}_missing_entries"] += 1
                continue
            details[f"{split}_annotation_entries"] += 1
            if ann["trainingData"] != (1 if split == "train" else 0):
                raise RuntimeError(f"TRAIN_TEST_FLAG_MISMATCH: {path.name}")
            if ann["baseline"].size > 0 and np.isfinite(ann["baseline"]).any():
                details[f"{split}_labeled_candidates"] += 1
    extras = sorted(set(exp).difference({p.name.casefold() for p in tr + te}))
    if extras:
        raise RuntimeError(f"EXPERT_RECORDS_NOT_MAPPED: count={len(extras)}")
    return mod, cfg, exp, tr, te, details


def mask_intervals(mask, ivals, fs=FS):
    n = len(mask)
    for a, b in ivals:
        start = max(0, min(n, int(np.round(a * 60 * fs))))
        end = max(0, min(n, int(np.round(b * 60 * fs))))
        if end > start:
            mask[start:end] = False
    return mask


def external_wrapper_events(mod, fhr, baseline, cfg):
    """Recover [start,end) events using exactly compute_features' condition/runs.

    This is not a new peak detector. Count equality to compute_features is enforced.
    """
    if not np.isfinite(baseline):
        return {"accel": [], "decel": []}
    m = int(np.ceil(float(cfg["event_min_seconds"]) * int(cfg["fs_hz"])))
    ans = {}
    for key, threshold, sign in (("accel", "accel_threshold_bpm", 1),
                                 ("decel", "decel_threshold_bpm", -1)):
        condition = (fhr >= baseline + cfg[threshold]) if sign == 1 else (fhr <= baseline - cfg[threshold])
        s, e = mod._runs(condition & np.isfinite(fhr))
        ans[key] = [(int(a), int(b)) for a, b in zip(s, e) if b - a >= m]
    return ans


def evaluable_events(events, valid_mask):
    ans = []
    for a, b in events:
        if not (0 <= a < b <= len(valid_mask)):
            raise ValueError("EVENT_INTERVAL_OUTSIDE_SIGNAL")
        # Source: FHRMA statscompare.m filteraccident(): >1/3 unavailable excluded.
        invalid_frac = 1.0 - float(np.mean(valid_mask[a:b]))
        if invalid_frac <= MATLAB_INVALID_EVENT_FRAC:
            ans.append((a, b))
    return ans


def expert_events_samples(ann, field, n):
    iv = ann[field]
    out = []
    for start_m, end_m in iv:
        a = int(np.round(float(start_m) * 60 * FS))
        b = int(np.round(float(end_m) * 60 * FS))
        # Fail on gross misalignment rather than clipping silently.
        if a >= n + FS or b > n + FS or a < 0:
            raise ValueError(f"EXPERT_EVENT_OUT_OF_BOUNDS: {field}")
        a, b = max(0, min(a, n)), max(0, min(b, n))
        if b > a:
            out.append((a, b))
    return out


def match_events(pred, truth):
    """Maximum-cardinality, one-to-one matching, overlap >= 50% truth duration.

    Threshold is expert-normalized, not union-normalized. Assignment prioritizes
    number of matches, then total overlap; order does not affect TP.
    """
    npred, ntrue = len(pred), len(truth)
    if not npred or not ntrue:
        return 0, npred, ntrue
    n = npred + ntrue
    cost = np.zeros((n, n), dtype=np.float64)
    for i, (ps, pe) in enumerate(pred):
        for j, (ts, te) in enumerate(truth):
            overlap = max(0, min(pe, te) - max(ps, ts))
            if overlap > 0 and overlap >= 0.5 * (te - ts):
                # The bonus guarantees cardinality > summed overlaps.
                cost[i, j] = -1_000_000_000.0 - float(overlap)
    rr, cc = linear_sum_assignment(cost)
    tp = sum(int(i < npred and j < ntrue and cost[i, j] < 0) for i, j in zip(rr, cc))
    return tp, npred - tp, ntrue - tp


def f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return (2 * tp / den) if den else None


def describe_record(record_path, split, ann, mod, cfg):
    rid = record_path.stem
    row = {"record_id": rid, "split": split, "label_status": "UNAVAILABLE",
           "baseline_mae_bpm": "", "baseline_labeled_samples": 0,
           "accel_tp": "", "accel_fp": "", "accel_fn": "", "accel_f1": "",
           "decel_tp": "", "decel_fp": "", "decel_fn": "", "decel_f1": ""}
    if ann is None or ann["baseline"].size == 0 or not np.isfinite(ann["baseline"]).any():
        return row, None
    fhr1, fhr2, uc = read_fhr(record_path)
    n = len(fhr1)
    ref = ann["baseline"]
    if len(ref) != n:
        raise RuntimeError(f"REFERENCE_BASELINE_LENGTH_MISMATCH: {rid}, length={len(ref)}, signal={n}")
    # Same input selector as FHRMA/preprocess.m: FHR=max(FHR1,FHR2).
    fhr = np.maximum(fhr1, fhr2)
    feed_ok = np.ones(n, dtype=bool)
    mask_intervals(feed_ok, ann["unreliableSignal"])
    fhr[~feed_ok] = np.nan
    fhr_clean, uc_clean = mod.clean_signals(fhr, uc, cfg)
    features = mod.compute_features(fhr_clean, uc_clean, cfg)
    base = float(features["baseline"])
    pred = external_wrapper_events(mod, fhr_clean, base, cfg)
    for kind in ("accel", "decel"):
        count = features[f"n_{kind}"]
        if not np.isfinite(count) or len(pred[kind]) != int(count):
            raise RuntimeError(f"WRAPPER_COUNT_MISMATCH: {rid} {kind}, features={count}, events={len(pred[kind])}")
    # Expert-confirmed non-evaluable intervals, plus one final minute from evaluateaam.m.
    valid = np.isfinite(fhr_clean) & feed_ok
    mask_intervals(valid, ann["notToAnalyse"])
    valid[max(0, n - FS * FHRMA_END_MASK_SECONDS):] = False
    valid &= np.isfinite(ref) & (ref > 0)
    count = int(np.sum(valid))
    if count == 0 or not np.isfinite(base):
        row["label_status"] = "UNSCORABLE_NO_VALID_BASELINE"
        return row, None
    absolute_sum = float(np.abs(ref[valid] - base).sum())
    row["baseline_mae_bpm"] = absolute_sum / count
    row["baseline_labeled_samples"] = count
    data = {"baseline_abs_sum": absolute_sum, "baseline_n": count}
    for kind, field in (("accel", "accelerations"), ("decel", "decelerations")):
        expert = expert_events_samples(ann, field, n)
        truth_keep = evaluable_events(expert, valid)
        pred_keep = evaluable_events(pred[kind], valid)
        tp, fp, fn = match_events(pred_keep, truth_keep)
        data.update({f"{kind}_tp": tp, f"{kind}_fp": fp, f"{kind}_fn": fn})
        row.update({f"{kind}_tp": tp, f"{kind}_fp": fp, f"{kind}_fn": fn,
                    f"{kind}_f1": "" if f1(tp, fp, fn) is None else f1(tp, fp, fn)})
    row["label_status"] = "SCORED"
    return row, data


def aggregate(items):
    if not items:
        return {"baseline_mae_bpm": None, "accel_f1": None, "decel_f1": None}
    d = {"baseline_abs_sum": sum(i["baseline_abs_sum"] for i in items),
         "baseline_n": sum(i["baseline_n"] for i in items)}
    d["baseline_mae_bpm"] = d["baseline_abs_sum"] / d["baseline_n"] if d["baseline_n"] else None
    for k in ("accel", "decel"):
        tp, fp, fn = (sum(i[f"{k}_{s}"] for i in items) for s in ("tp", "fp", "fn"))
        d.update({f"{k}_tp": tp, f"{k}_fp": fp, f"{k}_fn": fn, f"{k}_f1": f1(tp, fp, fn)})
    return d


def bootstrap(items, nboot=N_BOOT, seed=BOOT_SEED):
    if not items:
        return {k: {"estimate": None, "ci95_cluster_record": None, "valid_replicates": 0}
                for k in ("baseline_mae_bpm", "accel_f1", "decel_f1")}
    point = aggregate(items)
    rng = np.random.default_rng(seed)
    sampled = {k: [] for k in ("baseline_mae_bpm", "accel_f1", "decel_f1")}
    for _ in range(nboot):
        ix = rng.integers(len(items), size=len(items))
        z = aggregate([items[j] for j in ix])
        for k in sampled:
            if z[k] is not None and np.isfinite(z[k]):
                sampled[k].append(z[k])
    result = {}
    for k, values in sampled.items():
        ci = [float(z) for z in np.quantile(values, [0.025, 0.975])] if values else None
        result[k] = {"estimate": point[k], "ci95_cluster_record": ci,
                     "valid_replicates": len(values)}
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="D:/Tuan/AI_training/data_raw/fhrma")
    ap.add_argument("--root", default="D:/Tuan/AI_training")
    ap.add_argument("--output", default="D:/Tuan/AI_training/P21_CTG_Acidemia/outputs/t07_fhrma")
    ap.add_argument("--stage", choices=["inspect", "run"], default="inspect")
    ap.add_argument("--test-expert-mat", default=None, help="Optional officially authorized test expert labels; never overwrite FHRMA raw data")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    root, repo, output = map(Path, (args.root, args.repo, args.output))
    if output.exists():
        raise RuntimeError(f"STOP_OUTPUT_EXISTS: {output}")
    mod, cfg, ann, train, test, details = preflight(root, repo, test_expert_mat=args.test_expert_mat)
    print("T07B_INPUT_SHA_PASS T02_feature=" + EXPECTED_FEATURE_SHA[:8] +
          " T02_config=" + EXPECTED_CFG_SHA[:8] + " FHRMA_commit=" + EXPECTED_REPO_COMMIT[:8])
    print("T07B_SCHEMA: " + "; ".join(f"{k}={v}" for k, v in details.items()
                                         if k not in ("expert_mat_sha256", "external_test_mat_sha256")))
    print("T07B_EXPERT_MAT_SHA8=" + details["expert_mat_sha256"][:8])
    print("T07B_TEST_GT_ACCESS=" + ("EXTERNAL_AUTHORIZED_FILE" if details["external_test_mat_sha256"] else
                                     ("PUBLIC_ONLY" if details["test_labeled_candidates"] else "NOT_AVAILABLE")))
    if args.stage == "inspect":
        print("T07B_INSPECT_PASS; no outputs written")
        return 0
    fields = ["record_id", "split", "label_status", "baseline_mae_bpm",
              "baseline_labeled_samples", "accel_tp", "accel_fp", "accel_fn", "accel_f1",
              "decel_tp", "decel_fp", "decel_fn", "decel_f1"]
    rows, scoring = [], {"train": [], "test": []}
    for split, paths in (("train", train), ("test", test)):
        for path in paths:
            record = ann.get(path.name.casefold())
            row, data = describe_record(path, split, record, mod, cfg)
            rows.append(row)
            if data is not None:
                scoring[split].append(data)
    split_result = {}
    for split, paths in (("train", train), ("test", test)):
        scored = scoring[split]
        split_result[split] = {"record_count": len(paths), "scored_count": len(scored),
                               "unscored_count": len(paths) - len(scored),
                               "metrics": bootstrap(scored, N_BOOT, BOOT_SEED),
                               "counts": aggregate(scored)}
    enough = all(len(scoring[s]) == len(paths) for s, paths in (("train", train), ("test", test)))
    test_metrics = split_result["test"]["metrics"]
    thresholds_available = all(test_metrics[k]["estimate"] is not None for k in test_metrics)
    if not enough:
        status = "GROUND_TRUTH_UNAVAILABLE_OR_UNSCORABLE"
        decision = "NOT_ASSESSABLE"
        exit_code = 4
    elif not thresholds_available:
        status = "TEST_METRICS_UNDEFINED"
        decision = "NOT_ASSESSABLE"
        exit_code = 4
    else:
        decision = ("PASS" if test_metrics["baseline_mae_bpm"]["estimate"] <= 5.0
                    and test_metrics["accel_f1"]["estimate"] >= 0.60
                    and test_metrics["decel_f1"]["estimate"] >= 0.60 else "FAIL")
        status = "PASS" if decision == "PASS" else "THRESHOLD_FAIL"
        exit_code = 0 if decision == "PASS" else 3
    report = {
        "task": "P21-T07b", "status": status, "decision": decision,
        "prereg_v1_4_sha256": PREREG, "fhrma_git_commit": EXPECTED_REPO_COMMIT,
        "reference_data_sha256": details["expert_mat_sha256"],
        "authorized_external_test_mat_sha256": details["external_test_mat_sha256"],
        "t02_feature_sha256": EXPECTED_FEATURE_SHA, "t02_config_sha256": EXPECTED_CFG_SHA,
        "sampling_hz": FS, "baseline_model": "scalar T02 baseline broadcast across labeled times",
        "event_pred": "exact T02 threshold and _runs; wrapper count-checked",
        "aggregation": "sample-weighted baseline MAE; micro-F1 from pooled 1-to-1 TP/FP/FN",
        "event_match": "one-to-one maximum cardinality, overlap >= 50% expert duration",
        "event_exclusions": "FHRMA >1/3 invalid support; notToAnalyse; unreliableSignal; last 60 s",
        "baseline_samples": "expert finite >0 AND valid measured FHR AND evaluable intervals",
        "unavailable_test_labels_are_not_negatives": True,
        "bootstrap": {"type": "record-cluster", "replicates": N_BOOT, "seed": BOOT_SEED},
        "inventory": details, "splits": split_result,
        "thresholds_test": {"baseline_mae_bpm_max": 5.0, "accel_f1_min": 0.60, "decel_f1_min": 0.60},
        "runtime_s": time.perf_counter() - t0,
    }
    # All computations complete before opening the final output directory.
    import tempfile
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".t07b_stage_", dir=output.parent) as stage_name:
        stage = Path(stage_name)
        with open(stage / "per_record.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        (stage / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if output.exists():
            raise RuntimeError(f"STOP_OUTPUT_EXISTS: {output}")
        stage.rename(output)
    print("=== REPORT P21-T07b ===")
    print(f"status={status}; decision={decision}; cpu_only=YES; original_features_modified=NO")
    print(f"PREREG_v1.4_SHA256={PREREG}")
    print("train_scored={}/66; test_scored={}/90".format(len(scoring["train"]), len(scoring["test"])))
    for s in ("train", "test"):
        x = split_result[s]["metrics"]
        bits = []
        for k in ("baseline_mae_bpm", "accel_f1", "decel_f1"):
            v = x[k]
            if v["estimate"] is None:
                bits.append(f"{k}=DATA_NOT_AVAILABLE")
            else:
                ci = v["ci95_cluster_record"]
                bits.append(f"{k}={v['estimate']:.4f}" + (f"[{ci[0]:.4f},{ci[1]:.4f}]" if ci else "[CI_UNAVAILABLE]"))
        print(s + ": " + "; ".join(bits))
    print(f"bootstrap_cluster_record={N_BOOT}; event_match=one_to_one_overlap_0.5_expert_duration")
    print("expert_mat_sha8=" + details["expert_mat_sha256"][:8])
    print("original_ctg_features_sha8=" + EXPECTED_FEATURE_SHA[:8])
    print("metrics.json_sha8=" + sha(output / "metrics.json")[:8])
    print("per_record.csv_sha8=" + sha(output / "per_record.csv")[:8])
    print("runtime_s={:.2f}".format(time.perf_counter() - t0))
    print("=== END REPORT P21-T07b ===")
    return exit_code


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:
        print("T07B_ABORT=" + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        code = 2
    print(f"P21_T07B_EXIT_CODE={code}")
    sys.exit(code)

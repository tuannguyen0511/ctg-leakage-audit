"""P21-T05 (PREREG 1.3): CPU-only, resumable, single-factor leakage audit.

Does not touch frozen T03/T04 artifacts. L1 S_seg is newly fitted because the
frozen record-level S is not a segment-level comparison. Every fit is saved
once, keyed by protocol/model/seed/repeat/fold, then independently verified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
import yaml

from common.io_wfdb import read_record
from common.metrics import discrimination
from P21_CTG_Acidemia.src.ctg_features import clean_signals, compute_features, build_dl_1hz
from P21_CTG_Acidemia.src import t03_baselines as t03
from P21_CTG_Acidemia.src import t04_dl as t04

PREREG = "40573556ecf4664c430430eb99eaaa20d1d0958aa32647cd30b7144434d048eb"
PROTOCOLS = ("S_seg", "L1", "L2", "L3")
MODELS = ("M1", "M2")
REPEATS = (0, 1, 2)
SEEDS = (0, 1, 2)
FOLDS = range(5)
OUTPUT_OOF = ("oof_S_seg.parquet", "oof_L1.parquet", "oof_L2.parquet",
              "oof_L3.parquet", "oof_L4.parquet")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def atomic_new(path, writer):
    """Publish one complete immutable artifact; hard link does not overwrite."""
    path = Path(path)
    check(not path.exists(), f"STOP: existing artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="t05_pending_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)  # Fails if path appeared while the unit was training.
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_json(path, data):
    payload = json.dumps(json_ready(data), indent=2, sort_keys=True,
                         allow_nan=False).encode("utf-8")
    atomic_new(path, lambda handle: handle.write(payload))


def config_check(cfg):
    expected = {
        "task": "P21-T05", "run_name": "t05_leakage", "prereg_version": "1.3",
        "prereg_sha256": PREREG, "repeats": [0, 1, 2], "outer_folds": 5,
        "seeds": [0, 1, 2], "segment_minutes": 10,
        "segment_step_minutes": 5, "segment_exclude_fhr_missing_pct": 50,
        "segment_cv_seed_base": 2026, "smote_k": 5, "smote_seed": 42,
        "oversample_seed": 42, "selection_seed": 42,
        "bootstrap_resamples": 10000, "bootstrap_seed": 42,
        "negative_control_bounds": [0.4, 0.6], "cpu_threads": 4,
        "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 32,
        "max_epochs": 60, "patience": 10, "validation_fraction": 0.2,
        "m1_learning_rate": [0.03, 0.1], "m1_max_depth": [2, 3],
        "m1_min_samples_leaf": [10, 20, 40],
    }
    check(set(cfg) == set(expected) | {"project_dir", "raw_dir", "derived_dir",
                                     "output_dir", "frozen_hash_file"},
          "Unexpected T05 config keys")
    for key, value in expected.items():
        check(cfg[key] == value, f"T05 config differs from preregistered {key}")
    for key in ("project_dir", "raw_dir", "derived_dir", "output_dir", "frozen_hash_file"):
        check(Path(cfg[key]).is_absolute(), f"{key} must be absolute")


def t03_cfg(cfg):
    return {"m1_learning_rate": cfg["m1_learning_rate"],
            "m1_max_depth": cfg["m1_max_depth"],
            "m1_min_samples_leaf": cfg["m1_min_samples_leaf"], "cv_inner_folds": 5}


def t02_cfg(project):
    filename = project / "configs" / "t02.yaml"
    value = yaml.safe_load(filename.read_text(encoding="utf-8"))
    check(value["run_name"] == "t02_inputs" and value["windows_minutes"] == [30, 60]
          and value["fs_hz"] == 4 and value["dl_fs_hz"] == 1
          and value["exclusion_fhr_nan_pct_w30"] == 50,
          "T02 cleaning settings do not match locked protocol")
    return value


def read_frozen_shas(lock):
    result = {}
    for line in Path(lock).read_text(encoding="utf-8-sig").splitlines():
        if not re.search(r"[/\\]A_primary[/\\]oof\.parquet", line, re.I):
            continue
        tokens = re.findall(r"(?<![0-9a-f])[0-9a-f]{8}(?:[0-9a-f]{56})?(?![0-9a-f])", line, re.I)
        check(len(tokens) == 1, f"Ambiguous A_primary OOF lock: {line}")
        check("A_primary" not in result or result["A_primary"] == tokens[0].lower(),
              "Conflicting T04 A_primary OOF lock")
        result["A_primary"] = tokens[0].lower()
    check("A_primary" in result, "T04 frozen SHA lock does not identify A_primary OOF")
    return result["A_primary"]


def load_inputs(cfg):
    project, derived = Path(cfg["project_dir"]), Path(cfg["derived_dir"])
    paths = {name: derived / name for name in ("labels.parquet", "features_w30.parquet",
                                               "dl_w30_1hz.npz", "split.csv")}
    paths["t03_oof"] = project / "outputs" / "t03_baselines" / "A_primary" / "oof.parquet"
    paths["t04_oof"] = project / "outputs" / "t04_dl" / "A_primary" / "oof.parquet"
    paths["t02_yaml"] = project / "configs" / "t02.yaml"
    for name, path in paths.items():
        check(path.is_file(), f"Missing frozen input {name}: {path}")
    lock = read_frozen_shas(cfg["frozen_hash_file"])
    check(sha(paths["t04_oof"]).startswith(lock), "T04 frozen OOF SHA mismatch")
    labels = pd.read_parquet(paths["labels.parquet"])
    check(len(labels) == 552 and labels.record_id.astype(str).is_unique,
          "T02 labels must contain 552 records")
    labels.record_id = labels.record_id.astype(str)
    labels = labels.loc[~labels.excluded.astype(bool)].set_index("record_id").sort_index()
    check(len(labels) == 518 and int(labels.y705.sum()) == 30,
          "Eligible cohort must be 518 records with 30 positives")
    features = pd.read_parquet(paths["features_w30.parquet"])
    features.record_id = features.record_id.astype(str)
    check(features.record_id.is_unique and set(labels.index).issubset(features.record_id),
          "T02 features do not align to eligible labels")
    features = features.set_index("record_id").loc[labels.index, list(t03.FEATURES)]
    x_feature = features.to_numpy(dtype=np.float64)
    check(not np.isinf(x_feature).any(), "Infinite feature values")
    with np.load(paths["dl_w30_1hz.npz"], allow_pickle=False) as data:
        source_ids = data["record_id"].astype(str)
        raw_dl = data["X"]
    check(raw_dl.shape == (552, 3, 1800) and len(set(source_ids)) == 552,
          "T02 DL input shape or IDs differ")
    pos = {rid: i for i, rid in enumerate(source_ids)}
    tensor = t04.fixed_normalize(raw_dl[[pos[rid] for rid in labels.index]])
    splits = pd.read_csv(paths["split.csv"], dtype={"record_id": str})
    t04.check_split(splits, labels.index.to_numpy(dtype=str))
    base = pd.read_parquet(paths["t03_oof"])
    deep = pd.read_parquet(paths["t04_oof"])
    for frame, names in ((base, ["p_M1"]), (deep, [f"p_M2_s{s}" for s in SEEDS])):
        check({"record_id", "repeat", "outer_fold", "y", *names}.issubset(frame.columns),
              "Frozen S OOF columns missing")
        frame.record_id = frame.record_id.astype(str)
        frame = frame.loc[frame.repeat.isin(REPEATS)]
        check(len(frame) == 518 * 3 and frame.groupby(["record_id", "repeat"]).size().eq(1).all(),
              "Frozen S OOF coverage is incomplete")
        check(np.isfinite(frame[names].to_numpy(dtype=float)).all(), "Nonfinite frozen OOF")
    merged = base.loc[base.repeat.isin(REPEATS)].merge(
        deep.loc[deep.repeat.isin(REPEATS),
                 ["record_id", "repeat", "outer_fold", "y", *[f"p_M2_s{s}" for s in SEEDS]]],
        on=["record_id", "repeat"], suffixes=("_t03", "_t04"), validate="one_to_one")
    check(len(merged) == 518 * 3 and
          (merged.y_t03 == merged.y_t04).all() and
          (merged.outer_fold_t03 == merged.outer_fold_t04).all(),
          "Frozen S T03/T04 label or fold mismatch")
    merged = merged.rename(columns={"y_t03": "y", "outer_fold_t03": "outer_fold"})
    check(set(merged.record_id) == set(labels.index), "S OOF record identity mismatch")
    return {"labels": labels, "features": x_feature, "tensor": tensor, "ids": labels.index.to_numpy(),
            "y": labels.y705.to_numpy(dtype=np.int64), "splits": splits,
            "s_oof": merged, "input_shas": {k: sha(v) for k, v in paths.items()},
            "t02_cfg": t02_cfg(project), "t04_lock": lock}


def stage1_segments(fhr, uc, fs=4, minutes=10, step_minutes=5, max_missing=50):
    """Full windows only: [0,10),[5,15),... minutes, no partial tails."""
    width, step = int(minutes * 60 * fs), int(step_minutes * 60 * fs)
    accepted, dropped = [], 0
    for start in range(0, max(0, len(fhr) - width + 1), step):
        end = start + width
        missing = float(100.0 * np.mean(~np.isfinite(fhr[start:end])))
        if missing > max_missing:
            dropped += 1
        else:
            accepted.append((start, end, missing))
    return accepted, dropped


def build_segments(cfg, data, out):
    """Create a frozen cache from raw stage-1 traces; never alter raw data."""
    cache, meta_path = out / "segments_cache.npz", out / "segments_meta.parquet"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            segments = {key: saved[key] for key in saved.files}
        check(set(segments) == {"row_id", "record_id", "y", "features", "tensor", "meta_json"},
              "Cached segment schema changed")
        meta = pd.DataFrame(json.loads(str(segments.pop("meta_json").item())))
        if meta_path.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(meta_path), meta, check_dtype=False)
        else:
            atomic_new(meta_path, lambda f: meta.to_parquet(f, index=False))
    else:
        idlist, feat, tensors, ys, records, rows = [], [], [], [], [], []
        raw_dir = Path(cfg["raw_dir"])
        check((raw_dir / "RECORDS").is_file(), "Missing raw RECORDS index")
        label_map = data["labels"]
        for idx, rid in enumerate(data["ids"]):
            rec = read_record(raw_dir, str(rid))
            check(rec.fs == 4 and {"FHR", "UC"}.issubset(set(map(str.upper, rec.channel_names))),
                  f"Invalid raw CTG channels: {rid}")
            channels = {name.upper(): i for i, name in enumerate(rec.channel_names)}
            fhr, uc = clean_signals(rec.signal[:, channels["FHR"]],
                                    rec.signal[:, channels["UC"]], data["t02_cfg"])
            stage = rec.clinical.get("Pos. II.st.", np.nan)
            try:
                stage = float(stage)
            except (TypeError, ValueError):
                stage = np.nan
            end = len(fhr) if not np.isfinite(stage) or stage < 0 else int(stage)
            check(0 <= end <= len(fhr), "Invalid stage-2 onset in raw trace")
            kept, dropped = stage1_segments(fhr[:end], uc[:end])
            rows.append({"record_id": rid, "n_segments_total": len(kept) + dropped,
                         "n_segments": len(kept), "n_dropped_missing": dropped})
            check(len(kept) > 0, f"Record {rid} has no valid full 10-min segment")
            for start, stop, _ in kept:
                idlist.append(f"{rid}__{start:08d}")
                records.append(rid)
                ys.append(int(label_map.loc[rid, "y705"]))
                f = compute_features(fhr[start:stop], uc[start:stop], data["t02_cfg"])
                feat.append([f[name] for name in t03.FEATURES])
                tensor = build_dl_1hz(fhr[start:stop], uc[start:stop], 10, data["t02_cfg"])
                tensors.append(tensor)
            if (idx + 1) % 100 == 0:
                print(f"SEGMENT_PREP={idx + 1}/518", flush=True)
        meta = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)
        raw_tensor = np.stack(tensors).astype(np.float32)
        segments = {"row_id": np.asarray(idlist, dtype=str),
                    "record_id": np.asarray(records, dtype=str),
                    "y": np.asarray(ys, dtype=np.int8),
                    "features": np.asarray(feat, dtype=np.float64),
                    "tensor": t04.fixed_normalize(raw_tensor)}
        check(not meta.record_id.duplicated().any() and len(meta) == 518,
              "Segment metadata must cover 518 eligible records")
        atomic_new(cache, lambda f: np.savez_compressed(
            f, **segments, meta_json=np.asarray(meta.to_json(orient="records"))))
        if meta_path.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(meta_path), meta, check_dtype=False)
        else:
            atomic_new(meta_path, lambda f: meta.to_parquet(f, index=False))
    check(len(meta) == 518 and meta.n_segments.min() >= 1 and
          int(meta.n_segments.sum()) == len(segments["row_id"]) and
          len(set(segments["row_id"])) == len(segments["row_id"]),
          "Cached segments do not agree with record-level metadata")
    check(np.isfinite(segments["tensor"]).all() and not np.isinf(segments["features"]).any(),
          "Segment arrays contain an invalid number")
    check(set(segments["record_id"]) == set(data["ids"]), "Segment cohort mismatch")
    return segments, meta


def smote_all(x, y, k=5, seed=42):
    """Standard continuous SMOTE (k nearest minority neighbors; seed 42)."""
    counts = np.bincount(y.astype(int), minlength=2)
    minority = int(np.argmin(counts))
    positives = np.flatnonzero(y == minority)
    check(len(positives) > k, "SMOTE minority class smaller than k+1")
    n_new = int(counts.max() - counts.min())
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(x[positives])
    nearby = neighbors.kneighbors(x[positives], return_distance=False)[:, 1:]
    rng = np.random.RandomState(seed)
    # Match imbalanced-learn's SMOTE neighbor/row sampling order: one randint
    # over the flattened minority-by-neighbor matrix, then uniform steps.
    samples_indices = rng.randint(0, nearby.size, size=n_new)
    bases = samples_indices // k
    choices = samples_indices % k
    weights = rng.uniform(0.0, 1.0, size=n_new)
    base_rows = positives[bases]
    other_rows = positives[nearby[bases, choices]]
    synth = x[base_rows] + weights[:, None] * (x[other_rows] - x[base_rows])
    return np.vstack([x, synth]), np.r_[y, np.full(n_new, minority, dtype=np.int64)], n_new


def random_oversample(rows, y, seed=42):
    counts = np.bincount(y.astype(int), minlength=2)
    minority = int(np.argmin(counts))
    minority_rows = np.flatnonzero(y == minority)
    rng = np.random.RandomState(seed)
    cloned = rng.choice(minority_rows, int(counts.max() - counts.min()), replace=True)
    return np.r_[rows, rows[cloned]], np.r_[y, y[cloned]], len(cloned)


def make_folds(row_y, repeat, base_seed=2026):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=base_seed + repeat)
    return [(fit, holdout) for fit, holdout in splitter.split(np.zeros(len(row_y)), row_y)]


def grouped_segment_split(segment_record_ids, train_indices, test_indices):
    train = set(np.asarray(segment_record_ids)[train_indices])
    test = set(np.asarray(segment_record_ids)[test_indices])
    return not bool(train & test)


def group_validation_indices(ids, y, repeat, fold, seed, fraction=0.2):
    """S_seg validation at record level, retaining all a record's segments."""
    ids = np.asarray(ids, dtype=str)
    y = np.asarray(y, dtype=np.int64)
    labels = pd.DataFrame({"rid": ids, "y": y}).drop_duplicates("rid")
    check(not labels.rid.duplicated().any() and
          pd.DataFrame({"rid": ids, "y": y}).groupby("rid").y.nunique().eq(1).all(),
          "A record has inconsistent segment labels")
    split = StratifiedShuffleSplit(n_splits=1, test_size=fraction,
                                   random_state=1000 * repeat + 10 * fold + seed)
    fit_rec, val_rec = next(split.split(np.zeros(len(labels)), labels.y))
    fit, val = set(labels.rid.iloc[fit_rec]), set(labels.rid.iloc[val_rec])
    fi, vi = np.flatnonzero(np.isin(ids, list(fit))), np.flatnonzero(np.isin(ids, list(val)))
    check(not set(ids[fi]) & set(ids[vi]) and len(fi) + len(vi) == len(ids),
          "Grouped validation leaked a record")
    return fi, vi


def train_m2(x_train, y_train, x_test, y_test, seed, repeat, fold, cfg,
             group_ids=None, use_outer_test_for_checkpoint=False):
    """T04 M2CNN, Adam, BCEWithLogits, patience, and fixed normalization.

    L3 changes exactly one element: the checkpoint selection score comes from
    outer-test AUROC rather than validation AUROC. Its 20% holdout stays unused.
    """
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    check(not torch.cuda.is_available() or os.environ["CUDA_VISIBLE_DEVICES"] == "-1",
          "GPU execution is forbidden")
    started = time.perf_counter()
    t04.seed_everything(seed)
    y_train, y_test = np.asarray(y_train, dtype=np.int64), np.asarray(y_test, dtype=np.int64)
    if group_ids is None:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=float(cfg["validation_fraction"]),
            random_state=1000 * repeat + 10 * fold + seed)
        fit_idx, val_idx = next(splitter.split(np.zeros(len(y_train)), y_train))
    else:
        fit_idx, val_idx = group_validation_indices(group_ids, y_train, repeat, fold, seed)
    check(len(np.unique(y_train[fit_idx])) == len(np.unique(y_train[val_idx])) == 2,
          "M2 training or validation lacks a class")
    model = t04.make_model("M2", in_channels=2).cpu()
    n_pos = int(y_train[fit_idx].sum())
    n_neg = int(len(fit_idx) - n_pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]),
                                 weight_decay=float(cfg["weight_decay"]))
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x_train[fit_idx], dtype=np.float32)),
        torch.from_numpy(y_train[fit_idx].astype(np.float32))),
        batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0, generator=generator)
    best_score, best_state, best_epoch, wait, stopped = -np.inf, None, 0, 0, 0
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            check(bool(torch.isfinite(loss)), "M2 nonfinite training loss")
            loss.backward()
            optimizer.step()
        if use_outer_test_for_checkpoint:
            select_y, select_x = y_test, x_test
        else:
            select_y, select_x = y_train[val_idx], x_train[val_idx]
        select_p = t04.predict_prob(model, select_x, int(cfg["batch_size"]))
        check(np.isfinite(select_p).all(), "Nonfinite M2 checkpoint selection score")
        score = float(roc_auc_score(select_y, select_p))
        if score > best_score:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        stopped = epoch
        if wait >= int(cfg["patience"]):
            break
    check(best_state is not None, "M2 has no checkpoint")
    model.load_state_dict(best_state)
    p = t04.predict_prob(model, x_test, int(cfg["batch_size"]))
    check(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid M2 OOF scores")
    return p, {"best_epoch": best_epoch, "epoch_stop": stopped,
               "checkpoint_score": best_score,
               "checkpoint_source": "outer_test_LEAKAGE" if use_outer_test_for_checkpoint else "validation",
               "n_train": int(len(fit_idx)), "n_val": int(len(val_idx)),
               "elapsed_s": float(time.perf_counter() - started)}


def select_l3_global(data, cfg, out):
    """Select L3 features and GBDT settings on all 518 outcomes, once."""
    path = out / "l3_global_selection.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        check(saved["input_features_sha"] == data["input_shas"]["features_w30.parquet"],
              "L3 selection no longer matches frozen features")
        return saved
    x = SimpleImputer(strategy="median").fit_transform(data["features"])
    y = data["y"]
    aucs = [float(roc_auc_score(y, x[:, j])) for j in range(x.shape[1])]
    order = sorted(range(len(aucs)), key=lambda j: (-aucs[j], j))
    chosen = order[:5]
    check(len(chosen) == 5, "L3 must select exactly five features")
    splits = make_folds(y, repeat=0, base_seed=int(cfg["selection_seed"]))
    best, best_auc = None, -np.inf
    for params in t03.grid("M1", t03_cfg(cfg)):
        values = []
        for fit, val in splits:
            estimator = t03.make_model("M1", params)
            estimator.fit(x[fit][:, chosen], y[fit])
            values.append(float(roc_auc_score(y[val], estimator.predict_proba(x[val][:, chosen])[:, 1])))
        score = float(np.mean(values))
        if score > best_auc:
            best, best_auc = dict(params), score
    chosen_names = [t03.FEATURES[j] for j in chosen]
    result = {"features": chosen_names, "feature_indices": chosen,
              "univariate_auc": {t03.FEATURES[j]: aucs[j] for j in range(len(aucs))},
              "params": best, "global_cv_auroc": best_auc,
              "input_features_sha": data["input_shas"]["features_w30.parquet"],
              "selection_seed": int(cfg["selection_seed"]),
              "intentional_leakage": "feature and parameter selection include outer-test labels"}
    save_json(path, result)
    return result


def contexts(cfg, data, segments, selected):
    """Construct deterministic experiment data; no test label enters a fit except L3."""
    ids = data["ids"]
    y = data["y"]
    sid, srec, sy = segments["row_id"], segments["record_id"], segments["y"].astype(int)
    seg_lookup = {rid: np.flatnonzero(srec == rid) for rid in ids}
    # Full-cohort imputation is the deliberate, explicitly prescribed L2 leak.
    imputed = SimpleImputer(strategy="median").fit_transform(data["features"])
    l2_m1_x, l2_m1_y, n_smote = smote_all(imputed, y, cfg["smote_k"], cfg["smote_seed"])
    m1_rows = np.asarray([f"real::{rid}" for rid in ids] +
                         [f"smote::{i:05d}" for i in range(n_smote)])
    m2_indices, l2_m2_y, n_duplicate = random_oversample(np.arange(len(y)), y, cfg["oversample_seed"])
    m2_rows = np.asarray([f"real::{rid}" for rid in ids] +
                         [f"duplicate::{i:05d}" for i in range(n_duplicate)])
    check(n_smote == n_duplicate == int((y == 0).sum() - (y == 1).sum()),
          "L2 oversampling failed to balance classes")
    aug = {
        "M1": {"ids": m1_rows, "y": l2_m1_y, "x": l2_m1_x,
               "tensor": None, "n_real": len(ids)},
        "M2": {"ids": m2_rows, "y": l2_m2_y, "x": None,
               "tensor": data["tensor"][m2_indices], "n_real": len(ids)},
    }
    for entry in aug.values():
        check(len(np.unique(entry["ids"])) == len(entry["ids"]), "Augmented row ID collision")
    y_lookup = dict(zip(ids, y))
    negative = dict(zip(ids, np.random.default_rng(42).permutation(y)))
    check(any(not grouped_segment_split(srec, train, test)
              for train, test in make_folds(sy, repeat=0, base_seed=cfg["segment_cv_seed_base"])),
          "L1 segment-level splitting did not produce the mandated record overlap")

    def get(protocol, model, seed, repeat, fold):
        if protocol in ("S_seg", "L1", "NC_L1"):
            if protocol == "S_seg":
                block = data["splits"].loc[(data["splits"].repeat == repeat) &
                                            (data["splits"].outer_fold == fold)]
                tr_map = dict(zip(block.loc[block.inner_fold >= 0, "record_id"],
                                  block.loc[block.inner_fold >= 0, "inner_fold"]))
                train_ids = list(tr_map)
                train_idx = np.concatenate([seg_lookup[r] for r in train_ids])
                test_ids = block.loc[block.inner_fold == -1, "record_id"].tolist()
                test_idx = np.concatenate([seg_lookup[r] for r in test_ids])
                inner = np.asarray([tr_map[r] for r in srec[train_idx]], dtype=int)
                check(grouped_segment_split(srec, train_idx, test_idx), "S_seg record leaked")
            else:
                train_idx, test_idx = make_folds(sy, repeat, cfg["segment_cv_seed_base"])[fold]
                inner_y = np.asarray([negative[r] for r in srec[train_idx]], dtype=int) if protocol == "NC_L1" else sy[train_idx]
                inside = StratifiedKFold(n_splits=5, shuffle=True,
                                         random_state=cfg["segment_cv_seed_base"] + 100 + 10 * repeat + fold)
                inner = np.empty(len(train_idx), dtype=int)
                for k, (_, v) in enumerate(inside.split(np.zeros(len(train_idx)), inner_y)):
                    inner[v] = k
            x = segments["features"] if model == "M1" else segments["tensor"]
            train_y = np.asarray([negative[r] for r in srec[train_idx]], dtype=int) if protocol == "NC_L1" else sy[train_idx]
            return {"row_id": sid[test_idx], "record_id": srec[test_idx], "y": sy[test_idx],
                    "x_train": x[train_idx], "y_train": train_y, "x_test": x[test_idx],
                    "inner": inner, "train_row_ids": sid[train_idx],
                    "train_record_ids": srec[train_idx] if protocol == "S_seg" else None,
                    "test_record_ids": srec[test_idx]}
        if protocol == "L2":
            obj = aug[model]
            if model == "M2":
                random_split = KFold(n_splits=5, shuffle=True,
                                     random_state=cfg["segment_cv_seed_base"] + repeat)
                train_idx, test_idx = list(random_split.split(obj["y"]))[fold]
            else:
                train_idx, test_idx = make_folds(obj["y"], repeat, cfg["segment_cv_seed_base"])[fold]
            inner_y = obj["y"][train_idx]
            inside = StratifiedKFold(n_splits=5, shuffle=True,
                                     random_state=cfg["segment_cv_seed_base"] + 100 + 10 * repeat + fold)
            inner = np.empty(len(train_idx), dtype=int)
            for k, (_, v) in enumerate(inside.split(np.zeros(len(train_idx)), inner_y)):
                inner[v] = k
            x = obj["x"] if model == "M1" else obj["tensor"]
            result_ids = obj["ids"][test_idx]
            record_ids = np.asarray([row[6:] if row.startswith("real::") else row
                                     for row in result_ids])
            return {"row_id": result_ids, "record_id": record_ids, "y": obj["y"][test_idx],
                    "x_train": x[train_idx], "y_train": obj["y"][train_idx],
                    "x_test": x[test_idx], "inner": inner,
                    "train_row_ids": obj["ids"][train_idx],
                    "train_record_ids": None, "test_record_ids": record_ids}
        if protocol == "L3":
            block = data["splits"].loc[(data["splits"].repeat == repeat) &
                                        (data["splits"].outer_fold == fold)]
            tr_ids = block.loc[block.inner_fold >= 0, "record_id"].to_numpy()
            te_ids = block.loc[block.inner_fold == -1, "record_id"].to_numpy()
            lookup = {rid: k for k, rid in enumerate(ids)}
            tr = np.asarray([lookup[r] for r in tr_ids])
            te = np.asarray([lookup[r] for r in te_ids])
            x = data["features"] if model == "M1" else data["tensor"]
            check(not set(tr_ids) & set(te_ids), "L3 train/test split is not record grouped")
            return {"row_id": te_ids, "record_id": te_ids, "y": y[te],
                    "x_train": x[tr], "y_train": y[tr], "x_test": x[te],
                    "inner": block.loc[block.inner_fold >= 0, "inner_fold"].to_numpy(dtype=int),
                    "train_row_ids": tr_ids, "train_record_ids": None, "test_record_ids": te_ids}
        raise ValueError(f"Unknown protocol {protocol}")

    return get, aug


def unit_path(out, key):
    protocol, model, seed, repeat, fold = key
    return out / "units" / protocol / f"{model}_s{seed}_r{repeat:02d}_f{fold}.npz"


def row_hash(values):
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def read_unit(path, expected, ctx):
    with np.load(path, allow_pickle=False) as saved:
        check(set(saved.files) == {"row_id", "record_id", "y", "p", "meta_json"},
              f"Corrupted resumed unit schema: {path}")
        row_ids = saved["row_id"].astype(str)
        record_ids = saved["record_id"].astype(str)
        yy = saved["y"].astype(int)
        pred = saved["p"].astype(float)
        info = json.loads(str(saved["meta_json"].item()))
    for item, value in expected.items():
        check(info.get(item) == value, f"Resumed unit provenance mismatch {item}: {path}")
    check(np.array_equal(row_ids, ctx["row_id"]) and
          np.array_equal(record_ids, ctx["record_id"]) and
          np.array_equal(yy, ctx["y"]) and pred.shape == (len(yy),) and
          np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all(),
          f"Resumed unit identity, label, or probability mismatch: {path}")
    return pred, info


def execute_unit(cfg, data, get_context, selected, out, key, identity):
    protocol, model, seed, repeat, fold = key
    ctx = get_context(protocol, model, seed, repeat, fold)
    path = unit_path(out, key)
    meta = {"key": list(key), "script_sha": identity["script_sha"],
            "config_sha": identity["config_sha"], "source_sha": identity["source_sha"],
            "test_rows_sha": row_hash(ctx["row_id"]), "train_rows_sha": row_hash(ctx["train_row_ids"]),
            "protocol_is_intentionally_leaky": protocol not in ("S_seg", "NC_L1")}
    if path.exists():
        pred, extra = read_unit(path, meta, ctx)
        return pred, extra, True
    started = time.perf_counter()
    if model == "M1":
        if protocol == "L3":
            use = selected["feature_indices"]
            estimator = t03.make_model("M1", selected["params"])
            estimator.fit(ctx["x_train"][:, use], ctx["y_train"])
            pred = estimator.predict_proba(ctx["x_test"][:, use])[:, 1]
            extra = {"global_selection": selected["features"], "global_params": selected["params"]}
        else:
            pred, params, cv_auc = t03.tune_and_predict(
                "M1", ctx["x_train"], ctx["y_train"], ctx["inner"],
                ctx["x_test"], t03_cfg(cfg), ctx["train_row_ids"], ctx["row_id"])
            extra = {"params": params, "inner_cv_auc": float(cv_auc)}
    else:
        pred, extra = train_m2(ctx["x_train"], ctx["y_train"], ctx["x_test"],
                              ctx["y"], seed, repeat, fold, cfg,
                              group_ids=ctx["train_record_ids"],
                              use_outer_test_for_checkpoint=protocol == "L3")
    pred = np.asarray(pred, dtype=np.float64)
    check(pred.shape == (len(ctx["y"]),) and np.isfinite(pred).all() and
          ((pred >= 0) & (pred <= 1)).all(), "Nonfinite or invalid model predictions")
    meta.update(extra)
    meta["elapsed_s"] = float(time.perf_counter() - started)
    atomic_new(path, lambda f: np.savez_compressed(
        f, row_id=ctx["row_id"].astype(str), record_id=ctx["record_id"].astype(str),
        y=ctx["y"].astype(np.int8), p=pred,
        meta_json=np.asarray(json.dumps(json_ready(meta), sort_keys=True))))
    read_unit(path, {k: meta[k] for k in ("key", "script_sha", "config_sha", "source_sha",
                                            "test_rows_sha", "train_rows_sha")}, ctx)
    return pred, meta, False


def all_units():
    for protocol in PROTOCOLS:
        for model in MODELS:
            seeds = SEEDS if model == "M2" else (0,)
            for repeat in REPEATS:
                for fold in FOLDS:
                    for seed in seeds:
                        yield (protocol, model, seed, repeat, fold)
    for fold in FOLDS:
        yield ("NC_L1", "M1", 0, 0, fold)


def frozen_identity(config_path, cfg, data):
    return {"prereg_sha": PREREG, "script_sha": sha(__file__),
            "config_sha": sha(config_path), "source_sha": data["input_shas"],
            "t04_frozen_sha_prefix": data["t04_lock"],
            "t03_features": list(t03.FEATURES)}


def average_oof(frame, model, seed=None):
    sub = frame.loc[frame.model.eq(model)]
    if seed is not None:
        sub = sub.loc[sub.seed.eq(seed)]
    check(len(sub) > 0, f"Missing OOF predictions for {model}")
    counts = sub.groupby("row_id", sort=True).size()
    expected = 3 * (3 if model == "M2" and seed is None else 1)
    check(counts.eq(expected).all(), f"Incomplete 3-repeat/3-seed OOF: {model}")
    result = sub.groupby(["row_id", "record_id", "y"], sort=True, as_index=False).p.mean()
    check(result.row_id.is_unique, "Duplicate row after repeated-probability averaging")
    return result


def frozen_s_scores(data, model, seed=None):
    frame = data["s_oof"].copy()
    if model == "M1":
        frame["p"] = frame.p_M1
    elif seed is not None:
        frame["p"] = frame[f"p_M2_s{seed}"]
    else:
        frame["p"] = frame[[f"p_M2_s{s}" for s in SEEDS]].mean(axis=1)
    result = frame.groupby("record_id", sort=True, as_index=False).agg(y=("y", "first"), p=("p", "mean"))
    check(len(result) == 518, "Frozen S OOF not 518 records")
    return result


def align_probabilities(left, right, on="row_id"):
    check(left[on].is_unique and right[on].is_unique, "Pairing keys must be unique")
    merged = left.merge(right[[on, "y", "p"]], on=on, how="inner", validate="one_to_one",
                        suffixes=("_l", "_s"))
    check(len(merged) == len(left) == len(right) and (merged.y_l == merged.y_s).all(),
          "Paired OOF coverage/labels differ")
    return (merged.y_l.to_numpy(dtype=int), merged.p_l.to_numpy(dtype=float),
            merged.p_s.to_numpy(dtype=float), merged)


def auc_fast(y, p):
    """Tie-aware Mann-Whitney AUROC, numerically matching sklearn AUROC."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    check(n0 > 0 and n1 > 0 and np.isfinite(p).all(), "AUROC requires two classes and finite scores")
    ranks = rankdata(p, method="average")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_auc_delta(y, p_leaky, p_strict, groups=None, B=10000, seed=42):
    """Paired index bootstrap; for L1, resample record clusters of segments."""
    y = np.asarray(y, dtype=int)
    a, b = np.asarray(p_leaky, dtype=float), np.asarray(p_strict, dtype=float)
    check(len(y) == len(a) == len(b) and np.unique(y).size == 2 and
          np.isfinite(a).all() and np.isfinite(b).all(), "Invalid paired AUROC input")
    point = auc_fast(y, a) - auc_fast(y, b)
    rng = np.random.default_rng(seed)
    if groups is not None:
        group_ids = np.asarray(groups, dtype=str)
        check(len(group_ids) == len(y), "Cluster IDs do not match segment scores")
        unique, inverse = np.unique(group_ids, return_inverse=True)
        indices = [np.flatnonzero(inverse == i) for i in range(len(unique))]
        check(len(indices) == 518, "L1 bootstrap must cluster on 518 original records")
        draw_n = len(unique)
    else:
        indices, draw_n = None, len(y)
    values = []
    for _ in range(B):
        draw = rng.integers(0, draw_n, size=draw_n)
        ix = np.concatenate([indices[i] for i in draw]) if indices is not None else draw
        yy = y[ix]
        if yy.min() == yy.max():
            continue
        auc_l = auc_fast(yy, a[ix])
        auc_s = auc_fast(yy, b[ix])
        values.append((auc_l - auc_s, auc_l, auc_s))
    check(len(values) > 0, "Bootstrap contained no valid two-class replicate")
    values = np.asarray(values, dtype=float)
    raw = min(1.0, max(1.0 / B, 2 * min(float(np.mean(values[:, 0] <= 0)),
                                              float(np.mean(values[:, 0] >= 0)))))
    ci = np.quantile(values, [0.025, 0.975], axis=0)
    return {"delta": float(point), "ci95": [float(ci[0, 0]), float(ci[1, 0])],
            "ci95_l": [float(ci[0, 1]), float(ci[1, 1])],
            "ci95_s": [float(ci[0, 2]), float(ci[1, 2])],
            "p_raw": float(raw), "bootstrap_valid": len(values)}


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    check(p.ndim == 1 and len(p) == 8 and np.isfinite(p).all() and
          ((p >= 0) & (p <= 1)).all(), "H2 Holm family requires 8 valid raw p-values")
    order = np.argsort(p, kind="stable")
    corrected = np.empty_like(p)
    maximum = 0.0
    for rank, ix in enumerate(order):
        maximum = max(maximum, min(1.0, float(p[ix]) * (len(p) - rank)))
        corrected[ix] = maximum
    return corrected.tolist()


def youden_cutoff(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    check(np.unique(y).size == 2 and np.isfinite(p).all(), "Youden needs both classes")
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def se_sp(y, p, threshold):
    y, pred = np.asarray(y, dtype=int), np.asarray(p, dtype=float) >= threshold
    check(np.unique(y).size == 2, "Se/Sp requires both outcome classes")
    return float(np.mean(pred[y == 1])), float(np.mean(~pred[y == 0]))


def bootstrap_sensitivity_delta(y, s, l, B=10000, seed=42):
    """Resample original records; each has mean S/L classification over 3 repeats."""
    y, s, l = np.asarray(y, dtype=int), np.asarray(s, dtype=float), np.asarray(l, dtype=float)
    check(len(y) == len(s) == len(l) == 518 and int(y.sum()) == 30,
          "L4 bootstrap must use the 518 original record clusters")
    point = float(l[y == 1].mean() - s[y == 1].mean())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        ix = rng.integers(0, len(y), len(y))
        positive = y[ix] == 1
        if not positive.any():
            continue
        draws.append(float(np.mean(l[ix][positive] - s[ix][positive])))
    check(len(draws) > 0, "L4 bootstrap has no positive records")
    draws = np.asarray(draws)
    raw = min(1.0, max(1.0 / B, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0))))
    return {"delta": point, "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "p_raw": float(raw), "bootstrap_valid": len(draws)}


def l4_oof(data, seed_override=None):
    """Youden S fitted on four other test folds; L4 fitted on the same test fold."""
    rows = []
    frame = data["s_oof"]
    for model in (("M2",) if seed_override is not None else MODELS):
        for repeat in REPEATS:
            sub = frame.loc[frame.repeat.eq(repeat)].copy()
            score_col = "p_M1" if model == "M1" else "p_M2"
            if model == "M2":
                sub[score_col] = (sub[[f"p_M2_s{s}" for s in SEEDS]].mean(axis=1)
                                  if seed_override is None else sub[f"p_M2_s{seed_override}"])
            for fold in FOLDS:
                test = sub.loc[sub.outer_fold.eq(fold)]
                other = sub.loc[~sub.outer_fold.eq(fold)]
                check(len(test) > 0 and len(other) > 0 and
                      np.unique(test.y).size == np.unique(other.y).size == 2,
                      "L4 fold lacks an outcome class")
                threshold_s = youden_cutoff(other.y, other[score_col])
                threshold_l = youden_cutoff(test.y, test[score_col])
                for rec in test.itertuples(index=False):
                    score = float(getattr(rec, score_col))
                    rows.append({"protocol": "L4", "model": model, "repeat": repeat,
                                 "outer_fold": fold, "record_id": rec.record_id,
                                 "y": int(rec.y), "p": score,
                                 "thr_S": threshold_s, "thr_L": threshold_l,
                                 "pred_S": int(score >= threshold_s),
                                 "pred_L": int(score >= threshold_l)})
    result = pd.DataFrame(rows)
    check(len(result) == 518 * 3 * (1 if seed_override is not None else 2) and
          result.groupby(["model", "repeat", "record_id"]).size().eq(1).all(),
          "L4 has an incomplete record-repeat assignment")
    return result


def h2_results(data, oofs, aug, cfg):
    rows = []
    for protocol in ("L1", "L2", "L3"):
        for model in MODELS:
            leaky = average_oof(oofs[protocol], model)
            if protocol == "L1":
                strict = average_oof(oofs["S_seg"], model)
                y, a, b, paired = align_probabilities(leaky, strict)
                score = bootstrap_auc_delta(y, a, b, paired.record_id.to_numpy(),
                                            cfg["bootstrap_resamples"], cfg["bootstrap_seed"])
                rec_l = leaky.groupby("record_id", as_index=False).agg(y=("y", "first"), p=("p", "mean"))
                rec_s = strict.groupby("record_id", as_index=False).agg(y=("y", "first"), p=("p", "mean"))
                rec_y, rec_a, rec_b, _ = align_probabilities(rec_l, rec_s, "record_id")
                rec_ci = bootstrap_auc_delta(rec_y, rec_a, rec_b, None,
                                              cfg["bootstrap_resamples"], cfg["bootstrap_seed"])
                descriptive = {"auroc_s_segment": auc_fast(y, b), "auroc_l_segment": auc_fast(y, a),
                               "ci95_s_segment": score["ci95_s"], "ci95_l_segment": score["ci95_l"],
                               "auroc_s_record": auc_fast(rec_y, rec_b),
                               "auroc_l_record": auc_fast(rec_y, rec_a),
                               "ci95_s_record": rec_ci["ci95_s"], "ci95_l_record": rec_ci["ci95_l"]}
            else:
                real = leaky if protocol == "L3" else leaky.loc[leaky.row_id.str.startswith("real::")].copy()
                real = real.assign(record_id=real.record_id.astype(str))
                check(len(real) == 518, f"{protocol} {model} original OOF records missing")
                strict = frozen_s_scores(data, model)
                rec_y, rec_a, rec_b, _ = align_probabilities(real, strict, "record_id")
                score = bootstrap_auc_delta(rec_y, rec_a, rec_b, None,
                                            cfg["bootstrap_resamples"], cfg["bootstrap_seed"])
                descriptive = {"auroc_s_record": auc_fast(rec_y, rec_b),
                               "auroc_l_record": auc_fast(rec_y, rec_a),
                               "ci95_s_record": score["ci95_s"], "ci95_l_record": score["ci95_l"]}
                if protocol == "L2":
                    check(len(leaky) == 2 * int((data["y"] == 0).sum()),
                          "L2 all-test balanced row count changed")
                    descriptive["auroc_l_all_test"] = auc_fast(leaky.y, leaky.p)
                    descriptive["delta_l_all_test_descriptive_only"] = (
                        descriptive["auroc_l_all_test"] - descriptive["auroc_s_record"])
                    descriptive["delta_l_real_paired"] = score["delta"]
            seed_deltas = "not_applicable"
            if model == "M2":
                values = []
                for seed in SEEDS:
                    seed_l = average_oof(oofs[protocol], model, seed=seed)
                    if protocol == "L1":
                        seed_s = average_oof(oofs["S_seg"], model, seed=seed)
                        yy, aa, bb, _ = align_probabilities(seed_l, seed_s)
                    else:
                        if protocol == "L2":
                            seed_l = seed_l.loc[seed_l.row_id.str.startswith("real::")]
                        seed_s = frozen_s_scores(data, model, seed=seed)
                        yy, aa, bb, _ = align_probabilities(seed_l, seed_s, "record_id")
                    values.append(float(auc_fast(yy, aa) - auc_fast(yy, bb)))
                seed_deltas = json.dumps(values)
            rows.append({"protocol": protocol, "model": model,
                         "contrast": "L-S", "metric": "delta_AUROC_segment_cluster" if protocol == "L1"
                         else "delta_AUROC_real_records_paired",
                         "delta": score["delta"], "ci95_low": score["ci95"][0],
                         "ci95_high": score["ci95"][1], "p_raw": score["p_raw"],
                         "bootstrap_valid": score["bootstrap_valid"],
                         "seed_deltas": seed_deltas, "details": json.dumps(descriptive, sort_keys=True)})
    l4 = l4_oof(data)
    for model in MODELS:
        subset = l4.loc[l4.model.eq(model)]
        agg = subset.groupby("record_id", as_index=False).agg(
            y=("y", "first"), s=("pred_S", "mean"), l=("pred_L", "mean"))
        score = bootstrap_sensitivity_delta(agg.y, agg.s, agg.l,
                                            cfg["bootstrap_resamples"], cfg["bootstrap_seed"])
        fold_stats = subset.groupby(["repeat", "outer_fold"]).apply(
            lambda f: pd.Series({"se_s": float(f.loc[f.y == 1, "pred_S"].mean()),
                                 "sp_s": float(1 - f.loc[f.y == 0, "pred_S"].mean()),
                                 "se_l": float(f.loc[f.y == 1, "pred_L"].mean()),
                                 "sp_l": float(1 - f.loc[f.y == 0, "pred_L"].mean())}),
            include_groups=False)
        details = {"mean_fold_Se_S": float(fold_stats.se_s.mean()),
                   "mean_fold_Sp_S": float(fold_stats.sp_s.mean()),
                   "mean_fold_Se_L": float(fold_stats.se_l.mean()),
                   "mean_fold_Sp_L": float(fold_stats.sp_l.mean()),
                   "record_mean_Se_S": float(agg.loc[agg.y == 1, "s"].mean()),
                   "record_mean_Se_L": float(agg.loc[agg.y == 1, "l"].mean())}
        seed_deltas = "not_applicable"
        if model == "M2":
            seed_se = []
            for seed in SEEDS:
                per_seed = l4_oof(data, seed_override=seed)
                seed_agg = per_seed.groupby("record_id", as_index=False).agg(
                    y=("y", "first"), s=("pred_S", "mean"), l=("pred_L", "mean"))
                seed_se.append(float(seed_agg.loc[seed_agg.y == 1, "l"].mean() -
                                     seed_agg.loc[seed_agg.y == 1, "s"].mean()))
            seed_deltas = json.dumps(seed_se)
        rows.append({"protocol": "L4", "model": model, "contrast": "L4-S",
                     "metric": "delta_Se_record_cluster", "delta": score["delta"],
                     "ci95_low": score["ci95"][0], "ci95_high": score["ci95"][1],
                     "p_raw": score["p_raw"], "bootstrap_valid": score["bootstrap_valid"],
                     "seed_deltas": seed_deltas, "details": json.dumps(details, sort_keys=True)})
    check(len(rows) == 8, "H2 family is not exactly 8 comparisons")
    for row, padj in zip(rows, holm([r["p_raw"] for r in rows])):
        row["p_holm"] = padj
    result = pd.DataFrame(rows)
    check(np.isfinite(result[["delta", "ci95_low", "ci95_high", "p_raw", "p_holm"]].to_numpy()).all(),
          "H2 output contains NaN or infinity")
    return result, l4


def negative_control(oof):
    control = oof.loc[oof.model.eq("M1")]
    check(len(control) > 0 and set(control.repeat) == {0}, "Negative control missing")
    check(control.row_id.is_unique and control.record_id.nunique() == 518,
          "Negative-control segment coverage changed")
    return auc_fast(control.y, control.p)


def store_frame(path, frame):
    """Finalize once; on restart assert equality instead of overwriting."""
    if path.exists():
        stored = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        pd.testing.assert_frame_equal(stored, frame, check_dtype=False, check_like=False)
    elif path.suffix == ".parquet":
        atomic_new(path, lambda f: frame.to_parquet(f, index=False))
    else:
        atomic_new(path, lambda f: f.write(frame.to_csv(index=False).encode("utf-8")))


def write_report(out, cfg, data, meta, h2, control_auc, pilot, runtime):
    def f(v):
        return f"{v:+.4f}"
    lines = ["=== REPORT P21-T05 ===",
             f"status: PASS; PC-A CPU; runtime_s={runtime:.2f}; GPU=unused",
             f"PREREG_v1.3_SHA256={PREREG}",
             "acceptance: pytest=PASS; frozen_T04=PASS; records=518; repeats=0,1,2; "
             "all_units=PASS; no_NaN=PASS; S_seg_no_overlap=PASS; L1_overlap=PASS",
             f"segments: total={int(meta.n_segments.sum())}; median_per_record={meta.n_segments.median():.1f}; "
             f"excluded_missing={int(meta.n_dropped_missing.sum())}; candidates={int(meta.n_segments_total.sum())}",
             f"pilot: M2_L1_one_unit_s={pilot['unit_s']:.2f}; "
             f"projected_M2_total_h={pilot['projected_m2_h']:.2f}; projected_only_not_budget_gate",
             "H2 CONFIRMATORY: 8 tests; paired record-cluster bootstrap 10000/seed42; Holm family=8",
             "protocol/model | AUROC_S | AUROC_L | delta[CI95] | p_raw,p_Holm"]
    for row in h2.itertuples(index=False):
        info = json.loads(row.details)
        if row.protocol in ("L1", "L2", "L3"):
            key_s = "auroc_s_segment" if row.protocol == "L1" else "auroc_s_record"
            key_l = "auroc_l_segment" if row.protocol == "L1" else "auroc_l_record"
            ci_key = "segment" if row.protocol == "L1" else "record"
            lines.append(f"{row.protocol}/{row.model}: "
                         f"{info[key_s]:.4f}[{f(info[f'ci95_s_{ci_key}'][0])},{f(info[f'ci95_s_{ci_key}'][1])}] | "
                         f"{info[key_l]:.4f}[{f(info[f'ci95_l_{ci_key}'][0])},{f(info[f'ci95_l_{ci_key}'][1])}] | "
                         f"{f(row.delta)}[{f(row.ci95_low)},{f(row.ci95_high)}] | "
                         f"{row.p_raw:.4f},{row.p_holm:.4f}")
            if row.protocol == "L1":
                lines.append(f" L1/record/{row.model}: "
                             f"AUROC_S={info['auroc_s_record']:.4f}[{f(info['ci95_s_record'][0])},{f(info['ci95_s_record'][1])}]; "
                             f"AUROC_L={info['auroc_l_record']:.4f}[{f(info['ci95_l_record'][0])},{f(info['ci95_l_record'][1])}] descriptive")
            if row.protocol == "L2":
                lines.append(f" L2/all_test/{row.model}: AUROC_L={info['auroc_l_all_test']:.4f}; "
                             f"delta_vs_S={f(info['delta_l_all_test_descriptive_only'])} "
                             "DESCRIPTIVE; H2 paired CI/p use real rows only")
        else:
            lines.append(f"L4/{row.model}: Se/Sp_S={info['mean_fold_Se_S']:.4f}/{info['mean_fold_Sp_S']:.4f}; "
                         f"Se/Sp_L={info['mean_fold_Se_L']:.4f}/{info['mean_fold_Sp_L']:.4f}; "
                         f"delta_Se_record={f(row.delta)}[{f(row.ci95_low)},{f(row.ci95_high)}]; "
                         f"p={row.p_raw:.4f}/{row.p_holm:.4f}")
    for row in h2.loc[h2.model == "M2"].itertuples(index=False):
        lines.append(f"M2_seed_deltas/{row.protocol}={row.seed_deltas}")
    lines.extend([f"negative_control: L1 M1 repeat0 record-permuted labels segment_AUROC={control_auc:.4f}; "
                  f"bounds=[{cfg['negative_control_bounds'][0]:.2f},{cfg['negative_control_bounds'][1]:.2f}] PASS",
                  f"SHA256_8: script={sha(__file__)[:8]}; segments_meta={sha(out/'segments_meta.parquet')[:8]}; "
                  f"h2_tests={sha(out/'h2_tests.csv')[:8]}; summary={sha(out/'summary.json')[:8]}",
                  "SHA256_8: " + "; ".join(f"{p.name}={sha(p)[:8]}" for p in
                                     [out / name for name in OUTPUT_OOF]),
                  "=== END ==="])
    check(len(lines) <= 30, f"Report must be <=30 lines, got {len(lines)}")
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    report_path = Path(cfg["project_dir"]) / "reports" / "P21-T05.md"
    if report_path.exists():
        check(report_path.read_text(encoding="utf-8") == text,
              "Existing report differs; refuse to overwrite")
    else:
        atomic_new(report_path, lambda f: f.write(text.encode("utf-8")))


def run_tests(project):
    import subprocess
    command = [sys.executable, "-m", "pytest", "-q", str(project / "tests" / "test_t05.py")]
    completed = subprocess.run(command, cwd=str(project.parent),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, check=False)
    print(completed.stdout[-3000:], flush=True)
    check(completed.returncode == 0, "T05 synthetic pytest did not pass")


def source_code_shas(project):
    names = ("src/t03_baselines.py", "src/t04_dl.py", "src/ctg_features.py",
             "../common/metrics.py", "../common/bootstrap.py", "../common/io_wfdb.py")
    return {name: sha((project / name).resolve()) for name in names}


def run(config_path, stage="run"):
    started = time.perf_counter()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config_check(cfg)
    project, out = Path(cfg["project_dir"]), Path(cfg["output_dir"])
    check(out.resolve() == (project / "outputs" / "t05_leakage").resolve(),
          "Output must remain inside P21_CTG_Acidemia/outputs/t05_leakage")
    data = load_inputs(cfg)
    print(f"FROZEN_T04_SHA_A_primary={data['t04_lock']} PASS; records={len(data['ids'])}", flush=True)
    identity = frozen_identity(config_path, cfg, data)
    identity["code_shas"] = source_code_shas(project)
    check(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CUDA must remain disabled")
    run_tests(project)
    if stage == "inspect":
        print(f"T05_INSPECT_PASS; expected_units={len(list(all_units()))}; no files written", flush=True)
        return 0
    check(stage == "run", "Use --stage inspect or --stage run")
    if out.exists():
        check(out.is_dir(), "T05 output path is not a directory")
        identity_file = out / "run_identity.json"
        check(identity_file.is_file(), "Existing T05 directory has no resumable provenance identity")
        existing = json.loads(identity_file.read_text(encoding="utf-8"))
        check(existing == identity, "Resumed T05 identity differs; STOP without overwriting")
        if (out / "summary.json").exists():
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            check(summary.get("status") == "PASS", "Final summary exists but is not PASS")
            for filename, digest in summary["output_sha256"].items():
                check(sha(out / filename) == digest, f"Completed output changed: {filename}")
            if not (project / "reports" / "P21-T05.md").exists():
                h2 = pd.read_csv(out / "h2_tests.csv")
                meta = pd.read_parquet(out / "segments_meta.parquet")
                write_report(out, cfg, data, meta, h2,
                             float(summary["negative_control_auroc"]), summary["pilot"],
                             float(summary["run_time_s"]))
            print("P21_T05_ALREADY_COMPLETE=PASS; DO_NOT_RERUN_OR_OVERWRITE", flush=True)
            return 0
    else:
        out.mkdir(parents=True, exist_ok=False)
        save_json(out / "run_identity.json", identity)

    segments, meta = build_segments(cfg, data, out)
    get_context, aug = contexts(cfg, data, segments, selected=None)
    keys = list(all_units())
    check(len(keys) == 245 and len(set(keys)) == 245, "Expected 245 unique training units")
    # Automatic timing pilot. Its predictions ARE unit 1; it is not retrained.
    pilot_key = ("L1", "M2", 0, 0, 0)
    _, pinfo, reused = execute_unit(cfg, data, get_context, None, out, pilot_key, identity)
    pilot_duration = float(pinfo["elapsed_s"])
    check(pilot_duration > 0, "Pilot did not record positive M2 time")
    pilot = {"unit": list(pilot_key), "unit_s": pilot_duration,
             "projected_m2_h": pilot_duration * 180 / 3600,
             "m2_units": 180, "pilot_reused": bool(reused)}
    pilot_path = out / "pilot_estimate.json"
    if pilot_path.exists():
        original = json.loads(pilot_path.read_text(encoding="utf-8"))
        check(original["unit_s"] == pilot_duration and original["m2_units"] == 180,
              "Pilot timing provenance changed")
        pilot = original
    else:
        save_json(pilot_path, pilot)
    print(f"T05_M2_L1_PILOT_UNIT={pilot_duration:.2f}s; "
          f"PROJECTED_M2_H={pilot['projected_m2_h']:.2f}; "
          "projection excludes M1, raw processing, inference; continuation automatic", flush=True)

    selected = select_l3_global(data, cfg, out)
    get_context, aug = contexts(cfg, data, segments, selected)
    # Cheap control first. A failed preregistered control must stop before the
    # remaining CPU-intensive M2 fits and must never be tuned post hoc.
    control_rows = []
    for key in [u for u in keys if u[0] == "NC_L1"]:
        pred, _, _ = execute_unit(cfg, data, get_context, selected, out, key, identity)
        ctx = get_context(*key)
        control_rows.extend({"model": "M1", "repeat": 0, "row_id": str(rid),
                             "record_id": str(rec), "y": int(yy), "p": float(pp)}
                            for rid, rec, yy, pp in zip(ctx["row_id"], ctx["record_id"],
                                                        ctx["y"], pred))
    early_negative_auc = negative_control(pd.DataFrame(control_rows))
    print(f"T05_EARLY_NEGATIVE_CONTROL={early_negative_auc:.4f}; "
          f"in_bounds={cfg['negative_control_bounds'][0] <= early_negative_auc <= cfg['negative_control_bounds'][1]}",
          flush=True)
    check(cfg["negative_control_bounds"][0] <= early_negative_auc <= cfg["negative_control_bounds"][1],
          "Preregistered negative control outside [0.40,0.60]; stop without changing settings")
    total_skipped = 0
    for ix, key in enumerate(keys, 1):
        _, _, skipped = execute_unit(cfg, data, get_context, selected, out, key, identity)
        total_skipped += int(skipped)
        if ix % 10 == 0 or ix == len(keys):
            print(f"T05_UNITS={ix}/{len(keys)}; reused={total_skipped}", flush=True)

    oofs = {}
    for protocol in (*PROTOCOLS, "NC_L1"):
        rows = []
        for key in [u for u in keys if u[0] == protocol]:
            ctx = get_context(*key)
            core = {"key": list(key), "script_sha": identity["script_sha"],
                    "config_sha": identity["config_sha"], "source_sha": identity["source_sha"],
                    "test_rows_sha": row_hash(ctx["row_id"]),
                    "train_rows_sha": row_hash(ctx["train_row_ids"])}
            pred, _ = read_unit(unit_path(out, key), core, ctx)
            protocol_name, model, seed, repeat, fold = key
            rows.extend({"protocol": protocol_name, "model": model, "seed": seed,
                         "repeat": repeat, "outer_fold": fold, "row_id": str(row_id),
                         "record_id": str(rid), "y": int(y), "p": float(p)}
                        for row_id, rid, y, p in zip(ctx["row_id"], ctx["record_id"], ctx["y"], pred))
        frame = pd.DataFrame(rows).sort_values(
            ["model", "seed", "repeat", "row_id"]).reset_index(drop=True)
        check(not frame.duplicated(["model", "seed", "repeat", "row_id"]).any(),
              f"Duplicate OOF record in {protocol}")
        check(np.isfinite(frame.p).all() and frame.p.between(0, 1).all(),
              f"Invalid {protocol} OOF probabilities")
        oofs[protocol] = frame

    control_auc = negative_control(oofs["NC_L1"])
    print(f"T05_NEGATIVE_CONTROL_L1_M1_R0_AUROC={control_auc:.4f}", flush=True)
    check(cfg["negative_control_bounds"][0] <= control_auc <= cfg["negative_control_bounds"][1],
          f"Negative control failed preregistered [0.4,0.6] bounds: {control_auc:.4f}")
    h2, l4 = h2_results(data, oofs, aug, cfg)
    for protocol in PROTOCOLS:
        store_frame(out / f"oof_{protocol}.parquet", oofs[protocol])
    store_frame(out / "oof_L4.parquet", l4)
    store_frame(out / "h2_tests.csv", h2)
    duration = time.perf_counter() - started
    files = ["segments_meta.parquet", "segments_cache.npz", "l3_global_selection.json",
             "run_identity.json", "pilot_estimate.json", "h2_tests.csv", *OUTPUT_OOF]
    output_sha = {file: sha(out / file) for file in files}
    summary = {"status": "PASS", "task": "P21-T05", "prereg_sha256": PREREG,
               "run_time_s": duration, "frozen_sha_verified": True,
               "n_records": 518, "n_segments": int(meta.n_segments.sum()),
               "n_segments_excluded": int(meta.n_dropped_missing.sum()),
               "units": len(keys), "units_skipped_this_session": total_skipped,
               "negative_control_auroc": control_auc, "negative_control_pass": True,
               "h2_family": 8, "h2_results": h2.to_dict(orient="records"),
               "pilot": pilot, "output_sha256": output_sha}
    save_json(out / "summary.json", summary)
    write_report(out, cfg, data, meta, h2, control_auc, pilot, duration)
    print("P21_T05_EXECUTION_COMPLETE; STATUS=PASS", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P21-T05 resumable leakage audit, CPU only")
    parser.add_argument("--config", type=Path,
                        default=Path("P21_CTG_Acidemia/configs/t05.yaml"))
    parser.add_argument("--stage", choices=("inspect", "run"), default="run")
    args = parser.parse_args()
    try:
        exit_code = run(args.config, args.stage)
    except (ValueError, AssertionError, FileNotFoundError, FileExistsError,
            RuntimeError, OSError) as exc:
        print(f"P21_T05_STOP: {type(exc).__name__}: {exc}", flush=True)
        exit_code = 2
    sys.exit(exit_code)




"""P21-T04: CPU-only, fixed-design, resumable CTG deep-learning evaluation.

Only outer-training records enter the training/validation split. Outputs are
committed per (configuration, model, seed, repeat, outer fold). Finished unit
files are verified and never overwritten, including after an interrupted run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from common.bootstrap import discrimination_ci
from common.metrics import all_metrics, discrimination

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = TensorDataset = None


TASK = "P21-T04"
PREREG_SHA = "ebb799092e350b332849c07d277d9198ef3e6ef52a4b31df61958f6d1382558e"
T03_METRICS_SHA = "8d380a01c0cb859abfe2b17ed34b1b9720e1a4070651975cbf01dcf0faa8561e"
T03_BOOTSTRAP_SHA = "54136f611c4e0a6cac3fd8e44a075399d23636d5527954da7d2572dead6d2096"
CONFIGS = {
    "A_primary": {"window": "W30", "split": "split.csv", "target": "y705"},
    "B_noexcl": {"window": "W30", "split": "split_noexcl.csv", "target": "y705"},
    "C_w60": {"window": "W60", "split": "split.csv", "target": "y705"},
    "D_y715": {"window": "W30", "split": "split.csv", "target": "y715"},
    "E_y720": {"window": "W30", "split": "split.csv", "target": "y720"},
}
MODELS = ("M2", "M3")
SEEDS = (0, 1, 2)
LIMIT_SECONDS = 48 * 3600
SUMMARY_FILES = ("oof.parquet", "train_log.csv", "metrics.json")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def sha8(path):
    return sha256(path)[:8]


def _json_safe(item):
    if isinstance(item, dict):
        return {str(k): _json_safe(v) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        return [_json_safe(v) for v in item]
    if isinstance(item, (np.bool_, bool)):
        return bool(item)
    if isinstance(item, (np.integer, int)):
        return int(item)
    if isinstance(item, (np.floating, float)):
        return float(item) if np.isfinite(item) else None
    return item


def atomic_new(path, writer):
    """Publish a complete file once; never replace a pre-existing target."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="t04_pending_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"Target appeared during write: {path}")
        os.rename(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_json(path, obj):
    payload = json.dumps(_json_safe(obj), sort_keys=True, indent=2,
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    atomic_new(path, lambda f: f.write(payload))


def locked_config(cfg):
    expected = {
        "task", "run_name", "prereg_version", "prereg_sha256", "input_dir", "output_root",
        "configs", "models", "seeds", "cv_repeats", "cv_outer_folds", "reduced_repeats",
        "validation_fraction", "learning_rate", "weight_decay", "batch_size",
        "max_epochs", "patience", "torch_threads", "budget_hours",
        "negative_control_seed", "negative_control_bounds", "bootstrap_resamples",
        "bootstrap_seed", "specificity",
    }
    checks = [
        set(cfg) == expected,
        cfg["task"] == TASK,
        cfg["run_name"] == "t04_dl",
        cfg["prereg_version"] == "1.2",
        cfg["prereg_sha256"] == PREREG_SHA,
        cfg["input_dir"] == "D:/Tuan/AI_training/data_proc/ctu-uhb-ctgdb/1.0.0",
        cfg["output_root"] == "D:/Tuan/AI_training/P21_CTG_Acidemia/outputs",
        cfg["configs"] == CONFIGS,
        cfg["models"] == list(MODELS),
        cfg["seeds"] == list(SEEDS),
        [cfg["cv_repeats"], cfg["cv_outer_folds"], cfg["reduced_repeats"]] == [10, 5, 3],
        float(cfg["validation_fraction"]) == 0.2,
        float(cfg["learning_rate"]) == 0.001,
        float(cfg["weight_decay"]) == 0.0001,
        int(cfg["batch_size"]) == 32,
        int(cfg["max_epochs"]) == 60,
        int(cfg["patience"]) == 10,
        int(cfg["torch_threads"]) == 4,
        int(cfg["budget_hours"]) == 48,
        int(cfg["negative_control_seed"]) == 42,
        list(map(float, cfg["negative_control_bounds"])) == [0.4, 0.6],
        int(cfg["bootstrap_resamples"]) == 10000,
        int(cfg["bootstrap_seed"]) == 42,
        float(cfg["specificity"]) == 0.85,
    ]
    if not all(checks):
        raise ValueError("YAML differs from the preregistered P21-T04 specification")


def require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is missing; install the CPU-only torch wheel first")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != 4:
        raise AssertionError("torch CPU thread count is not 4")


def fixed_normalize(x, with_uc=False):
    """No fitted means, standard deviations, or subject-specific parameters."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected a finite (N,3,T) T02 array")
    if not np.isin(x[:, 1], [0.0, 1.0]).all():
        raise ValueError("Missing-data mask must remain exactly binary")
    output = np.empty((len(x), 3 if with_uc else 2, x.shape[-1]), dtype=np.float32)
    output[:, 0] = (x[:, 0] - 140.0) / 20.0
    output[:, 1] = x[:, 1]
    if with_uc:
        output[:, 2] = x[:, 2] / 100.0
    return output


def budget_estimate(t_m2, t_m3, hours=48):
    """Seconds, including UC, negative control, and one M2 reproducibility fit.

    One W60 training is estimated as twice a W30 training. UC uses M2's W30
    training time as its fixed pilot proxy; no post-hoc model-time adjustment.
    """
    if not (np.isfinite(t_m2) and np.isfinite(t_m3) and t_m2 > 0 and t_m3 > 0):
        raise ValueError("Positive finite pilot durations are required")
    bonus_m2 = (10 * 5 + 5 + 1) * t_m2
    full = 900 * (t_m2 + t_m3) + bonus_m2
    reduced = 480 * (t_m2 + t_m3) + bonus_m2
    mandatory = 300 * (t_m2 + t_m3) + bonus_m2
    limit = float(hours) * 3600
    if full <= limit:
        rule, counts = "FULL_10_ALL", {name: 10 for name in CONFIGS}
    elif mandatory > limit:
        rule, counts = "STOP_A_B_OVER_48H", None
    else:
        rule = "REDUCED_A_B_10_C_D_E_3"
        counts = {name: (10 if name in ("A_primary", "B_noexcl") else 3)
                  for name in CONFIGS}
    return {
        "rule": rule, "repeats": counts, "estimate_full_s": float(full),
        "estimate_reduced_s": float(reduced), "estimate_A_B_mandatory_s": float(mandatory),
        "budget_s": limit, "assumption_w60_factor": 2,
        "assumption_uc_and_control_w30_M2_time_s": float(t_m2),
        "reproducibility_extra_M2_units": 1,
    }


def check_split(frame, expected_ids):
    if list(frame.columns) != ["record_id", "repeat", "outer_fold", "inner_fold"]:
        raise ValueError("Unexpected T02 split columns")
    expected = set(expected_ids)
    if set(frame.repeat) != set(range(10)) or set(frame.outer_fold) != set(range(5)):
        raise AssertionError("Expected 10 repeats and five outer folds")
    if frame.duplicated(["record_id", "repeat", "outer_fold"]).any():
        raise AssertionError("Duplicated split row")
    for rep in range(10):
        seen_test = []
        for fold in range(5):
            block = frame.loc[(frame.repeat == rep) & (frame.outer_fold == fold)]
            if set(block.record_id) != expected or len(block) != len(expected):
                raise AssertionError("Outer fold does not partition expected records")
            if not set(block.inner_fold).issubset({-1, 0, 1, 2, 3, 4}):
                raise AssertionError("Unknown inner fold index")
            train = block.loc[block.inner_fold >= 0, "record_id"]
            test = block.loc[block.inner_fold == -1, "record_id"]
            if len(train) == 0 or len(test) == 0 or (set(train) & set(test)):
                raise AssertionError("Outer train/test overlap or empty partition")
            seen_test.extend(test.tolist())
        if len(seen_test) != len(expected) or set(seen_test) != expected:
            raise AssertionError("Record missing or repeated as outer-test")


def load_data(cfg):
    root = Path(cfg["input_dir"])
    names = ("labels.parquet", "split.csv", "split_noexcl.csv",
             "dl_w30_1hz.npz", "dl_w60_1hz.npz")
    if any(not (root / n).is_file() for n in names):
        raise FileNotFoundError("One or more frozen T02 input files are missing")
    labels = pd.read_parquet(root / "labels.parquet")
    if len(labels) != 552 or labels.record_id.astype(str).duplicated().any():
        raise ValueError("T02 labels must contain 552 unique records")
    labels["record_id"] = labels.record_id.astype(str)
    if not all(labels[c].isin([0, 1]).all() for c in ("y705", "y715", "y720")):
        raise ValueError("Outcome labels must be binary")
    if labels.excluded.isna().any() or int(labels.excluded.sum()) != 34:
        raise ValueError("T02 QC exclusion identity changed")
    labels = labels.set_index("record_id")
    ids = labels.index.to_numpy(dtype=str)
    tensors = {}
    for win, filename, length in (("W30", "dl_w30_1hz.npz", 1800),
                                   ("W60", "dl_w60_1hz.npz", 3600)):
        with np.load(root / filename, allow_pickle=False) as data:
            if set(data.files) != {"X", "record_id"}:
                raise ValueError("T02 NPZ keys changed")
            arr = data["X"]
            source_ids = data["record_id"].astype(str)
        if arr.shape != (552, 3, length):
            raise ValueError("T02 DL tensor shape changed")
        if len(set(source_ids)) != 552 or set(source_ids) != set(ids):
            raise AssertionError("NPZ-to-label record identity mismatch")
        lookup = {rid: j for j, rid in enumerate(source_ids)}
        tensors[win] = arr[[lookup[rid] for rid in ids]].astype(np.float32)
        fixed_normalize(tensors[win])  # preflight includes masks and finiteness
    splits = {}
    for filename in ("split.csv", "split_noexcl.csv"):
        frame = pd.read_csv(root / filename, dtype={"record_id": str})
        expected = ids if filename == "split_noexcl.csv" else ids[~labels.excluded.to_numpy(dtype=bool)]
        check_split(frame, expected)
        splits[filename] = frame
    return labels, tensors, splits, {name: sha256(root / name) for name in names}


def fold_ids(splits, spec, repeat, fold):
    split = splits[spec["split"]]
    block = split.loc[(split.repeat == repeat) & (split.outer_fold == fold)]
    train_ids = block.loc[block.inner_fold >= 0, "record_id"].to_numpy(dtype=str)
    test_ids = block.loc[block.inner_fold == -1, "record_id"].to_numpy(dtype=str)
    if not len(train_ids) or not len(test_ids) or set(train_ids) & set(test_ids):
        raise AssertionError("Outer-test data entered outer-train")
    return train_ids, test_ids


def train_val_indices(train_ids, train_y, test_ids, repeat, fold, seed):
    if set(train_ids) & set(test_ids):
        raise AssertionError("Outer-test ID entered the training or validation pool")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=1000 * repeat + 10 * fold + seed
    )
    fit_idx, val_idx = next(splitter.split(np.zeros(len(train_y)), train_y))
    fit_ids, val_ids = set(train_ids[fit_idx]), set(train_ids[val_idx])
    if fit_ids & val_ids or fit_ids & set(test_ids) or val_ids & set(test_ids):
        raise AssertionError("Train/validation/outer-test leakage")
    if len(np.unique(train_y[fit_idx])) != 2 or len(np.unique(train_y[val_idx])) != 2:
        raise ValueError("A split has only one class")
    return fit_idx, val_idx


if nn is not None:
    class M2CNN(nn.Module):
        def __init__(self, in_channels=2):
            super().__init__()
            layers = []
            previous = in_channels
            for channels in (16, 32, 64, 64):
                layers.extend([
                    nn.Conv1d(previous, channels, kernel_size=7, padding="same"),
                    nn.BatchNorm1d(channels), nn.ReLU(), nn.MaxPool1d(2),
                ])
                previous = channels
            self.features = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(64, 1)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).squeeze(-1)
            return self.classifier(self.dropout(x)).squeeze(-1)


    class ResidualBlock(nn.Module):
        """Wang-style post-activation residual block (8/5/3 kernels)."""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, 8, padding="same")
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, 5, padding="same")
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.conv3 = nn.Conv1d(out_channels, out_channels, 3, padding="same")
            self.bn3 = nn.BatchNorm1d(out_channels)
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
            self.shortcut_bn = nn.BatchNorm1d(out_channels)
            self.activation = nn.ReLU()

        def forward(self, x):
            z = self.activation(self.bn1(self.conv1(x)))
            z = self.activation(self.bn2(self.conv2(z)))
            z = self.bn3(self.conv3(z))
            return self.activation(z + self.shortcut_bn(self.shortcut(x)))


    class M3ResNet(nn.Module):
        def __init__(self, in_channels=2):
            super().__init__()
            self.blocks = nn.Sequential(
                ResidualBlock(in_channels, 32),
                ResidualBlock(32, 64),
                ResidualBlock(64, 64),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(64, 1)

        def forward(self, x):
            return self.classifier(self.pool(self.blocks(x)).squeeze(-1)).squeeze(-1)


def make_model(name, in_channels=2):
    require_torch()
    if name in ("M2", "M2UC", "NC_M2"):
        return M2CNN(in_channels)
    if name == "M3":
        if in_channels != 2:
            raise ValueError("M3 accepts only two channels")
        return M3ResNet(in_channels)
    raise ValueError("Unregistered architecture")


def seed_everything(seed):
    require_torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def predict_prob(model, x, batch_size=32):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            tensor = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size]))
            outputs.append(torch.sigmoid(model(tensor)).cpu().numpy().reshape(-1))
    return np.concatenate(outputs).astype(np.float64)


def fit_predict(model_name, seed, x_train, y_train, train_ids, test_ids,
                x_test, repeat, fold, cfg):
    """Train on 80% of outer-train; validate on remaining 20%; never use test y."""
    started = time.perf_counter()
    seed_everything(seed)
    fit_idx, val_idx = train_val_indices(
        train_ids, y_train, test_ids, repeat, fold, seed
    )
    x_fit = np.ascontiguousarray(x_train[fit_idx])
    x_val = np.ascontiguousarray(x_train[val_idx])
    y_fit = y_train[fit_idx].astype(np.float32)
    y_val = y_train[val_idx].astype(np.int64)
    positives = int(y_fit.sum())
    negatives = int(len(y_fit) - positives)
    if positives < 1 or negatives < 1:
        raise AssertionError("BCEWithLogits pos_weight needs both training classes")
    model = make_model(model_name, x_train.shape[1]).cpu()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(seed))
    dataset = TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit))
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
                        num_workers=0, generator=generator)
    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    wait = 0
    losses = []
    stopped_epoch = 0
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y_batch)
        epoch_loss = total_loss / len(y_fit)
        losses.append(epoch_loss)
        probabilities = predict_prob(model, x_val, cfg["batch_size"])
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Nonfinite validation prediction")
        validation_auc = float(roc_auc_score(y_val, probabilities))
        if validation_auc > best_auc:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        stopped_epoch = epoch
        if wait >= int(cfg["patience"]):
            break
    if best_state is None:
        raise AssertionError("No best-validation checkpoint was retained")
    model.load_state_dict(best_state)
    predictions = None if x_test is None else predict_prob(model, x_test, cfg["batch_size"])
    if predictions is not None and (
        not np.isfinite(predictions).all()
        or np.any((predictions < 0) | (predictions > 1))
    ):
        raise FloatingPointError("Invalid outer-test probability")
    return predictions, {
        "epoch_stop": stopped_epoch, "best_epoch": best_epoch,
        "best_validation_auroc": best_auc,
        "train_loss_initial": losses[0], "train_loss_final": losses[-1],
        "train_loss_decreased": bool(losses[-1] < losses[0]),
        "runtime_s": time.perf_counter() - started,
        "n_train": len(fit_idx), "n_validation": len(val_idx),
        "n_positive_train": positives, "n_negative_train": negatives,
    }


def _unit_filename(model_name, seed, repeat, fold):
    return f"{model_name}_s{seed}_r{repeat:02d}_f{fold}.npz"


def _unit_path(out_root, config_name, model_name, seed, repeat, fold):
    return (Path(out_root) / config_name / "units" /
            _unit_filename(model_name, seed, repeat, fold))


def _fold_tensor(data, labels, tensors, splits, spec, repeat, fold, use_uc=False):
    train_ids, test_ids = fold_ids(splits, spec, repeat, fold)
    all_ids = labels.index.to_numpy(dtype=str)
    lookup = {rid: ix for ix, rid in enumerate(all_ids)}
    indices_train = [lookup[rid] for rid in train_ids]
    indices_test = [lookup[rid] for rid in test_ids]
    x = tensors[(spec["window"], use_uc)]
    y_train = labels.loc[train_ids, spec["target"]].to_numpy(dtype=np.int64)
    y_test = labels.loc[test_ids, spec["target"]].to_numpy(dtype=np.int64)
    return (train_ids, test_ids, x[indices_train], x[indices_test], y_train, y_test)


def _control_permutation(labels, splits, seed=42, fold=0):
    """Match T03's one RNG stream across outer-test folds 0..4."""
    rng = np.random.default_rng(int(seed))
    spec = CONFIGS["A_primary"]
    permuted = None
    for current in range(fold + 1):
        train_ids, _ = fold_ids(splits, spec, repeat=0, fold=current)
        y_train = labels.loc[train_ids, "y705"].to_numpy(dtype=np.int64)
        permuted = rng.permutation(y_train)
    return permuted


def expected_units(repeat_counts):
    """Order is fixed so an interrupted run resumes without changing priorities."""
    for config_name in CONFIGS:
        for repeat in range(repeat_counts[config_name]):
            for fold in range(5):
                for model_name in MODELS:
                    for seed in SEEDS:
                        yield config_name, model_name, seed, repeat, fold
                if config_name == "A_primary":
                    yield config_name, "M2UC", 0, repeat, fold
                    if repeat == 0:
                        yield config_name, "NC_M2", 0, repeat, fold


def _unit_meta(config_name, model_name, seed, repeat, fold, train_info,
               script_hash, config_hash, test_ids):
    return {
        "task": TASK, "prereg_sha256": PREREG_SHA,
        "script_sha256": script_hash, "config_sha256": config_hash,
        "unit": f"{config_name}/{_unit_filename(model_name, seed, repeat, fold)}",
        "config": config_name, "model": model_name, "seed": seed,
        "repeat": repeat, "outer_fold": fold,
        "test_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
        "no_outer_test_leakage": True,
        **train_info,
    }


def read_unit(path, expected_meta, test_ids, y_test):
    """Every resumed unit is validated against the frozen split and source."""
    with np.load(path, allow_pickle=False) as unit:
        if set(unit.files) != {"record_id", "y", "p", "meta_json"}:
            raise ValueError(f"Corrupt unit schema: {path.name}")
        unit_ids = unit["record_id"].astype(str)
        unit_y = unit["y"].astype(np.int64)
        probabilities = unit["p"].astype(np.float64)
        meta = json.loads(str(unit["meta_json"].item()))
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise AssertionError(f"Resumed unit metadata mismatch ({key}): {path.name}")
    expected_hash = hashlib.sha256("\n".join(test_ids).encode()).hexdigest()
    if meta.get("test_ids_sha256") != expected_hash or not meta.get("no_outer_test_leakage"):
        raise AssertionError("Unit has no verified train/test boundary")
    if (not np.array_equal(unit_ids, test_ids) or
        not np.array_equal(unit_y, y_test) or
        probabilities.shape != (len(test_ids),) or
        not np.isfinite(probabilities).all() or
        np.any((probabilities < 0) | (probabilities > 1))):
        raise AssertionError("Unit predictions do not match the locked outer-test")
    return probabilities, meta


def _unit_data(labels, tensors, splits, config_name, model_name, seed, repeat, fold,
               cfg):
    spec = CONFIGS[config_name]
    if model_name == "M2UC" and (config_name != "A_primary" or seed != 0):
        raise ValueError("M2UC is only preregistered for A_primary seed 0")
    if model_name == "NC_M2" and (config_name != "A_primary" or repeat != 0 or seed != 0):
        raise ValueError("Negative control is only A_primary repeat 0 seed 0")
    use_uc = model_name == "M2UC"
    train_ids, test_ids, x_train, x_test, y_train, y_test = _fold_tensor(
        data=None, labels=labels, tensors=tensors, splits=splits,
        spec=spec, repeat=repeat, fold=fold, use_uc=use_uc,
    )
    if model_name == "NC_M2":
        y_train = _control_permutation(labels, splits,
                                       seed=cfg["negative_control_seed"], fold=fold)
    return train_ids, test_ids, x_train, x_test, y_train, y_test


def run_one_unit(cfg, out_root, labels, tensors, splits, unit_key,
                 script_hash, config_hash):
    config_name, model_name, seed, repeat, fold = unit_key
    path = _unit_path(out_root, *unit_key)
    train_ids, test_ids, x_train, x_test, y_train, y_test = _unit_data(
        labels, tensors, splits, config_name, model_name, seed, repeat, fold, cfg
    )
    metadata = {
        "task": TASK, "prereg_sha256": PREREG_SHA,
        "script_sha256": script_hash, "config_sha256": config_hash,
        "config": config_name, "model": model_name,
        "seed": seed, "repeat": repeat, "outer_fold": fold,
    }
    if path.exists():
        probabilities, info = read_unit(path, metadata, test_ids, y_test)
        return probabilities, info, True

    probabilities, training = fit_predict(
        model_name, seed, x_train, y_train, train_ids, test_ids, x_test,
        repeat, fold, cfg,
    )
    info = _unit_meta(config_name, model_name, seed, repeat, fold,
                      training, script_hash, config_hash, test_ids)
    atomic_new(path, lambda f: np.savez_compressed(
        f, record_id=test_ids.astype(str), y=y_test.astype(np.int8),
        p=probabilities.astype(np.float64),
        meta_json=np.asarray(json.dumps(_json_safe(info), sort_keys=True)),
    ))
    read_unit(path, metadata, test_ids, y_test)
    return probabilities, info, False


def _pytest_t04():
    project = Path(__file__).resolve().parents[1]
    finished = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(project / "tests" / "test_t04.py")],
        cwd=str(project.parent), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    )
    if finished.returncode:
        print(finished.stdout[-3000:], flush=True)
        raise RuntimeError("P21-T04 synthetic pytest failed")
    return True


def _source_identity(config_path):
    identity = {
        "script_sha256": sha256(__file__),
        "config_sha256": sha256(config_path),
        "common_metrics_sha256": sha256(Path(__file__).resolve().parents[2] / "common" / "metrics.py"),
        "common_bootstrap_sha256": sha256(Path(__file__).resolve().parents[2] / "common" / "bootstrap.py"),
    }
    if (identity["common_metrics_sha256"] != T03_METRICS_SHA or
        identity["common_bootstrap_sha256"] != T03_BOOTSTRAP_SHA):
        raise AssertionError("Frozen T03 common metrics/bootstrap hashes differ; do not edit them")
    return identity


def _pilot(cfg, config_path):
    started = time.perf_counter()
    require_torch()
    _pytest_t04()
    source = _source_identity(config_path)
    out_root = Path(cfg["output_root"]) / cfg["run_name"]
    if (out_root / "pilot.json").exists() or (out_root / "run_lock.json").exists():
        raise FileExistsError("Pilot or full-run identity already exists; never overwrite")
    labels, arrays, splits, input_hashes = load_data(cfg)
    tensors = {(win, uc): fixed_normalize(arr, with_uc=uc)
               for win, arr in arrays.items() for uc in (False, True)}
    spec = CONFIGS["A_primary"]
    train_ids, test_ids, x_train, _, y_train, _ = _fold_tensor(
        None, labels, tensors, splits, spec, 0, 0, False
    )
    timing = {}
    print("=== P21-T04 PILOT START ===", flush=True)
    for model_name in MODELS:
        _, info = fit_predict(
            model_name, 0, x_train, y_train, train_ids, test_ids,
            None, 0, 0, cfg,
        )
        timing[model_name] = {
            "train_runtime_s": info["runtime_s"],
            "epoch_stop": info["epoch_stop"],
            "train_loss_initial": info["train_loss_initial"],
            "train_loss_final": info["train_loss_final"],
            "train_loss_decreased": info["train_loss_decreased"],
        }
    plan = budget_estimate(timing["M2"]["train_runtime_s"],
                           timing["M3"]["train_runtime_s"], cfg["budget_hours"])
    payload = {
        "task": TASK, "prereg_version": "1.2", "prereg_sha256": PREREG_SHA,
        "torch_version": torch.__version__, "python": sys.version.split()[0],
        "device": "cpu", "torch_threads": 4,
        "source": source, "input_sha256": input_hashes,
        "pilot": timing, "budget": plan,
    }
    atomic_json(out_root / "pilot.json", payload)
    print("=== REPORT P21-T04-PILOT ===")
    print("status: PASS; mode=pilot; PC-A CPU; torch=" + torch.__version__ + "; python=" + sys.version.split()[0])
    print("prereg_v1.2_sha256: " + PREREG_SHA)
    for name, value in timing.items():
        print(f"{name}: train_s={value['train_runtime_s']:.2f}; epoch_stop={value['epoch_stop']}; "
              f"loss_initial={value['train_loss_initial']:.6f}; "
              f"loss_final={value['train_loss_final']:.6f}; "
              f"loss_decreased={value['train_loss_decreased']}")
    print(f"W60_time_estimate: M2={2*timing['M2']['train_runtime_s']:.2f}s; "
          f"M3={2*timing['M3']['train_runtime_s']:.2f}s")
    print(f"estimate_full_h={plan['estimate_full_s']/3600:.2f}; "
          f"estimate_reduced_h={plan['estimate_reduced_s']/3600:.2f}; "
          f"estimate_A_B_h={plan['estimate_A_B_mandatory_s']/3600:.2f}")
    print("budget_rule: " + plan["rule"])
    print(f"runtime_s: {time.perf_counter()-started:.2f}")
    print("pilot_file_sha256_8: " + sha8(out_root / "pilot.json"))
    print("pilot_AUROC: NOT REPORTED")
    print("=== END ===")
    return 0


def _negative_auc(cfg, out_root, labels, tensors, splits, script_hash, config_hash):
    spec = CONFIGS["A_primary"]
    pieces = []
    for fold in range(5):
        key = ("A_primary", "NC_M2", 0, 0, fold)
        train_ids, test_ids, _, _, _, y_test = _unit_data(
            labels, tensors, splits, *key, cfg
        )
        path = _unit_path(out_root, *key)
        meta_expected = {
            "task": TASK, "prereg_sha256": PREREG_SHA,
            "script_sha256": script_hash, "config_sha256": config_hash,
            "config": "A_primary", "model": "NC_M2", "seed": 0,
            "repeat": 0, "outer_fold": fold,
        }
        if not path.is_file():
            raise FileNotFoundError("Negative control is missing an outer fold")
        p, _ = read_unit(path, meta_expected, test_ids, y_test)
        pieces.append(pd.DataFrame({"record_id": test_ids, "y": y_test, "p": p}))
    all_scores = pd.concat(pieces, ignore_index=True)
    expected = set(labels.index[~labels.excluded.astype(bool)])
    if (len(all_scores) != len(expected) or
        all_scores.record_id.duplicated().any() or
        set(all_scores.record_id) != expected):
        raise AssertionError("Negative control lacks five distinct outer-test folds")
    return float(discrimination(all_scores.y, all_scores.p)[0])


def _reproducibility(cfg, out_root, labels, tensors, splits, source):
    """Train one actual full-run unit twice, without changing either unit file."""
    record = Path(out_root) / "reproducibility.json"
    if record.exists():
        result = json.loads(record.read_text(encoding="utf-8"))
        if (result.get("script_sha256") != source["script_sha256"] or
            result.get("max_abs_diff", np.inf) > 1e-6):
            raise AssertionError("Resumed reproducibility certificate differs")
        return result
    key = ("A_primary", "M2", 0, 0, 0)
    train_ids, test_ids, x_train, x_test, y_train, y_test = _unit_data(
        labels, tensors, splits, *key, cfg
    )
    expected_meta = {
        "task": TASK, "prereg_sha256": PREREG_SHA,
        "script_sha256": source["script_sha256"],
        "config_sha256": source["config_sha256"],
        "config": "A_primary", "model": "M2", "seed": 0,
        "repeat": 0, "outer_fold": 0,
    }
    first_p, _ = read_unit(_unit_path(out_root, *key), expected_meta, test_ids, y_test)
    second_p, _ = fit_predict("M2", 0, x_train, y_train, train_ids,
                              test_ids, x_test, 0, 0, cfg)
    difference = float(np.max(np.abs(first_p - second_p)))
    if difference > 1e-6:
        raise AssertionError(f"Deterministic unit rerun differs: {difference:.3g}")
    result = {"unit": "A_primary_M2_s0_r00_f0", "max_abs_diff": difference,
              "tolerance": 1e-6, "script_sha256": source["script_sha256"]}
    atomic_json(record, result)
    return result


def _unit_aggregation(cfg, config_name, out_root, labels, tensors, splits,
                      repeat_count, script_hash, config_hash):
    rows = []
    train_logs = []
    spec = CONFIGS[config_name]
    for repeat in range(repeat_count):
        for fold in range(5):
            train_ids, test_ids = fold_ids(splits, spec, repeat, fold)
            true_y = labels.loc[test_ids, spec["target"]].to_numpy(dtype=np.int64)
            block = pd.DataFrame({
                "record_id": test_ids, "repeat": repeat,
                "outer_fold": fold, "y": true_y,
            })
            for model in ("M2", "M3"):
                for seed in SEEDS:
                    key = (config_name, model, seed, repeat, fold)
                    meta_expected = {
                        "task": TASK, "prereg_sha256": PREREG_SHA,
                        "script_sha256": script_hash, "config_sha256": config_hash,
                        "config": config_name, "model": model, "seed": seed,
                        "repeat": repeat, "outer_fold": fold,
                    }
                    p, meta = read_unit(
                        _unit_path(out_root, *key), meta_expected, test_ids, true_y
                    )
                    block[f"p_{model}_s{seed}"] = p
                    train_logs.append(meta)
            if config_name == "A_primary":
                extra = ["M2UC"] + (["NC_M2"] if repeat == 0 else [])
                for model in extra:
                    key = (config_name, model, 0, repeat, fold)
                    meta_expected = {
                        "task": TASK, "prereg_sha256": PREREG_SHA,
                        "script_sha256": script_hash, "config_sha256": config_hash,
                        "config": config_name, "model": model, "seed": 0,
                        "repeat": repeat, "outer_fold": fold,
                    }
                    p, meta = read_unit(
                        _unit_path(out_root, *key), meta_expected, test_ids, true_y
                    )
                    if model == "M2UC":
                        block["p_M2UC_s0"] = p
                    train_logs.append(meta)
            rows.append(block)
    oof = pd.concat(rows, ignore_index=True).sort_values(
        ["repeat", "record_id"]
    ).reset_index(drop=True)
    expected_ids = (set(labels.index) if spec["split"] == "split_noexcl.csv" else
                    set(labels.index[~labels.excluded.astype(bool)]))
    if (len(oof) != len(expected_ids) * repeat_count or
        oof.duplicated(["record_id", "repeat"]).any() or
        set(oof.record_id) != expected_ids or
        not oof.groupby("record_id").repeat.nunique().eq(repeat_count).all() or
        oof.groupby("record_id").y.nunique().max() != 1):
        raise AssertionError("Missing, duplicated, or mislabeled OOF predictions")
    columns = [name for name in oof if name.startswith("p_")]
    if not np.isfinite(oof[columns].to_numpy(dtype=float)).all():
        raise AssertionError("Nonfinite OOF prediction")
    train_log = pd.DataFrame(train_logs).sort_values(
        ["repeat", "outer_fold", "model", "seed"]
    ).reset_index(drop=True)
    return oof, train_log


def _summary_metrics(y, p, cfg):
    numbers = all_metrics(y, p)
    ci = discrimination_ci(y, p, n_resamples=cfg["bootstrap_resamples"],
                           seed=cfg["bootstrap_seed"])
    return {**numbers, **ci}


def _model_metrics(oof, cfg, model, seeds=(0, 1, 2)):
    unique_ids = oof.groupby("record_id", sort=True).y.first()
    expected_repeats = oof.repeat.nunique()
    averaged = oof.groupby("record_id", sort=True).size()
    if not averaged.eq(expected_repeats).all():
        raise AssertionError("Missing repeated out-of-fold prediction")
    seed_scores = {}
    for seed in seeds:
        col = f"p_{model}_s{seed}"
        means = oof.groupby("record_id", sort=True)[col].mean()
        seed_scores[f"s{seed}"] = float(discrimination(unique_ids, means)[0])
    maximum_seed_spread = max(seed_scores.values()) - min(seed_scores.values())
    seed0_auroc = []
    seed0_auprc = []
    for repeat in sorted(oof.repeat.unique()):
        frame = oof.loc[oof.repeat == repeat]
        auc, ap = discrimination(frame.y, frame[f"p_{model}_s0"])
        seed0_auroc.append(auc)
        seed0_auprc.append(ap)
    cols = [f"p_{model}_s{s}" for s in seeds]
    combined = oof.assign(_p=oof[cols].mean(axis=1)).groupby(
        "record_id", sort=True
    )._p.mean()
    return {
        "seed_auroc_averaged_over_repeats": seed_scores,
        "max_seed_auroc_difference": maximum_seed_spread,
        "seed0_repeat_auroc_mean": float(np.mean(seed0_auroc)),
        "seed0_repeat_auroc_sd": float(np.std(seed0_auroc, ddof=1)),
        "seed0_repeat_auprc_mean": float(np.mean(seed0_auprc)),
        "seed0_repeat_auprc_sd": float(np.std(seed0_auprc, ddof=1)),
        "mean_probability_over_repeats_and_seeds": _summary_metrics(unique_ids, combined, cfg),
    }


def _unit_manifest_sha8(directory):
    digest = hashlib.sha256()
    units = sorted((Path(directory) / "units").glob("*.npz"))
    for path in units:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()[:8], len(units)


def _publish_summary(path, payload_writer):
    """Allow interrupted finalization to resume without overwriting prior bytes."""
    import io
    buffer = io.BytesIO()
    payload_writer(buffer)
    desired = buffer.getvalue()
    if path.exists():
        if sha256(path) != hashlib.sha256(desired).hexdigest():
            raise AssertionError(f"Existing final output differs: {path.name}")
        return False
    atomic_new(path, lambda stream: stream.write(desired))
    return True


def _finalize(cfg, out_root, labels, tensors, splits, plan,
              source, started, control_auc, reproducibility, complete_count):
    summaries = {}
    epoch_values = []
    saved_training_seconds = 0.0
    for name, count in plan["repeats"].items():
        print(f"{name}: aggregating OOF and 10000x record bootstraps", flush=True)
        oof, train_log = _unit_aggregation(
            cfg, name, out_root, labels, tensors, splits, count,
            source["script_sha256"], source["config_sha256"],
        )
        epoch_values.extend(train_log.epoch_stop.astype(int).tolist())
        saved_training_seconds += float(train_log.runtime_s.sum())
        model_results = {model: _model_metrics(oof, cfg, model) for model in MODELS}
        if name == "A_primary":
            model_results["M2UC"] = _model_metrics(oof, cfg, "M2UC", seeds=(0,))
        metrics = {
            "task": TASK, "prereg_sha256": PREREG_SHA, "config": name,
            "num_repeats": count, "budget_rule": plan["rule"],
            "models": model_results,
        }
        folder = Path(out_root) / name
        _publish_summary(folder / "oof.parquet", lambda f: oof.to_parquet(f, index=False))
        _publish_summary(folder / "train_log.csv", lambda f: f.write(
            train_log.to_csv(index=False).encode("utf-8")
        ))
        _publish_summary(folder / "metrics.json", lambda f: f.write(
            json.dumps(_json_safe(metrics), sort_keys=True, indent=2,
                       allow_nan=False).encode("utf-8")
        ))
        unit_hash, unit_count = _unit_manifest_sha8(folder)
        hashes = {filename: sha8(folder / filename) for filename in SUMMARY_FILES}
        hashes["units_manifest"] = unit_hash
        hashes["unit_count"] = unit_count
        summaries[name] = {"metrics": metrics, "hashes": hashes}
    epoch_values = np.asarray(epoch_values, dtype=float)
    q25, median, q75 = np.percentile(epoch_values, [25, 50, 75])
    print("=== REPORT P21-T04 ===")
    print("status: PASS")
    print(f"env: PC-A CPU; python={sys.version.split()[0]}; torch={torch.__version__}; "
          "torch_threads=4; GPU=unused")
    print(f"runtime_this_session_s={time.perf_counter() - started:.2f}; "
          f"completed_unit_training_s={saved_training_seconds:.2f}")
    print("script_sha256_8: " + source["script_sha256"][:8])
    print("prereg_version: 1.2; prereg_sha256_director_declared: " + PREREG_SHA)
    print(f"budget_rule: {plan['rule']}; full_est_h={plan['estimate_full_s']/3600:.2f}; "
          f"AB_est_h={plan['estimate_A_B_mandatory_s']/3600:.2f}; "
          f"repeats={plan['repeats']}")
    print(f"units_completed: {complete_count}; expected={sum(1 for _ in expected_units(plan['repeats']))}")
    print(f"negative_control_A_M2_repeat0_s0_AUROC: {control_auc:.4f}")
    print(f"reproducibility_max_abs_diff: {reproducibility['max_abs_diff']:.3g}; limit=1e-6")
    print(f"epoch_stop_median_IQR: {median:.1f}[{q25:.1f},{q75:.1f}]")
    for name, result in summaries.items():
        for model, scores in result["metrics"]["models"].items():
            averaged = scores["mean_probability_over_repeats_and_seeds"]
            seed_aucs = ",".join(f"{v:.4f}" for v in scores["seed_auroc_averaged_over_repeats"].values())
            print(f"{name}_{model}: seed_auc=[{seed_aucs}]; "
                  f"seed_delta_max={scores['max_seed_auroc_difference']:.4f}; "
                  f"AUC_repeat_s0={scores['seed0_repeat_auroc_mean']:.4f}+/-{scores['seed0_repeat_auroc_sd']:.4f}; "
                  f"AUROC_avg={averaged['auroc']:.4f}[{averaged['auroc_ci95'][0]:.4f},{averaged['auroc_ci95'][1]:.4f}]; "
                  f"AUPRC={averaged['auprc_average_precision']:.4f}[{averaged['auprc_ci95'][0]:.4f},{averaged['auprc_ci95'][1]:.4f}]; "
                  f"Brier={averaged['brier']:.4f}; "
                  f"cal_int={averaged['calibration_intercept_offset_logit']:.4f}; "
                  f"cal_slope={averaged['calibration_slope']:.4f}; "
                  f"Se@Sp85={averaged['sensitivity_at_specificity_0_85']:.4f}")
    for name, result in summaries.items():
        hashes = result["hashes"]
        print(f"{name}_sha8: units_manifest={hashes['units_manifest']}; "
              f"oof={hashes['oof.parquet']}; log={hashes['train_log.csv']}; "
              f"metrics={hashes['metrics.json']}")
    print("acceptance: pytest=PASS; budget=PASS; all_units=PASS; "
          "oof_once_per_repeat_seed_model=PASS; no_nan=PASS; "
          "no_outer_test_leakage=PASS; negative_control=PASS; "
          "fold_reproducibility=PASS; no_unit_overwrite=PASS")
    print("warnings: torch may emit an expected padding='same' even-kernel performance warning; see log")
    print("=== END ===")


def _full(cfg, config_path):
    started = time.perf_counter()
    require_torch()
    _pytest_t04()
    out_root = Path(cfg["output_root"]) / cfg["run_name"]
    pilot_file = out_root / "pilot.json"
    if not pilot_file.is_file():
        raise FileNotFoundError("Run --pilot first; no budget decision is available")
    pilot = json.loads(pilot_file.read_text(encoding="utf-8"))
    source = _source_identity(config_path)
    if pilot["source"] != source or pilot["prereg_sha256"] != PREREG_SHA:
        raise AssertionError("Pilot source/config/common hashes changed")
    if (pilot["torch_version"] != torch.__version__ or
        pilot["python"] != sys.version.split()[0]):
        raise AssertionError("PyTorch or Python version differs from pilot")
    plan = budget_estimate(pilot["pilot"]["M2"]["train_runtime_s"],
                           pilot["pilot"]["M3"]["train_runtime_s"], cfg["budget_hours"])
    if plan != pilot["budget"]:
        raise AssertionError("Pilot budget changed after preregistration")
    print("=== P21-T04 FULL START ===", flush=True)
    print(f"budget_rule: {plan['rule']}; "
          f"full_est_h={plan['estimate_full_s']/3600:.2f}; "
          f"A_B_est_h={plan['estimate_A_B_mandatory_s']/3600:.2f}; "
          f"repeats={plan['repeats']}", flush=True)
    if plan["repeats"] is None:
        print("=== REPORT P21-T04 ===\nstatus: STOP_BUDGET\n"
              "detail: mandatory A_primary+B_noexcl estimate exceeds 48h\n"
              "outputs: no DL training performed\n=== END ===")
        return 2
    if (out_root / "FINALIZED.json").exists():
        raise FileExistsError("T04 already finalized; do not re-run its results")
    labels, arrays, splits, input_hashes = load_data(cfg)
    if input_hashes != pilot["input_sha256"]:
        raise AssertionError("One or more frozen T02 inputs changed since pilot")
    lock = {"task": TASK, "prereg_sha256": PREREG_SHA,
            "source": source, "input_sha256": input_hashes,
            "torch_version": torch.__version__, "python": sys.version.split()[0],
            "budget": plan, "pilot_sha256": sha256(pilot_file)}
    lock_path = out_root / "run_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise AssertionError("Resume lock differs from the original full run")
    else:
        atomic_json(lock_path, lock)
    tensors = {(win, uc): fixed_normalize(arr, with_uc=uc)
               for win, arr in arrays.items() for uc in (False, True)}
    units = list(expected_units(plan["repeats"]))
    expected_files = {
        (Path(out_root) / name / "units" / _unit_filename(model, seed, repeat, fold))
        for name, model, seed, repeat, fold in units
    }
    for name in CONFIGS:
        folder = out_root / name / "units"
        if folder.exists():
            unexpected = set(folder.glob("*.npz")) - expected_files
            if unexpected:
                raise AssertionError("Found units from another plan; nothing overwritten")
    print(f"units_planned: {len(units)}", flush=True)
    control_auc = None
    done = 0
    resumed = 0
    for key in units:
        _, _, was_cached = run_one_unit(
            cfg, out_root, labels, tensors, splits, key,
            source["script_sha256"], source["config_sha256"],
        )
        done += 1
        resumed += int(was_cached)
        if done % 10 == 0 or done == len(units):
            print(f"progress: {done}/{len(units)} units; resumed={resumed}; "
                  f"last_config={key[0]}; last_repeat={key[3]}; "
                  f"elapsed_s={time.perf_counter()-started:.1f}", flush=True)
        if key == ("A_primary", "NC_M2", 0, 0, 4):
            control_auc = _negative_auc(
                cfg, out_root, labels, tensors, splits,
                source["script_sha256"], source["config_sha256"],
            )
            lower, upper = cfg["negative_control_bounds"]
            print(f"negative_control_A_M2_repeat0_s0_AUROC: {control_auc:.4f}", flush=True)
            if not (lower <= control_auc <= upper):
                print("=== REPORT P21-T04 ===\nstatus: FAIL\n"
                      "reason: preregistered negative control outside [0.40,0.60]\n"
                      "action: STOP; no seed changes; retain saved units\n=== END ===")
                return 1
    if control_auc is None:
        raise AssertionError("Negative control was not completed")
    reproducibility = _reproducibility(
        cfg, out_root, labels, tensors, splits, source
    )
    _finalize(cfg, out_root, labels, tensors, splits, plan, source,
              started, control_auc, reproducibility, done)
    # Marker is written last. An interrupted summary remains idempotently resumable.
    atomic_json(out_root / "FINALIZED.json", {
        "status": "PASS", "prereg_sha256": PREREG_SHA,
        "script_sha256": source["script_sha256"],
        "unit_count": done, "budget_rule": plan["rule"],
        "negative_control_auroc": control_auc,
        "reproducibility_max_abs_diff": reproducibility["max_abs_diff"],
    })
    return 0


def main():
    parser = argparse.ArgumentParser(description=TASK)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--config", type=Path,
                        default=Path("P21_CTG_Acidemia/configs/t04.yaml"))
    args = parser.parse_args()
    try:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        locked_config(cfg)
        return _pilot(cfg, args.config) if args.pilot else _full(cfg, args.config)
    except KeyboardInterrupt:
        print("=== REPORT P21-T04 ===\nstatus: INTERRUPTED\n"
              "action: run --full again; completed units remain intact\n=== END ===", flush=True)
        return 130
    except Exception as exc:
        print("=== REPORT P21-T04 ===")
        print("status: FAIL")
        print("error_type: " + type(exc).__name__)
        print("error: " + str(exc))
        print("action: STOP; do not delete or overwrite units")
        print("=== END ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())

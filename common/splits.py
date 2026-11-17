"""Deterministic repeated nested stratified splits at record level."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def build_nested_splits(record_ids, labels, repeats=10, outer_folds=5, inner_folds=5, seed=2026):
    """Long format: inner_fold=-1 marks an outer-test record."""
    ids = np.asarray(record_ids, dtype=str)
    y = np.asarray(labels, dtype=int)
    if len(ids) != len(y) or len(ids) != len(set(ids)):
        raise ValueError("Record IDs must be unique and match labels")
    rows = []
    for repeat in range(repeats):
        outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed + repeat)
        for outer_fold, (train_idx, test_idx) in enumerate(outer.split(ids, y)):
            inner_assignment = np.full(len(ids), -2, dtype=int)
            inner_assignment[test_idx] = -1
            inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + repeat)
            for inner_fold, (_, inner_val_idx) in enumerate(inner.split(ids[train_idx], y[train_idx])):
                inner_assignment[train_idx[inner_val_idx]] = inner_fold
            if np.any(inner_assignment == -2):
                raise AssertionError("Incomplete inner-fold assignment")
            rows.extend(
                (record_id, repeat, outer_fold, int(inner_assignment[position]))
                for position, record_id in enumerate(ids)
            )
    return pd.DataFrame(rows, columns=["record_id", "repeat", "outer_fold", "inner_fold"])


def verify_nested_splits(splits, record_ids, repeats=10, outer_folds=5, inner_folds=5):
    ids = set(map(str, record_ids))
    if len(ids) != len(record_ids):
        return False
    if len(splits) != len(ids) * repeats * outer_folds:
        return False
    for repeat in range(repeats):
        test_seen = []
        for fold in range(outer_folds):
            block = splits[(splits.repeat == repeat) & (splits.outer_fold == fold)]
            if len(block) != len(ids) or set(block.record_id) != ids or block.record_id.duplicated().any():
                return False
            train = set(block.loc[block.inner_fold >= 0, "record_id"])
            test = set(block.loc[block.inner_fold == -1, "record_id"])
            if train & test or train | test != ids:
                return False
            if set(block.inner_fold) - set(range(-1, inner_folds)):
                return False
            if set(block.loc[block.inner_fold >= 0, "inner_fold"]) != set(range(inner_folds)):
                return False
            test_seen.extend(test)
        if len(test_seen) != len(ids) or set(test_seen) != ids:
            return False
    return True

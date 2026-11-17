"""Record-level percentile bootstrap for P21-T03 discrimination CIs."""

import numpy as np

from common.metrics import discrimination


def discrimination_ci(y, p, n_resamples=10000, seed=42):
    """Resample records (not repeats); return AUROC and AP percentile CIs."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if n_resamples <= 0 or len(y) != len(p) or not len(y):
        raise ValueError("Invalid bootstrap dimensions")
    # Reject invalid inputs before any resampling.
    discrimination(y, p)
    rng = np.random.default_rng(seed)
    scores = np.empty((n_resamples, 2), dtype=np.float64)
    valid = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), size=len(y))
        if np.unique(y[idx]).size != 2:
            continue
        scores[valid] = discrimination(y[idx], p[idx])
        valid += 1
    if valid == 0:
        raise ValueError("All bootstrap replicates contain one class")
    bounds = np.percentile(scores[:valid], [2.5, 97.5], axis=0)
    return {
        "auroc_ci95": [float(bounds[0, 0]), float(bounds[1, 0])],
        "auprc_ci95": [float(bounds[0, 1]), float(bounds[1, 1])],
        "bootstrap_seed": int(seed),
        "bootstrap_requested": int(n_resamples),
        "bootstrap_valid": int(valid),
        "bootstrap_unit": "record",
    }

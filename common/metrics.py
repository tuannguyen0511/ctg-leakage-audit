"""Binary discrimination and calibration metrics for P21-T03.

ROC operating-point interpolation is descriptive only. No decision threshold is
selected, retained, or applied to outer-test records.
"""

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def _validated(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("y and p must be nonempty, equal-length vectors")
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError("Both binary classes are required for metrics")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Predicted probabilities must be finite in [0, 1]")
    return y, p


def discrimination(y, p):
    """ROC AUROC and average precision (non-interpolated AUPRC)."""
    y, p = _validated(y, p)
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def calibration_intercept(y, p):
    """Unregularized calibration-in-the-large offset of logit(p)."""
    y, p = _validated(y, p)
    x = logit(np.clip(p, 1e-6, 1.0 - 1e-6))
    target = float(y.mean())
    return float(brentq(lambda a: float(expit(x + a).mean()) - target, -100, 100))


def calibration_slope(y, p):
    """Unregularized logistic fit of outcome on predicted log odds."""
    y, p = _validated(y, p)
    x = logit(np.clip(p, 1e-6, 1.0 - 1e-6))
    if float(np.ptp(x)) < 1e-12:
        return float("nan")  # A constant predictor does not identify a slope.

    def loss_grad(params):
        z = params[0] + params[1] * x
        loss = float(np.logaddexp(0.0, z).sum() - np.dot(y, z))
        residual = expit(z) - y
        grad = np.array([residual.sum(), np.dot(residual, x)])
        return loss, grad

    fit = minimize(
        loss_grad, x0=np.array([0.0, 1.0]), method="L-BFGS-B", jac=True,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        return float("nan")
    return float(fit.x[1])


def sensitivity_at_specificity(y, p, specificity=0.85):
    """Interpolate observed ROC at FPR = 1-specificity; no fitted cutoff."""
    y, p = _validated(y, p)
    if not 0 < specificity < 1:
        raise ValueError("Specificity must be in (0, 1)")
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    fpr_unique = np.unique(fpr)
    best_tpr = np.array([np.max(tpr[fpr == x]) for x in fpr_unique])
    return float(np.interp(1.0 - specificity, fpr_unique, best_tpr))


def all_metrics(y, p):
    y, p = _validated(y, p)
    auc, ap = discrimination(y, p)
    return {
        "auroc": auc,
        "auprc_average_precision": ap,
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept_offset_logit": calibration_intercept(y, p),
        "calibration_slope": calibration_slope(y, p),
        "sensitivity_at_specificity_0_85": sensitivity_at_specificity(y, p),
        "prevalence": float(y.mean()),
    }

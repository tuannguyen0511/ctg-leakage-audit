"""P21-T11b: post hoc retraining experiments requested by the 2026-11-14 review.

Run from D:/Tuan/AI_training in Git Bash (CPU only, 4 threads, as T05):
    envs/pn/Scripts/python -m P21_CTG_Acidemia.src.t11b_review_runs            # all steps
    envs/pn/Scripts/python -m P21_CTG_Acidemia.src.t11b_review_runs e2b e8c   # some steps
Every fitted unit is saved once under outputs/t11b_review/units/<step>/ and is
skipped on a rerun, so the run can be stopped (Ctrl+C) and restarted at any time.
Frozen T03-T05 outputs and raw data are only read.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

from common.io_wfdb import read_record
from P21_CTG_Acidemia.src.ctg_features import clean_signals, compute_features, build_dl_1hz
from P21_CTG_Acidemia.src import t05_leakage as t05  # sets 4 threads, disables CUDA

t03, t04 = t05.t03, t05.t04
P = Path(__file__).resolve().parents[1]
OUT = P / "outputs" / "t11b_review"
CFG = yaml.safe_load((P / "configs" / "t05.yaml").read_text(encoding="utf-8"))
T03 = t05.t03_cfg(CFG)
FS = 4
N_PERM_S, N_PERM_L2, N_PERM_L1 = 100, 20, 10
B = 10000
STEPS = ("e2b", "e8c", "e10a", "e4d", "e2", "e8a", "e7", "e3", "e9", "e1b")
auc = t05.auc_fast


# ---------------------------------------------------------------- infrastructure
def unit(step, name, fn):
    """Fit once, save, reuse. fn returns a dict of numpy arrays."""
    path = OUT / "units" / step / f"{name}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    started = time.perf_counter()
    res = {k: np.asarray(v) for k, v in fn().items()}
    res["elapsed_s"] = np.asarray(time.perf_counter() - started)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **res)
    os.replace(tmp, path)
    print(f"{step}/{name} {float(res['elapsed_s']):.0f}s", flush=True)
    return res


def write(df, name):
    df.to_csv(OUT / name, index=False)
    print(f"WROTE {name}", flush=True)


def load():
    data = t05.load_inputs(CFG)
    seg, _ = t05.build_segments(CFG, data, Path(CFG["output_dir"]))  # frozen cache, read only
    return data, seg


def s_fold(data, r, f):
    sp = data["splits"]
    b = sp[(sp.repeat == r) & (sp.outer_fold == f)]
    pos = {rid: i for i, rid in enumerate(data["ids"])}
    tr, te = b[b.inner_fold >= 0], b[b.inner_fold == -1]
    return (np.array([pos[x] for x in tr.record_id]), np.array([pos[x] for x in te.record_id]),
            tr.inner_fold.to_numpy(int))


def inner_folds(y, seed):
    inner = np.empty(len(y), dtype=int)
    for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        inner[v] = k
    return inner


def m1(x_tr, y_tr, inner, x_te, tr_ids=None, te_ids=None):
    tr_ids = np.char.add("tr", np.arange(len(y_tr)).astype(str)) if tr_ids is None else tr_ids
    te_ids = np.char.add("te", np.arange(len(x_te)).astype(str)) if te_ids is None else te_ids
    return t03.tune_and_predict("M1", x_tr, y_tr, inner, x_te, T03, tr_ids, te_ids)[0]


def net(arch, x_tr, y_tr, x_te, r, f, seed=0, groups=None):
    """T05 training loop unchanged (grouped validation when groups is given).
    shortcut: swaps the architecture by patching t04.make_model; fine for this single-process script."""
    original = t04.make_model
    t04.make_model = lambda _name, in_channels=2: original(arch, in_channels)
    try:
        return t05.train_m2(x_tr, y_tr, x_te, np.zeros(len(x_te), int), seed, r, f, CFG, group_ids=groups)[0]
    finally:
        t04.make_model = original


def m2_record(x, y, ids, tr, te, r, f, seed=0):
    return t04.fit_predict("M2", seed, x[tr], y[tr], ids[tr], ids[te], x[te], r, f, CFG)[0]


def frozen_s(model, col):
    path = "t03_baselines" if model in ("M0", "M1") else "t04_dl"
    f = pd.read_parquet(P / "outputs" / path / "A_primary" / "oof.parquet")
    f["record_id"] = f.record_id.astype(str)
    return {r: g.set_index("record_id")[col] for r, g in f.groupby("repeat")}


def boot(n, stat, groups=None, seed=42):
    """Percentile CI of stat(idx); groups -> resample whole clusters (recordings)."""
    rng = np.random.default_rng(seed)
    if groups is not None:
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    vals = []
    for _ in range(B):
        if groups is None:
            ix = rng.integers(0, n, n)
        else:
            ix = np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))])
        try:
            vals.append(stat(ix))
        except AssertionError:  # one-class replicate
            continue
    v = np.asarray(vals)
    return {"ci95_low": np.quantile(v, .025), "ci95_high": np.quantile(v, .975),
            "p_raw": min(1.0, max(1 / len(v), 2 * min(np.mean(v <= 0), np.mean(v >= 0))))}


def raw_trace(rid):
    rec = read_record(Path(CFG["raw_dir"]), str(rid))
    ch = {c.upper(): i for i, c in enumerate(rec.channel_names)}
    fhr, uc = clean_signals(rec.signal[:, ch["FHR"]], rec.signal[:, ch["UC"]], t05.t02_cfg(P))
    pos = rec.clinical.get("Pos. II.st.", np.nan)
    try:
        pos = float(pos)
    except (TypeError, ValueError):
        pos = np.nan
    no2 = not (np.isfinite(pos) and pos >= 0)
    return fhr, uc, (len(fhr) if no2 else int(pos)), no2


# ---------------------------------------------------------------- record windows (E10a, E4d)
def window_inputs(data, kind):
    """kind='last30rec': last 30 min of the recording (stage 2 included);
    kind='shift10': stage-1 window ending 10 min earlier for recordings without stage 2."""
    path = OUT / f"inputs_{kind}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return z["features"], z["tensor"]
    cfg2 = t05.t02_cfg(P)
    feats, tens = [], []
    for rid in data["ids"]:
        fhr, uc, end, no2 = raw_trace(rid)
        if kind == "last30rec":
            end = len(fhr)
        elif no2:
            end = max(0, end - 10 * 60 * FS)
        w = slice(max(0, end - 30 * 60 * FS), end)
        f = compute_features(fhr[w], uc[w], cfg2)
        feats.append([f[n] for n in t03.FEATURES])
        tens.append(build_dl_1hz(fhr[w], uc[w], 30, cfg2))
    x, t = np.asarray(feats, float), t04.fixed_normalize(np.stack(tens))
    np.savez_compressed(path, features=x, tensor=t)
    return x, t


def run_window(data, kind, nets=True):
    x, t = window_inputs(data, kind)
    y, ids = data["y"], data["ids"]
    rows = []
    m1f, m2f = frozen_s("M1", "p_M1"), frozen_s("M2", "p_M2_s0")
    for r in range(10):
        p1, p2 = np.zeros(len(y)), np.zeros(len(y))
        for f in range(5):
            tr, te, inner = s_fold(data, r, f)
            p1[te] = unit(f"e_{kind}", f"M1_r{r}_f{f}", lambda: {"p": m1(x[tr], y[tr], inner, x[te], ids[tr], ids[te])})["p"]
            if nets:
                p2[te] = unit(f"e_{kind}", f"M2_r{r}_f{f}", lambda: {"p": m2_record(t, y, ids, tr, te, r, f)})["p"]
        rows.append({"window": kind, "repeat": r, "M1": auc(y, p1), "M1_W30": auc(y, m1f[r].loc[ids].to_numpy()),
                     "M2_s0": auc(y, p2) if nets else np.nan, "M2_s0_W30": auc(y, m2f[r].loc[ids].to_numpy())})
    write(pd.DataFrame(rows), f"{kind}_summary.csv")


# ---------------------------------------------------------------- E2b mask ablation
def e2b(data, seg):
    y, ids, x = data["y"], data["ids"], data["tensor"][:, :1]
    nan = data["labels"].loc[ids, "fhr_nan_pct_w30"].to_numpy(float)
    m2f = frozen_s("M2", "p_M2_s0")
    rows = []
    for r in range(10):
        p = np.zeros(len(y))
        for f in range(5):
            tr, te, _ = s_fold(data, r, f)
            p[te] = unit("e2b", f"M2nomask_r{r}_f{f}", lambda: {"p": m2_record(x, y, ids, tr, te, r, f)})["p"]
        q = m2f[r].loc[ids].to_numpy()
        rows.append({"repeat": r, "AUROC_nomask": auc(y, p), "AUROC_mask": auc(y, q),
                     "rho_nan_nomask": spearmanr(p, nan)[0], "rho_nan_mask": spearmanr(q, nan)[0]})
    write(pd.DataFrame(rows), "e2b_summary.csv")


# ---------------------------------------------------------------- E8c L3 winner's curse
def track(x_tr, y_tr, tr_ids, te_ids, x_te, r, f, epochs=60, seed=0):
    """T04 M2 training without early stopping; returns per-epoch val AUROC and test probs."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    t04.seed_everything(seed)
    fit, val = t04.train_val_indices(tr_ids, y_tr, te_ids, r, f, seed)
    model = t04.make_model("M2", x_tr.shape[1])
    pw = (len(fit) - y_tr[fit].sum()) / y_tr[fit].sum()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    loader = DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(x_tr[fit])),
                                      torch.from_numpy(y_tr[fit].astype(np.float32))),
                        batch_size=CFG["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    va, tp = [], []
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            opt.step()
        va.append(auc(y_tr[val], t04.predict_prob(model, x_tr[val])))
        tp.append(t04.predict_prob(model, x_te))
    return {"val_auc": np.array(va), "test_p": np.stack(tp)}


def e8c(data, seg):
    y, ids, x = data["y"], data["ids"], data["tensor"]
    rows = []
    for r in range(3):
        res, tes = [], []
        for f in range(5):
            tr, te, _ = s_fold(data, r, f)
            res.append(unit("e8c", f"track_r{r}_f{f}", lambda: track(x[tr], y[tr], ids[tr], ids[te], x[te], r, f)))
            tes.append(te)
        for k in (1, 5, 10, 20, 30, 60):
            pv, pt = np.zeros(len(y)), np.zeros(len(y))
            for u, te in zip(res, tes):
                pv[te] = u["test_p"][np.argmax(u["val_auc"][:k])]
                fold_auc = [auc(y[te], p) for p in u["test_p"][:k]]
                pt[te] = u["test_p"][int(np.argmax(fold_auc))]
            rows.append({"repeat": r, "candidates": k, "AUROC_val_selected": auc(y, pv),
                         "AUROC_test_selected": auc(y, pt), "inflation": auc(y, pt) - auc(y, pv)})
    write(pd.DataFrame(rows), "e8c_summary.csv")


# ---------------------------------------------------------------- E2 identical duplication for M1
def dup_rows(y, factor=None):
    if factor is None:  # balance, exactly as T05 L2-M2
        return t05.random_oversample(np.arange(len(y)), y, CFG["oversample_seed"])[:2]
    rows = np.r_[np.arange(len(y)), np.repeat(np.flatnonzero(y == 1), factor - 1)]
    return rows, y[rows]


def l2_dup_m1(xi, rows, yd, r, f, tag, step):
    tr, te = list(KFold(5, shuffle=True, random_state=2026 + r).split(yd))[f]
    inner = inner_folds(yd[tr], 2026 + 100 + 10 * r + f)
    return te, unit(step, f"{tag}_r{r}_f{f}", lambda: {"p": m1(xi[rows][tr], yd[tr], inner, xi[rows][te])})["p"]


def e2(data, seg):
    y = data["y"]
    xi = SimpleImputer(strategy="median").fit_transform(data["features"])  # full-cohort, as T05 L2-M1
    rows, yd = dup_rows(y)
    m1f = frozen_s("M1", "p_M1")
    l2 = pd.read_parquet(P / "outputs/t05_leakage/oof_L2.parquet")
    l2 = l2[(l2.model == "M2") & l2.row_id.str.startswith("real::")]
    l2["record_id"] = l2.record_id.astype(str)
    m2f = {s: frozen_s("M2", f"p_M2_s{s}") for s in range(3)}
    p_dup, p_s1, p_l2m2, p_s2 = [], [], [], []
    out = []
    for r in range(3):
        p = np.zeros(len(yd))
        for f in range(5):
            te, pp = l2_dup_m1(xi, rows, yd, r, f, "M1dup", "e2")
            p[te] = pp
        rp = p[:len(y)]
        p_dup.append(rp)
        p_s1.append(m1f[r].loc[data["ids"]].to_numpy())
        for s in range(3):
            g = l2[(l2.repeat == r) & (l2.seed == s)].set_index("record_id").loc[data["ids"]]
            p_l2m2.append(g.p.to_numpy())
            p_s2.append(m2f[s][r].loc[data["ids"]].to_numpy())
        out.append({"repeat": r, "AUROC_all_rows": auc(yd, p), "AUROC_real": auc(y, rp),
                    "AUROC_S": auc(y, p_s1[-1]), "delta_real": auc(y, rp) - auc(y, p_s1[-1])})

    def d1(ix):
        return np.mean([auc(y[ix], a[ix]) - auc(y[ix], b[ix]) for a, b in zip(p_dup, p_s1)])

    def d2(ix):
        return np.mean([auc(y[ix], a[ix]) - auc(y[ix], b[ix]) for a, b in zip(p_l2m2, p_s2)])
    all_ = np.arange(len(y))
    out.append({"repeat": "pooled single-model", "delta_real": d1(all_), **{f"M1_{k}": v for k, v in boot(len(y), d1).items()},
                "delta_M2": d2(all_), "interaction_M2_minus_M1": d2(all_) - d1(all_),
                **{f"int_{k}": v for k, v in boot(len(y), lambda ix: d2(ix) - d1(ix)).items()}})
    write(pd.DataFrame(out), "e2_summary.csv")


# ---------------------------------------------------------------- E8a recording-identification probe
def e8a(data, seg):
    import torch
    from torch import nn
    rec = seg["record_id"]
    start = np.array([int(s.split("__")[1]) for s in seg["row_id"]])
    rng = np.random.default_rng(42)
    rows = []
    for variant in ("random", "temporal_gap"):
        tr, te = [], []
        for rid in np.unique(rec):
            ix = np.flatnonzero(rec == rid)
            ix = ix[np.argsort(start[ix])]
            if variant == "random" and len(ix) >= 2:
                ix = rng.permutation(ix)
                k = max(1, int(round(0.2 * len(ix))))
                te += list(ix[:k]); tr += list(ix[k:])
            elif variant == "temporal_gap" and len(ix) >= 3:
                h = len(ix) // 2
                tr += list(ix[:h]); te += list(ix[h + 1:])  # one-segment gap: no shared samples
        tr, te = np.array(tr), np.array(te)
        classes, ytr = np.unique(rec[tr], return_inverse=True)
        keep = np.isin(rec[te], classes)
        te = te[keep]
        yte = np.searchsorted(classes, rec[te])
        k_cls = len(classes)

        def topk(scores):
            order = np.argsort(-scores, axis=1)
            return float(np.mean(order[:, 0] == yte)), float(np.mean((order[:, :5] == yte[:, None]).any(1)))

        def gbdt():
            m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3,
                                               early_stopping=False, random_state=42)
            return {"s": m.fit(seg["features"][tr], ytr).predict_proba(seg["features"][te])}

        def knn():
            xi = SimpleImputer(strategy="median").fit(seg["features"][tr])
            a, b = xi.transform(seg["features"][tr]), xi.transform(seg["features"][te])
            mu, sd = a.mean(0), a.std(0) + 1e-9
            m = KNeighborsClassifier(n_neighbors=5).fit((a - mu) / sd, ytr)
            return {"s": m.predict_proba((b - mu) / sd)}

        def cnn():
            t04.seed_everything(0)
            model = t04.make_model("M2", 2)
            model.classifier = nn.Linear(64, k_cls)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            xt = torch.from_numpy(np.ascontiguousarray(seg["tensor"][tr]))
            yt = torch.from_numpy(ytr.astype(np.int64))
            g = torch.Generator().manual_seed(0)
            for _ in range(40):
                model.train()
                for b in torch.randperm(len(yt), generator=g).split(64):
                    opt.zero_grad(set_to_none=True)
                    nn.functional.cross_entropy(model(xt[b]), yt[b]).backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                s = torch.cat([model(torch.from_numpy(np.ascontiguousarray(seg["tensor"][te][i:i + 256])))
                               for i in range(0, len(te), 256)])
            return {"s": s.numpy()}

        for name, fn in (("GBDT_features", gbdt), ("kNN_features", knn), ("CNN_signal", cnn)):
            s = unit("e8a", f"{variant}_{name}", fn)["s"]
            t1, t5 = topk(s)
            rows.append({"variant": variant, "probe": name, "n_recordings": k_cls, "n_test_segments": len(te),
                         "top1": t1, "top5": t5, "chance_top1": 1 / k_cls, "chance_top5": 5 / k_cls})
    write(pd.DataFrame(rows), "e8a_summary.csv")


# ---------------------------------------------------------------- E7 richer published-style features
RICH = ("fhr_mean", "fhr_sd", "fhr_iqr", "pow_vlf", "pow_lf", "pow_mf", "pow_hf", "ratio_lf_mfhf",
        "sampen", "dfa_alpha", "lzc")


def sampen(x, m=2, rf=0.2):
    r = rf * np.std(x)
    if r == 0 or len(x) < m + 2:
        return np.nan

    def count(mm):
        t = sliding_window_view(x, mm)[: len(x) - m]
        c = 0
        for i in range(len(t) - 1):
            c += int(np.sum(np.max(np.abs(t[i + 1:] - t[i]), axis=1) <= r))
        return c
    a, b = count(m + 1), count(m)
    return -np.log(a / b) if a > 0 and b > 0 else np.nan


def dfa(x, scales=(4, 8, 16, 32, 64)):
    y = np.cumsum(x - x.mean())
    fl = []
    for n in scales:
        k = len(y) // n
        if k < 4:
            return np.nan
        seg = y[: k * n].reshape(k, n)
        t = np.arange(n)
        coef = np.polyfit(t, seg.T, 1)
        fit = np.outer(coef[0], t) + coef[1][:, None]
        fl.append(np.sqrt(np.mean((seg - fit) ** 2)))
    return float(np.polyfit(np.log(scales), np.log(fl), 1)[0])


def lzc(bits):
    s, n = "".join(map(str, bits)), len(bits)
    i, c, l = 0, 1, 1
    k, k_max = 1, 1
    while True:  # Kaspar-Schuster LZ76
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c * np.log2(n) / n


def rich(fhr):
    v = np.isfinite(fhr)
    if v.sum() < 60 * FS:
        return [np.nan] * len(RICH)
    x = np.interp(np.arange(len(fhr)), np.flatnonzero(v), fhr[v])
    f, pxx = welch(x - x.mean(), fs=FS, nperseg=min(1024, len(x)), detrend="linear")
    band = lambda lo, hi: float(np.trapezoid(pxx[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]))
    vlf, lf, mf, hf = band(0, .03), band(.03, .15), band(.15, .5), band(.5, 1.01)
    x1 = x[: len(x) // FS * FS].reshape(-1, FS).mean(1)
    q = np.percentile(fhr[v], [25, 75])
    return [fhr[v].mean(), fhr[v].std(), q[1] - q[0], vlf, lf, mf, hf, lf / (mf + hf) if mf + hf > 0 else np.nan,
            sampen(x1), dfa(x1), lzc((np.diff(x1) > 0).astype(int))]


def rich_inputs(data, seg):
    path = OUT / "inputs_rich.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return z["rec"], z["seg"]
    rec, segs = [], {}
    for rid in data["ids"]:
        fhr, uc, end, _ = raw_trace(rid)
        rec.append(rich(fhr[max(0, end - 30 * 60 * FS):end]))
        kept, _ = t05.stage1_segments(fhr[:end], uc[:end])  # same segments as the T05 cache
        for a, b, _ in kept:
            segs[f"{rid}__{a:08d}"] = rich(fhr[a:b])
    assert set(segs) == set(seg["row_id"]), "rebuilt segments differ from T05 cache"
    xr = np.asarray(rec, float)
    xs = np.asarray([segs[k] for k in seg["row_id"]], float)
    np.savez_compressed(path, rec=xr, seg=xs)
    return xr, xs


def e7(data, seg):
    xr, xs = rich_inputs(data, seg)
    X = np.c_[data["features"], xr]
    XS = np.c_[seg["features"], xs]
    y, ids = data["y"], data["ids"]
    sid, srec, sy = seg["row_id"], seg["record_id"], seg["y"].astype(int)
    rows, s_oof = [], {}
    # S, 10 repetitions
    for r in range(10):
        p = np.zeros(len(y))
        for f in range(5):
            tr, te, inner = s_fold(data, r, f)
            p[te] = unit("e7", f"S_r{r}_f{f}", lambda: {"p": m1(X[tr], y[tr], inner, X[te], ids[tr], ids[te])})["p"]
        s_oof[r] = p
        rows.append({"protocol": "S", "repeat": r, "AUROC": auc(y, p)})
    # S_seg and L1, repetitions 0-2
    look = {rid: np.flatnonzero(srec == rid) for rid in ids}
    for r in range(3):
        ps, pl = {}, {}
        for f in range(5):
            tr, te, inner = s_fold(data, r, f)
            tri = np.concatenate([look[ids[i]] for i in tr])
            tei = np.concatenate([look[ids[i]] for i in te])
            inn = np.concatenate([np.full(len(look[ids[i]]), k) for i, k in zip(tr, inner)])
            pp = unit("e7", f"Sseg_r{r}_f{f}", lambda: {"p": m1(XS[tri], sy[tri], inn, XS[tei], sid[tri], sid[tei])})["p"]
            ps.update(zip(sid[tei], pp))
            a, b = t05.make_folds(sy, r, 2026)[f]
            inn = inner_folds(sy[a], 2026 + 100 + 10 * r + f)
            pp = unit("e7", f"L1_r{r}_f{f}", lambda: {"p": m1(XS[a], sy[a], inn, XS[b])})["p"]
            pl.update(zip(sid[b], pp))
        s_ = np.array([ps[k] for k in sid])
        l_ = np.array([pl[k] for k in sid])
        rows.append({"protocol": "S_seg", "repeat": r, "AUROC": auc(sy, s_)})
        rows.append({"protocol": "L1", "repeat": r, "AUROC": auc(sy, l_), "delta": auc(sy, l_) - auc(sy, s_)})
    # L2: SMOTE (as T05) and duplication (as E2), repetitions 0-2
    xi = SimpleImputer(strategy="median").fit_transform(X)
    xsm, ysm, _ = t05.smote_all(xi, y, 5, 42)
    rows_d, yd = dup_rows(y)
    for r in range(3):
        s_p = s_oof[r]
        p = np.zeros(len(ysm))
        for f in range(5):
            a, b = t05.make_folds(ysm, r, 2026)[f]
            inn = inner_folds(ysm[a], 2026 + 100 + 10 * r + f)
            p[b] = unit("e7", f"L2smote_r{r}_f{f}", lambda: {"p": m1(xsm[a], ysm[a], inn, xsm[b])})["p"]
        rows.append({"protocol": "L2_SMOTE", "repeat": r, "AUROC": auc(y, p[:len(y)]),
                     "delta": auc(y, p[:len(y)]) - auc(y, s_p)})
        p = np.zeros(len(yd))
        for f in range(5):
            te, pp = l2_dup_m1(xi, rows_d, yd, r, f, "L2dup", "e7")
            p[te] = pp
        rows.append({"protocol": "L2_dup", "repeat": r, "AUROC": auc(y, p[:len(y)]),
                     "delta": auc(y, p[:len(y)]) - auc(y, s_p)})
    # L3: features and parameters selected once on all 518 outcomes (as T05)
    aucs = [auc(y, xi[:, j]) for j in range(xi.shape[1])]
    chosen = sorted(range(len(aucs)), key=lambda j: (-aucs[j], j))[:5]
    best, best_auc = None, -np.inf
    for params in t03.grid("M1", T03):
        v = []
        for a, b in t05.make_folds(y, 0, 42):
            est = t03.make_model("M1", params).fit(xi[a][:, chosen], y[a])
            v.append(auc(y[b], est.predict_proba(xi[b][:, chosen])[:, 1]))
        if np.mean(v) > best_auc:
            best, best_auc = params, np.mean(v)
    for r in range(3):
        p = np.zeros(len(y))
        for f in range(5):
            tr, te, _ = s_fold(data, r, f)
            p[te] = unit("e7", f"L3_r{r}_f{f}", lambda: {"p": t03.make_model("M1", best).fit(
                X[tr][:, chosen], y[tr]).predict_proba(X[te][:, chosen])[:, 1]})["p"]
        rows.append({"protocol": "L3", "repeat": r, "AUROC": auc(y, p), "delta": auc(y, p) - auc(y, s_oof[r])})
    write(pd.DataFrame(rows), "e7_summary.csv")


# ---------------------------------------------------------------- E3 permutation controls
def perm_y(y, k):
    return np.random.default_rng(1000 + k).permutation(y)


def e3(data, seg):
    y, ids, x, t = data["y"], data["ids"], data["features"], data["tensor"]
    rows = []
    # (a) strict protocol, repetition 0, 100 permutations, M1 and M2 seed 0
    for k in range(N_PERM_S):
        yk = perm_y(y, k)
        p1, p2 = np.zeros(len(y)), np.zeros(len(y))
        for f in range(5):
            tr, te, _ = s_fold(data, 0, f)
            inner = inner_folds(yk[tr], 2126 + f)  # restratified on the permuted labels
            p1[te] = unit("e3", f"S_M1_k{k}_f{f}", lambda: {"p": m1(x[tr], yk[tr], inner, x[te], ids[tr], ids[te])})["p"]
            p2[te] = unit("e3", f"S_M2_k{k}_f{f}", lambda: {"p": m2_record(t, yk, ids, tr, te, 0, f)})["p"]
        rows += [{"control": "S", "model": "M1", "perm": k, "AUROC": auc(yk, p1)},
                 {"control": "S", "model": "M2", "perm": k, "AUROC": auc(yk, p2)}]
    # (b) oversampling before splitting with permuted labels, repetition 0
    xi = SimpleImputer(strategy="median").fit_transform(x)
    for k in range(N_PERM_L2):
        yk = perm_y(y, k)
        xs, ys, _ = t05.smote_all(xi, yk, 5, 42)
        rows_d, yd = dup_rows(yk)
        ps, pd_, pn = np.zeros(len(ys)), np.zeros(len(yd)), np.zeros(len(yd))
        for f in range(5):
            a, b = t05.make_folds(ys, 0, 2026)[f]
            inn = inner_folds(ys[a], 2126 + f)
            ps[b] = unit("e3", f"L2smote_M1_k{k}_f{f}", lambda: {"p": m1(xs[a], ys[a], inn, xs[b])})["p"]
            te, pp = l2_dup_m1(xi, rows_d, yd, 0, f, f"L2dup_M1_k{k}", "e3")
            pd_[te] = pp
            a, b = list(KFold(5, shuffle=True, random_state=2026).split(yd))[f]
            pn[b] = unit("e3", f"L2dup_M2_k{k}_f{f}", lambda: {"p": net("M2", t[rows_d][a], yd[a], t[rows_d][b], 0, f)})["p"]
        n = len(y)
        for name, p, yy in (("M1_SMOTE", ps, ys), ("M1_dup", pd_, yd), ("M2_dup", pn, yd)):
            rows.append({"control": "L2", "model": name, "perm": k, "AUROC": auc(yy, p),
                         "AUROC_real_rows": auc(yy[:n], p[:n])})
    # (c) segment-level splitting with permuted recording labels, repetition 0
    srec = seg["record_id"]
    for k in range(N_PERM_L1):
        lab = dict(zip(ids, perm_y(y, k)))
        sy = np.array([lab[r] for r in srec])
        p1, p2 = np.zeros(len(sy)), np.zeros(len(sy))
        for f in range(5):
            a, b = t05.make_folds(sy, 0, 2026)[f]
            inn = inner_folds(sy[a], 2126 + f)
            p1[b] = unit("e3", f"L1_M1_k{k}_f{f}", lambda: {"p": m1(seg["features"][a], sy[a], inn, seg["features"][b])})["p"]
            p2[b] = unit("e3", f"L1_M2_k{k}_f{f}", lambda: {"p": net("M2", seg["tensor"][a], sy[a], seg["tensor"][b], 0, f)})["p"]
        rows += [{"control": "L1", "model": "M1", "perm": k, "AUROC": auc(sy, p1)},
                 {"control": "L1", "model": "M2", "perm": k, "AUROC": auc(sy, p2)}]
    write(pd.DataFrame(rows), "e3_perms.csv")


# ---------------------------------------------------------------- E9 dose-response
def seg_inputs(data, minutes, step):
    path = OUT / f"inputs_seg_{minutes}_{step}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    cfg2 = t05.t02_cfg(P)
    lab = dict(zip(data["ids"], data["y"]))
    rid, feat, ten = [], [], []
    for r in data["ids"]:
        fhr, uc, end, _ = raw_trace(r)
        for a, b, _ in t05.stage1_segments(fhr[:end], uc[:end], minutes=minutes, step_minutes=step)[0]:
            f = compute_features(fhr[a:b], uc[a:b], cfg2)
            rid.append(r)
            feat.append([f[n] for n in t03.FEATURES])
            ten.append(build_dl_1hz(fhr[a:b], uc[a:b], minutes, cfg2))
    out = {"record_id": np.asarray(rid), "y": np.array([lab[r] for r in rid]),
           "features": np.asarray(feat, float), "tensor": t04.fixed_normalize(np.stack(ten))}
    np.savez_compressed(path, **out)
    return out


def dose_point(tag, S, records):
    """S_seg vs L1 on repetition 0 for M1 and M2 seed 0, restricted to `records`."""
    keep = np.isin(S["record_id"], records)
    rec, sy = S["record_id"][keep], S["y"][keep].astype(int)
    xf, xt = S["features"][keep], S["tensor"][keep]
    ry = pd.Series(sy, index=rec).groupby(level=0).first()
    rows = []
    preds = {}
    for proto in ("S_seg", "L1"):
        for model in ("M1", "M2"):
            p = np.zeros(len(sy))
            for f in range(5):
                if proto == "S_seg":
                    rtr, rte = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(ry.index, ry.values))[f]
                    a, b = np.flatnonzero(np.isin(rec, ry.index[rtr])), np.flatnonzero(np.isin(rec, ry.index[rte]))
                    rin = pd.Series(inner_folds(ry.values[rtr], 2126 + f), index=ry.index[rtr])
                    inn, groups = rin.loc[rec[a]].to_numpy(), rec[a]
                else:
                    a, b = t05.make_folds(sy, 0, 2026)[f]
                    inn, groups = inner_folds(sy[a], 2126 + f), None
                fn = (lambda: {"p": m1(xf[a], sy[a], inn, xf[b])}) if model == "M1" else \
                     (lambda: {"p": net("M2", xt[a], sy[a], xt[b], 0, f, groups=groups)})
                p[b] = unit("e9", f"{tag}_{proto}_{model}_f{f}", fn)["p"]
            preds[(proto, model)] = p
    for model in ("M1", "M2"):
        s, l = preds[("S_seg", model)], preds[("L1", model)]
        df = pd.DataFrame({"r": rec, "y": sy, "s": s, "l": l}).groupby("r").mean()
        rows.append({"setting": tag, "model": model, "n_segments": len(sy), "n_recordings": len(ry),
                     "n_pos_recordings": int(ry.sum()), "AUROC_S_seg": auc(sy, s), "AUROC_L1": auc(sy, l),
                     "delta_segment": auc(sy, l) - auc(sy, s),
                     "delta_recording": auc(df.y.to_numpy(int), df.l.to_numpy()) - auc(df.y.to_numpy(int), df.s.to_numpy())})
    return rows


def e9(data, seg):
    rows = []
    for tag, minutes, step in (("len10_ov50", 10, 5), ("len10_ov0", 10, 10), ("len10_ov75", 10, 2.5),
                               ("len5_ov50", 5, 2.5), ("len20_ov50", 20, 10)):
        rows += dose_point(tag, seg_inputs(data, minutes, step), data["ids"])
    base = seg_inputs(data, 10, 5)
    for frac in (0.25, 0.5, 0.75):
        sub, _ = next(StratifiedShuffleSplit(1, train_size=frac, random_state=42).split(data["ids"], data["y"]))
        rows += dose_point(f"sub{int(frac * 100)}", base, data["ids"][sub])
    # oversampling factor (duplication before splitting), repetition 0
    y, t = data["y"], data["tensor"]
    xi = SimpleImputer(strategy="median").fit_transform(data["features"])
    for factor in (1, 2, 5, 10, None):
        tag = f"dup{factor or 'bal'}"
        rws, yd = dup_rows(y, factor)
        p1, p2 = np.zeros(len(yd)), np.zeros(len(yd))
        for f in range(5):
            te, pp = l2_dup_m1(xi, rws, yd, 0, f, f"{tag}_M1", "e9")
            p1[te] = pp
            a, b = list(KFold(5, shuffle=True, random_state=2026).split(yd))[f]
            p2[b] = unit("e9", f"{tag}_M2_f{f}", lambda: {"p": net("M2", t[rws][a], yd[a], t[rws][b], 0, f)})["p"]
        n = len(y)
        for model, p in (("M1", p1), ("M2", p2)):
            rows.append({"setting": tag, "model": model, "n_segments": len(yd),
                         "AUROC_L1": np.nan, "AUROC_L2_real": auc(y, p[:n]), "AUROC_L2_all": auc(yd, p)})
    write(pd.DataFrame(rows), "e9_summary.csv")


# ---------------------------------------------------------------- E1b L1 for M3
def e1b(data, seg):
    get, _ = t05.contexts(CFG, data, seg, None)
    l1 = pd.read_parquet(P / "outputs/t05_leakage/oof_L1.parquet")
    ss = pd.read_parquet(P / "outputs/t05_leakage/oof_S_seg.parquet")
    sid, srec, sy = seg["row_id"], seg["record_id"], seg["y"].astype(int)
    res = {}
    for proto in ("S_seg", "L1"):
        for r in range(3):
            p = pd.Series(0.0, index=sid)
            for f in range(5):
                c = get(proto, "M2", 0, r, f)
                p.loc[c["row_id"]] = unit("e1b", f"{proto}_M3_r{r}_f{f}", lambda: {"p": net(
                    "M3", c["x_train"], c["y_train"], c["x_test"], r, f, groups=c["train_record_ids"])})["p"]
            res[(proto, "M3", r)] = p.loc[sid].to_numpy()
            fr = l1 if proto == "L1" else ss
            g = fr[(fr.model == "M1") & (fr.repeat == r)].set_index("row_id")
            res[(proto, "M1", r)] = g.loc[sid, "p"].to_numpy()
    rows = []
    for level in ("segment", "recording"):
        if level == "segment":
            yy, grp = sy, srec
            get_p = lambda k: res[k]
        else:
            frame = pd.DataFrame({k: v for k, v in res.items()}, index=srec)
            agg = frame.groupby(level=0).mean()
            yy = pd.Series(sy, index=srec).groupby(level=0).first().loc[agg.index].to_numpy()
            grp = None
            get_p = lambda k: agg[k].to_numpy()

        def delta(m, ix):
            return np.mean([auc(yy[ix], get_p(("L1", m, r))[ix]) - auc(yy[ix], get_p(("S_seg", m, r))[ix]) for r in range(3)])
        all_ = np.arange(len(yy))
        for m in ("M1", "M3"):
            rows.append({"level": level, "quantity": f"Delta_{m}", "estimate": delta(m, all_),
                         **boot(len(yy), lambda ix: delta(m, ix), grp)})
        rows.append({"level": level, "quantity": "Delta_M3 - Delta_M1",
                     "estimate": delta("M3", all_) - delta("M1", all_),
                     **boot(len(yy), lambda ix: delta("M3", ix) - delta("M1", ix), grp)})
        for r in range(3):
            rows.append({"level": level, "quantity": f"M3_r{r}", "AUROC_S_seg": auc(yy, get_p(("S_seg", "M3", r))),
                         "AUROC_L1": auc(yy, get_p(("L1", "M3", r)))})
    write(pd.DataFrame(rows), "e1b_summary.csv")


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    data, seg = load()
    for step in argv or STEPS:
        print(f"=== {step} ===", flush=True)
        if step == "e10a":
            run_window(data, "last30rec")
        elif step == "e4d":
            run_window(data, "shift10", nets=False)
        else:
            globals()[step](data, seg)
    print("T11B_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

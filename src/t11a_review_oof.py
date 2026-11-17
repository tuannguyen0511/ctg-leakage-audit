"""P21-T11a: post hoc analyses requested by the 2026-11-14 review (E0, E1, E4a-c,
E5, E6, E8b, L4 Youden J, Table 1 extension, E10b count).

Reads frozen outputs and derived tables only; writes a NEW folder
outputs/t11a_review and refuses to overwrite it. No model is retrained.
Usage (from D:/Tuan/AI_training):  python -m P21_CTG_Acidemia.src.t11a_review_oof
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import fisher_exact, rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from common.io_wfdb import read_record
from P21_CTG_Acidemia.src.ctg_features import clean_signals
from P21_CTG_Acidemia.src import t05_leakage as t05

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "P21_CTG_Acidemia"
DER = ROOT / "data_proc" / "ctu-uhb-ctgdb" / "1.0.0"
RAW = ROOT / "data_raw" / "ctu-uhb-ctgdb" / "1.0.0"
OUT = P / "outputs" / "t11a_review"
B, SEED = 10000, 42
THR = np.round(np.arange(0.02, 0.2001, 0.01), 2)


# ---------- small helpers ----------
def auc(y, p):
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def boot_p(d):
    d = np.asarray(d)
    return float(min(1.0, max(1 / len(d), 2 * min(np.mean(d <= 0), np.mean(d >= 0)))))


def summ(v):
    v = np.asarray(v, float)
    return {"mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else 0.0,
            "p2.5": np.quantile(v, .025), "p97.5": np.quantile(v, .975), "n_units": len(v)}


def draws(n, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p, kind="stable")
    out, m = np.empty_like(p), 0.0
    for k, i in enumerate(o):
        m = max(m, min(1.0, p[i] * (len(p) - k)))
        out[i] = m
    return out


def save(df, name):
    df.to_csv(OUT / name, index=False)


# ---------- load frozen S OOF as unit matrices (units x 518 records) ----------
def load_units():
    t3 = pd.read_parquet(P / "outputs/t03_baselines/A_primary/oof.parquet")
    t4 = pd.read_parquet(P / "outputs/t04_dl/A_primary/oof.parquet")
    for f in (t3, t4):
        f["record_id"] = f.record_id.astype(str)
    ids = np.sort(t3.record_id.unique())
    assert len(ids) == 518
    y = t3.drop_duplicates("record_id").set_index("record_id").loc[ids, "y"].to_numpy(int)
    assert y.sum() == 30
    units, fold = {}, {}
    for frame, cols in ((t3, ["p_M0", "p_M1"]),
                        (t4, [c for c in t4.columns if c.startswith("p_M")])):
        for col in cols:
            for r, g in frame.groupby("repeat"):
                g = g.set_index("record_id").loc[ids]
                model, _, s = col[2:].partition("_s")
                units.setdefault(model, []).append(
                    {"repeat": int(r), "seed": int(s or 0), "p": g[col].to_numpy(float)})
                fold[int(r)] = g.outer_fold.to_numpy(int)
    return ids, y, units, fold


# ---------- E5 + E0: single-model estimand and H1 on it ----------
def e5(ids, y, units):
    rows = []
    for m, us in units.items():
        a = [auc(y, u["p"]) for u in us]
        pr = [average_precision_score(y, u["p"]) for u in us]
        br = [brier(y, u["p"]) for u in us]
        ens = np.mean([u["p"] for u in us], axis=0)
        for metric, v, e in (("AUROC", a, auc(y, ens)),
                             ("AUPRC", pr, average_precision_score(y, ens)),
                             ("Brier", br, brier(y, ens))):
            rows.append({"model": m, "metric": metric, **summ(v), "ensemble": e})
    save(pd.DataFrame(rows), "E5_single_model_metrics.csv")

    # H1 on the single-model estimand: theta = mean over units of AUROC.
    idx = draws(len(y))
    boot = {}
    for m in ("M1", "M2", "M3", "M2UC"):
        P_ = np.stack([u["p"] for u in units[m]])
        boot[m] = np.array([np.mean([auc(y[i], p[i]) for p in P_]) for i in idx])
    out = []
    pairs = [("M2", "M1"), ("M3", "M1")]
    for a_, b_ in pairs:
        point = (np.mean([auc(y, u["p"]) for u in units[a_]]) -
                 np.mean([auc(y, u["p"]) for u in units[b_]]))
        d = boot[a_] - boot[b_]
        per_seed = []
        for s in (0, 1, 2):
            per_seed.append(np.mean([auc(y, u["p"]) for u in units[a_] if u["seed"] == s]))
        m1 = {u["repeat"]: auc(y, u["p"]) for u in units[b_]}
        rep_delta = [np.mean([auc(y, u["p"]) for u in units[a_] if u["repeat"] == r]) - m1[r]
                     for r in sorted(m1)]
        out.append({"comparison": f"{a_}-{b_}", "delta": point,
                    "ci95_low": np.quantile(d, .025), "ci95_high": np.quantile(d, .975),
                    "ci90_low": np.quantile(d, .05), "ci90_high": np.quantile(d, .95),
                    "p_raw": boot_p(d), "seed_auroc_spread": max(per_seed) - min(per_seed),
                    "seed_deltas": json.dumps([s - np.mean(list(m1.values())) for s in per_seed]),
                    "rep_delta_mean": np.mean(rep_delta), "rep_delta_sd": np.std(rep_delta, ddof=1),
                    "rep_delta_min": min(rep_delta), "rep_delta_max": max(rep_delta)})
    h1 = pd.DataFrame(out)
    h1["p_holm"] = holm(h1.p_raw)
    h1["gate_holm"] = h1.p_holm < .05
    h1["gate_spread"] = h1.delta.abs() > h1.seed_auroc_spread
    h1["gate_sign"] = [all(np.sign(json.loads(s)) == np.sign(d)) for s, d in zip(h1.seed_deltas, h1.delta)]
    h1["tost"] = np.where((h1.ci90_low > -.05) & (h1.ci90_high < .05), "equivalent",
                          np.where((h1.ci90_low > .05) | (h1.ci90_high < -.05), "non-equivalent", "inconclusive"))
    save(h1, "E5_H1_single_model.csv")
    # E0: M2+UC vs M2 seed 0, paired per repetition (descriptive).
    m2s0 = {u["repeat"]: u["p"] for u in units["M2"] if u["seed"] == 0}
    d = [auc(y, u["p"]) - auc(y, m2s0[u["repeat"]]) for u in units["M2UC"]]
    save(pd.DataFrame([{"comparison": "M2UC_s0 - M2_s0", **summ(d)}]), "E0_M2UC_vs_M2.csv")
    return h1


# ---------- E1 + per-unit H2 deltas ----------
def seg_frames():
    f = {k: pd.read_parquet(P / f"outputs/t05_leakage/oof_{k}.parquet") for k in ("L1", "S_seg", "L2", "L3")}
    for v in f.values():
        v["record_id"] = v.record_id.astype(str)
    return f


def cluster_index(rec):
    uniq, inv = np.unique(rec, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(SEED)
    return [np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))]) for _ in range(B)]


def e1(f):
    l1, ss = f["L1"], f["S_seg"]
    rows = sorted(l1.row_id.unique())
    rec = l1.drop_duplicates("row_id").set_index("row_id").loc[rows, "record_id"].to_numpy()
    y = l1.drop_duplicates("row_id").set_index("row_id").loc[rows, "y"].to_numpy(int)
    cidx = cluster_index(rec)

    def mat(frame, model, seed=None):
        g = frame[frame.model == model]
        if seed is not None:
            g = g[g.seed == seed]
        return {(r, s): gg.set_index("row_id").loc[rows, "p"].to_numpy(float)
                for (r, s), gg in g.groupby(["repeat", "seed"])}

    res = []
    variants = {"ensemble (H2 estimand)": None, "single model": "unit", "single model, M2 seed 0": "unit0"}
    for name, kind in variants.items():
        deltas, boots = {}, {}
        for m in ("M1", "M2"):
            seed = 0 if kind == "unit0" and m == "M2" else None
            L, S = mat(l1, m, seed), mat(ss, m, seed)
            if kind is None:
                pl, ps = [np.mean(list(L.values()), 0)], [np.mean(list(S.values()), 0)]
            else:
                pl, ps = [L[k] for k in sorted(L)], [S[k] for k in sorted(S)]
            for level in ("segment", "recording"):
                if level == "recording":
                    df = pd.DataFrame({"rec": rec, "y": y})
                    agg = lambda p: df.assign(p=p).groupby("rec").agg(y=("y", "first"), p=("p", "mean"))
                    pl_, ps_ = [agg(p).p.to_numpy() for p in pl], [agg(p).p.to_numpy() for p in ps]
                    yy = agg(pl[0]).y.to_numpy(int)
                    ii = draws(len(yy))
                else:
                    pl_, ps_, yy, ii = pl, ps, y, cidx
                pt = np.mean([auc(yy, a) - auc(yy, b) for a, b in zip(pl_, ps_)])
                bt = np.array([np.mean([auc(yy[i], a[i]) - auc(yy[i], b[i]) for a, b in zip(pl_, ps_)])
                               for i in ii])
                deltas[(m, level)], boots[(m, level)] = pt, bt
                res.append({"variant": name, "level": level, "quantity": f"Delta_{m}", "estimate": pt,
                            "ci95_low": np.quantile(bt, .025), "ci95_high": np.quantile(bt, .975),
                            "p_raw": boot_p(bt)})
        for level in ("segment", "recording"):
            d = boots[("M2", level)] - boots[("M1", level)]
            res.append({"variant": name, "level": level, "quantity": "Delta_M2 - Delta_M1",
                         "estimate": deltas[("M2", level)] - deltas[("M1", level)],
                         "ci95_low": np.quantile(d, .025), "ci95_high": np.quantile(d, .975),
                         "p_raw": boot_p(d)})
    save(pd.DataFrame(res), "E1_L1_interaction.csv")


def h2_units(f, ids, y, units):
    """Per-unit (single-model) H2 deltas, descriptive mean +/- SD across units."""
    out = []
    S = {(m, u["repeat"], u["seed"]): u["p"] for m in ("M1", "M2") for u in units[m]}
    for prot in ("L1", "L2", "L3"):
        for m in ("M1", "M2"):
            g = f[prot][f[prot].model == m]
            d = []
            for (r, s), gg in g.groupby(["repeat", "seed"]):
                if prot == "L1":
                    ref = f["S_seg"]
                    ref = ref[(ref.model == m) & (ref.repeat == r) & (ref.seed == s)].set_index("row_id")
                    gg = gg.set_index("row_id").loc[ref.index]
                    d.append(auc(gg.y.to_numpy(int), gg.p.to_numpy()) - auc(ref.y.to_numpy(int), ref.p.to_numpy()))
                else:
                    gg = gg[gg.row_id.str.startswith("real::")] if prot == "L2" else gg
                    gg = gg.set_index("record_id").loc[ids]
                    d.append(auc(y, gg.p.to_numpy()) - auc(y, S[(m, r, s)]))
            out.append({"protocol": prot, "model": m, **summ(d)})
    save(pd.DataFrame(out), "E5_H2_single_model_deltas.csv")


# ---------- L4 with Youden J ----------
def l4_j():
    f = pd.read_parquet(P / "outputs/t05_leakage/oof_L4.parquet")
    rows = []
    for m, g in f.groupby("model"):
        agg = g.groupby("record_id").agg(y=("y", "first"), s=("pred_S", "mean"), l=("pred_L", "mean"))
        yy, s, l = agg.y.to_numpy(int), agg.s.to_numpy(), agg.l.to_numpy()

        def j(ix, v):
            return v[ix][yy[ix] == 1].mean() + (1 - v[ix][yy[ix] == 0]).mean() - 1
        all_ = np.arange(len(yy))
        bt = np.array([j(i, l) - j(i, s) for i in draws(len(yy))])
        fold = g.groupby(["repeat", "outer_fold"]).apply(
            lambda x: pd.Series({"jS": x.loc[x.y == 1, "pred_S"].mean() - x.loc[x.y == 0, "pred_S"].mean(),
                                 "jL": x.loc[x.y == 1, "pred_L"].mean() - x.loc[x.y == 0, "pred_L"].mean()}))
        rows.append({"model": m, "J_S_record": j(all_, s), "J_L_record": j(all_, l),
                     "delta_J": j(all_, l) - j(all_, s), "ci95_low": np.quantile(bt, .025),
                     "ci95_high": np.quantile(bt, .975), "p_raw": boot_p(bt),
                     "J_S_foldmean": fold.jS.mean(), "J_L_foldmean": fold.jL.mean()})
    save(pd.DataFrame(rows), "L4_youdenJ.csv")


# ---------- E6: cross-fitted logistic recalibration, Brier, DCA ----------
def cal_int_slope(y, p):
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    slope = LogisticRegression(C=np.inf).fit(lp[:, None], y).coef_[0, 0]
    inter = brentq(lambda a: np.sum(y - expit(a + lp)), -20, 20)
    return inter, slope


def net_benefit(y, p, t):
    n = len(y)
    tp = np.sum((p >= t) & (y == 1))
    fp = np.sum((p >= t) & (y == 0))
    return tp / n - fp / n * t / (1 - t)


def e6(y, units, fold):
    per, dca = [], []
    prev = y.mean()
    b_null = prev * (1 - prev)
    for m in ("M0", "M1", "M2", "M3"):
        for u in units[m]:
            p, f = u["p"], fold[u["repeat"]]
            q = np.empty_like(p)
            for k in range(5):
                tr, te = f != k, f == k
                lr = LogisticRegression(C=np.inf).fit(logit(np.clip(p[tr], 1e-6, 1 - 1e-6))[:, None], y[tr])
                q[te] = lr.predict_proba(logit(np.clip(p[te], 1e-6, 1 - 1e-6))[:, None])[:, 1]
            for kind, v in (("raw", p), ("recalibrated", q)):
                ci, cs = cal_int_slope(y, v)
                per.append({"model": m, "repeat": u["repeat"], "seed": u["seed"], "kind": kind,
                            "AUROC": auc(y, v), "Brier": brier(y, v),
                            "scaled_Brier": 1 - brier(y, v) / b_null, "cal_intercept": ci,
                            "cal_slope": cs})
                for t in THR:
                    dca.append({"model": m, "kind": kind, "repeat": u["repeat"], "seed": u["seed"],
                                "threshold": t, "net_benefit": net_benefit(y, v, t)})
    per = pd.DataFrame(per)
    save(per, "E6_recal_units.csv")
    s = per.groupby(["model", "kind"]).agg(["mean", "std"]).drop(columns=["repeat", "seed"])
    s.columns = ["_".join(c) for c in s.columns]
    s.reset_index().to_csv(OUT / "E6_recal_summary.csv", index=False)
    dca = pd.DataFrame(dca).groupby(["model", "kind", "threshold"]).net_benefit.agg(["mean", "std"]).reset_index()
    for t in THR:
        dca.loc[len(dca)] = ["treat_all", "-", t, prev - (1 - prev) * t / (1 - t), 0.0]
    save(dca, "E6_dca.csv")
    save(pd.DataFrame([{"prevalence": prev, "brier_null": b_null}]), "E6_null.csv")


# ---------- E4: exclusion x delivery mode; shifted windows ----------
def mh_or(tables):
    """Mantel-Haenszel OR with Robins-Breslow-Greenland 95% CI. tables: [[a,b],[c,d]]."""
    R = S = PR = PS_QR = QS = 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        if n == 0:
            continue
        r, s = a * d / n, b * c / n
        p_, q_ = (a + d) / n, (b + c) / n
        R += r; S += s; PR += p_ * r; PS_QR += p_ * s + q_ * r; QS += q_ * s
    or_ = R / S
    se = np.sqrt(PR / (2 * R ** 2) + PS_QR / (2 * R * S) + QS / (2 * S ** 2))
    return or_, or_ * np.exp(-1.96 * se), or_ * np.exp(1.96 * se)


def two_by_two(excl, y):
    # rows: acidaemic / not; cols: excluded / kept
    return [[int(np.sum(excl & (y == 1))), int(np.sum(~excl & (y == 1)))],
            [int(np.sum(excl & (y == 0))), int(np.sum(~excl & (y == 0)))]]


def shifted_nan(cfg):
    rows = []
    for name in [l.strip() for l in (RAW / "RECORDS").read_text(encoding="utf-8-sig").splitlines() if l.strip()]:
        rec = read_record(RAW, name)
        ch = {c.upper(): i for i, c in enumerate(rec.channel_names)}
        fhr, _ = clean_signals(rec.signal[:, ch["FHR"]], rec.signal[:, ch["UC"]], cfg)
        pos = rec.clinical.get("Pos. II.st.", np.nan)
        try:
            pos = float(pos)
        except (TypeError, ValueError):
            pos = np.nan
        no2 = not (np.isfinite(pos) and pos >= 0)
        end0 = len(fhr) if no2 else int(pos)
        out = {"record_id": name, "no_stage2": no2, "rec_minutes": len(fhr) / 240}
        for shift in (0, 5, 10):
            end = end0 - (shift * 240 if no2 else 0)
            w = fhr[max(0, end - 7200):end]
            out[f"nan_pct_shift{shift}"] = 100 * np.mean(~np.isfinite(w)) if len(w) else 100.0
        w = fhr[max(0, len(fhr) - 7200):]  # last 30 min of the recording (E10a window)
        out["nan_pct_last30_recording"] = 100 * np.mean(~np.isfinite(w))
        rows.append(out)
    return pd.DataFrame(rows)


def e4():
    lab = pd.read_parquet(DER / "labels.parquet")
    lab["record_id"] = lab.record_id.astype(str)
    cl = pd.read_parquet(DER / "clinical.parquet").rename(columns={"record_name": "record_id"})
    cl["record_id"] = cl.record_id.astype(str)
    d = lab.merge(cl, on="record_id")
    cfg = t05.t02_cfg(P)
    sh = shifted_nan(cfg)
    d = d.merge(sh, on="record_id", suffixes=("", "_t11"))
    assert np.allclose(d.nan_pct_shift0, d.fhr_nan_pct_w30), "shift-0 must reproduce T02"
    sh.to_csv(OUT / "E4_shifted_window_nan.csv", index=False)  # local only: record-level
    y = d.y705.to_numpy(int)
    rows = []
    for rule, excl in (("original", d.excluded.to_numpy(bool)),
                       ("shift5", (d.nan_pct_shift5 > 50).to_numpy()),
                       ("shift10", (d.nan_pct_shift10 > 50).to_numpy()),
                       ("last30_recording", (d.nan_pct_last30_recording > 50).to_numpy())):
        crude = two_by_two(excl, y)
        orc, pc = fisher_exact(crude)
        strata = {}
        for col in ("Deliv. type", "no_stage2"):
            tabs = []
            for lev, g in d.groupby(col):
                m = d[col].eq(lev).to_numpy()
                t = two_by_two(excl[m], y[m])
                tabs.append(t)
                o, p = fisher_exact(t)
                rows.append({"rule": rule, "stratifier": col, "level": lev, "exc_pos": t[0][0], "kept_pos": t[0][1],
                             "exc_neg": t[1][0], "kept_neg": t[1][1], "OR": o, "fisher_p": p})
            strata[col] = mh_or(tabs)
        rows.append({"rule": rule, "stratifier": "none", "level": "all", "exc_pos": crude[0][0],
                     "kept_pos": crude[0][1], "exc_neg": crude[1][0], "kept_neg": crude[1][1],
                     "OR": orc, "fisher_p": pc,
                     "MH_OR_delivery": strata["Deliv. type"][0],
                     "MH_CI_delivery": json.dumps(strata["Deliv. type"][1:]),
                     "MH_OR_stage2": strata["no_stage2"][0],
                     "MH_CI_stage2": json.dumps(strata["no_stage2"][1:])})
    save(pd.DataFrame(rows), "E4_exclusion_strata.csv")
    # cross-tab exclusion x delivery x stage-2 x outcome (aggregate counts)
    ct = d.groupby(["excluded", "Deliv. type", "no_stage2", "y705"]).size().rename("n").reset_index()
    save(ct, "E4_crosstab.csv")
    return d


# ---------- Table 1 extension + E10b ----------
def table1(d):
    d = d.copy()
    d["group"] = np.where(d.excluded, "excluded", np.where(d.y705 == 1, "acidaemic", "non-acidaemic"))
    num = ["Age", "Parity", "Gest. weeks", "Weight(g)", "Apgar5", "I.stage", "II.stage", "BDecf", "pH"]
    rows = []
    for g, x in d.groupby("group"):
        r = {"group": g, "n": len(x)}
        for c in num:
            v = pd.to_numeric(x[c], errors="coerce")
            v = v[v >= 0] if c in ("II.stage", "I.stage") else v
            r[c] = f"{v.median():.1f} ({v.quantile(.25):.1f}-{v.quantile(.75):.1f})"
        r["caesarean_n"] = int((x["Deliv. type"] == 2).sum())
        r["no_stage2_n"] = int(x.no_stage2.sum())
        r["BDecf>=12_n"] = int((pd.to_numeric(x.BDecf, errors="coerce") >= 12).sum())
        rows.append(r)
    save(pd.DataFrame(rows), "T1_extended.csv")
    e = d[~d.excluded]
    bd = pd.to_numeric(e.BDecf, errors="coerce")
    save(pd.DataFrame([{"n_eligible": len(e), "pH<7.05": int(e.y705.sum()),
                        "pH<7.05_and_BDecf>=12": int(((e.y705 == 1) & (bd >= 12)).sum()),
                        "BDecf_missing_in_pos": int(bd[e.y705 == 1].isna().sum())}]), "E10b_counts.csv")


# ---------- E10b: frozen single-model predictions against pH<7.05 with BDecf>=12
def e10b(ids, y, units):
    cl = pd.read_parquet(DER / "clinical.parquet")
    cl["record_name"] = cl.record_name.astype(str)
    bd = pd.to_numeric(cl.set_index("record_name").loc[ids, "BDecf"], errors="coerce").to_numpy()
    keep = (y == 0) | (bd >= 12)  # acidaemic records with BDecf <12 or missing are left out
    ym = ((y == 1) & (bd >= 12)).astype(int)[keep]
    rows = [{"model": m, "n": int(keep.sum()), "events": int(ym.sum()),
             **summ([auc(ym, u["p"][keep]) for u in units[m]])} for m in ("M0", "M1", "M2", "M3")]
    save(pd.DataFrame(rows), "E10b_metabolic_auroc.csv")


# ---------- E8b: what the test fold shares with training under L2 ----------
def e8b(ids, y):
    lab = pd.read_parquet(DER / "labels.parquet")
    lab["record_id"] = lab.record_id.astype(str)
    feats = pd.read_parquet(DER / "features_w30.parquet")
    feats["record_id"] = feats.record_id.astype(str)
    x = SimpleImputer(strategy="median").fit_transform(
        feats.set_index("record_id").loc[ids, list(t05.t03.FEATURES)].to_numpy(float))
    x = (x - x.mean(0)) / x.std(0)
    xs, ys, n_new = t05.smote_all(x, y, 5, 42)
    rows_dup, ydup, _ = t05.random_oversample(np.arange(len(y)), y, 42)
    out = []
    for r in (0, 1, 2):
        # duplication (M2 design): share of positive test rows with an identical copy in training
        tr_te = list(KFold(5, shuffle=True, random_state=2026 + r).split(ydup))
        share = []
        for tr, te in tr_te:
            src_tr = set(rows_dup[tr])
            pos_te = te[ydup[te] == 1]
            share.append(np.mean([rows_dup[i] in src_tr for i in pos_te]))
        out.append({"design": "duplication", "repeat": r, "quantity": "pos_test_rows_with_copy_in_train",
                    "value": float(np.mean(share))})
        # SMOTE (M1 design): NN distance of positive test rows to nearest training positive
        d_l2 = []
        for tr, te in t05.make_folds(ys, r, 2026):
            nn = NearestNeighbors(n_neighbors=1).fit(xs[tr][ys[tr] == 1])
            d_l2.append(nn.kneighbors(xs[te][ys[te] == 1])[0].ravel())
        d_s = []
        for tr, te in t05.make_folds(y, r, 2026):
            nn = NearestNeighbors(n_neighbors=1).fit(x[tr][y[tr] == 1])
            d_s.append(nn.kneighbors(x[te][y[te] == 1])[0].ravel())
        out.append({"design": "SMOTE", "repeat": r, "quantity": "median_NN_dist_pos_test_to_train_pos",
                    "value": float(np.median(np.concatenate(d_l2)))})
        out.append({"design": "strict", "repeat": r, "quantity": "median_NN_dist_pos_test_to_train_pos",
                    "value": float(np.median(np.concatenate(d_s)))})
    save(pd.DataFrame(out), "E8b_L2_overlap.csv")


STEPS = ("e5", "h2u", "e1", "l4j", "e6", "e4", "e8b", "e10b")


def main(argv):
    """Run the steps named in argv (default: all); a step whose CSV exists is skipped."""
    OUT.mkdir(parents=True, exist_ok=True)
    done = {p.name for p in OUT.glob("*.csv")}
    marks = {"e5": "E5_H1_single_model.csv", "h2u": "E5_H2_single_model_deltas.csv",
             "e1": "E1_L1_interaction.csv", "l4j": "L4_youdenJ.csv", "e6": "E6_dca.csv",
             "e4": "T1_extended.csv", "e8b": "E8b_L2_overlap.csv", "e10b": "E10b_metabolic_auroc.csv"}
    ids, y, units, fold = load_units()
    for step in (argv or STEPS):
        if marks[step] in done:
            print("skip", step)
            continue
        if step == "e5":
            e5(ids, y, units)
        elif step == "h2u":
            h2_units(seg_frames(), ids, y, units)
        elif step == "e1":
            e1(seg_frames())
        elif step == "l4j":
            l4_j()
        elif step == "e6":
            e6(y, units, fold)
        elif step == "e4":
            table1(e4())
        elif step == "e8b":
            e8b(ids, y)
        elif step == "e10b":
            e10b(ids, y, units)
        print("done", step, flush=True)
    with open(OUT / "OUTPUT_SHA256.txt", "w") as fh:
        for p in sorted(OUT.glob("*.csv")):
            fh.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
        src = Path(__file__)
        fh.write(f"{hashlib.sha256(src.read_bytes()).hexdigest()}  {src.name}\n")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

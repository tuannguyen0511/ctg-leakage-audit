"""P21-T10c: figures for manuscript v0.4 (review 2026-11-14). Presentation only.
Reads frozen T05/T10b/T11a/T11b outputs and the U1 audit; writes outputs/t10c_figures.
Run from D:/Tuan/AI_training:  python -m P21_CTG_Acidemia.src.t10c_figures
Figures: F1 flow, F2 forest (+L4 Youden J), F3 ROC/calibration/DCA, F4 mechanisms, F8 literature.
"""
import hashlib
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PathCollection, LineCollection
from PIL import Image
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from P21_CTG_Acidemia.src import t11a_review_oof as A

P = Path(__file__).resolve().parents[1]
OUT = P / "outputs" / "t10c_figures"
OUT.mkdir(parents=True, exist_ok=True)
T11A, T11B = P / "outputs/t11a_review", P / "outputs/t11b_review"

plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False, "lines.solid_capstyle": "butt"})
W1, W2 = 3.5, 7.16
REF = dict(color="#999999", lw=0.8, ls=(0, (1, 1.5)), zorder=0)
MOD = {"M0": ("Logistic (features)", "#000000", "o", "-"),
       "M1": ("GBDT (features)", "#0072B2", "s", (0, (4, 2))),
       "M2": ("CNN", "#E69F00", "^", (0, (1, 1.5))),
       "M3": ("ResNet", "#009E73", "D", (0, (4, 1.5, 1, 1.5)))}
CHECKS = {}


# ---------------------------------------------------------------- helpers (as T10b)
def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=8, ha="right", va="bottom")


def ci(ax, x, lo, hi, y, color, ls="-", horizontal=True, cap=0.08):
    kw = dict(color=color, lw=1.0, ls=ls, clip_on=False, zorder=2)
    if horizontal:
        ax.plot([lo, hi], [y, y], **kw)
        for v in (lo, hi):
            ax.plot([v, v], [y - cap, y + cap], color=color, lw=0.9, clip_on=False, zorder=2)
    else:
        ax.plot([x, x], [lo, hi], **kw)
        for v in (lo, hi):
            ax.plot([x - cap, x + cap], [v, v], color=color, lw=0.9, clip_on=False, zorder=2)


def mk(ax, x, y, color, marker, fill="full", ms=5):
    return ax.plot([x], [y], ls="none", marker=marker, ms=ms, mfc=color if fill == "full" else "white",
                   mec=color, mew=0.9, clip_on=False, zorder=4)[0]


def _pts(a, ax, r):
    pts, rad = [], 0.0
    if isinstance(a, Line2D):
        xy = a.get_transform().transform(np.column_stack([a.get_xdata(orig=False), a.get_ydata(orig=False)]).astype(float))
        if a.get_linestyle() not in ("None", "none", "", " ") and len(xy) > 1:
            for p, q in zip(xy[:-1], xy[1:]):
                pts.append(np.linspace(p, q, int(max(2, np.hypot(*(q - p)) // 0.7 + 2))))
        if a.get_marker() not in (None, "None", "none", "", " "):
            pts.append(xy); rad = a.get_markersize() / 72 * a.figure.dpi / 2
        elif not pts:
            pts.append(xy)
    elif isinstance(a, Rectangle):
        bb = a.get_window_extent(r)
        gx, gy = np.meshgrid(np.linspace(bb.x0, bb.x1, 12), np.linspace(bb.y0, bb.y1, 12))
        pts.append(np.column_stack([gx.ravel(), gy.ravel()]))
    elif isinstance(a, PathCollection):
        pts.append(a.get_offset_transform().transform(a.get_offsets()))
        rad = np.sqrt(np.max(a.get_sizes())) / 72 * a.figure.dpi / 2 if len(a.get_sizes()) else 0
    elif isinstance(a, LineCollection):
        for seg in a.get_segments():
            seg = a.get_transform().transform(np.asarray(seg))
            for p, q in zip(seg[:-1], seg[1:]):
                pts.append(np.linspace(p, q, int(np.hypot(*(q - p)) // 0.7 + 2)))
    return (np.vstack(pts) if pts else np.zeros((0, 2))), rad


def qa(fig, name, axes, skip_patch=False):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    tt = [(a.get_text()[:20], b.get_text()[:20]) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
          if ba.x0 < bb.x1 - 1 and bb.x0 < ba.x1 - 1 and ba.y0 < bb.y1 - 1 and bb.y0 < ba.y1 - 1]
    td = set()
    for ax in axes:
        for a in [a for a in ax.lines + ax.collections + ax.patches if a.get_visible()]:
            if skip_patch and isinstance(a, Rectangle):
                continue
            Q, rad = _pts(a, ax, r)
            if a.get_clip_on() and len(Q):
                pb = ax.get_window_extent(r)
                Q = Q[(Q[:, 0] >= pb.x0) & (Q[:, 0] <= pb.x1) & (Q[:, 1] >= pb.y0) & (Q[:, 1] <= pb.y1)]
            for t, bb in texts:
                if len(Q) and ((Q[:, 0] > bb.x0 - rad + 1) & (Q[:, 0] < bb.x1 + rad - 1) &
                               (Q[:, 1] > bb.y0 - rad + 1) & (Q[:, 1] < bb.y1 + rad - 1)).any():
                    td.add(t.get_text()[:20])
    sizes = sorted({round(t.get_fontsize(), 2) for t, _ in texts})
    CHECKS[name] = {"NO_TEXT_OVERLAP": "PASS" if not tt and not td else f"FAIL tt={tt[:4]} td={sorted(td)[:6]}",
                    "FONT_RANGE": "PASS" if min(sizes) >= 7.0 and max(sizes) <= 8.5 else f"FAIL {sizes}"}


def save(fig, name, data):
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    dpi = 800 if fig.get_figwidth() < 4 else 600
    fig.savefig(OUT / f"{name}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.tif", dpi=dpi, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(OUT / f"{name}.png") as im:
        im.convert("L").save(OUT / f"{name}_gray.png"); CHECKS[name]["SIZE"] = f"{im.size[0]}x{im.size[1]}px@{dpi}"
    pd.DataFrame(data).to_csv(OUT / f"{name}_plot_data.csv", index=False)
    plt.close(fig)


# ================================================================ F1 flow
def f1():
    BOX = (("A", .25, .935, "CTU-UHB recordings\nn = 552 (pH < 7.05: 40)"),
           ("S", .77, .935, "Sensitivity analysis:\nno exclusion, n = 552"),
           ("E", .75, .745, "Excluded: > 50% FHR missing in last 30 min of stage 1\nn = 34 (acidaemic 10)\n31 without second stage"),
           ("B", .25, .60, "Primary cohort\nn = 518 (pH < 7.05: 30)"),
           ("C", .22, .37, "Strict protocol S\nrecording-level nested CV, 10 × 5 × 5, 4 models"),
           ("D", .76, .37, "Single-factor perturbations L1–L4\nGBDT and CNN (+ ResNet for L1)\nrepetitions 0–2"),
           ("G", .22, .10, "Label-permutation controls\nS ×100; L1 ×10; L2 ×20"),
           ("F", .76, .10, "L1: 10-min segments, 50% overlap, whole first stage\n5,224 retained; 499 dropped"))
    fig, ax = plt.subplots(figsize=(W1, 4.6)); ax.set_xlim(0, 1); ax.set_ylim(0.0, 1.02); ax.axis("off")
    T = {}
    for k, x, y, s in BOX:
        s = "\n".join(textwrap.fill(l, 21, break_on_hyphens=False) for l in s.split("\n"))
        T[k] = ax.text(x, y, s, ha="center", va="center", fontsize=7.5, linespacing=1.2,
                       bbox=dict(boxstyle="square,pad=0.4", fc="#F2F2F2", ec="#4D4D4D", lw=0.8, ls="--" if k in "SG" else "-"))
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    E = {k: inv.transform(t.get_bbox_patch().get_window_extent(r)) for k, t in T.items()}
    ed = dict(color="#4D4D4D", lw=0.8)

    def arrow(p, q, ls="-"):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", mutation_scale=7, shrinkA=0, shrinkB=0, ls=ls, **ed))
    arrow((.25, E["A"][0][1]), (.25, E["B"][1][1]))
    arrow((.25, .745), (E["E"][0][0], .745))
    arrow((E["A"][1][0], .935), (E["S"][0][0], .935), ls="--")
    yb = (E["B"][0][1] + max(E["C"][1][1], E["D"][1][1])) / 2
    ax.plot([.25, .25], [E["B"][0][1], yb], **ed); ax.plot([.22, .76], [yb, yb], **ed)
    arrow((.22, yb), (.22, E["C"][1][1])); arrow((.76, yb), (.76, E["D"][1][1]))
    arrow((.76, E["D"][0][1]), (.76, E["F"][1][1])); arrow((.22, E["C"][0][1]), (.22, E["G"][1][1]), ls="--")
    ov = [(a, b) for i, a in enumerate(E) for b in list(E)[i + 1:]
          if E[a][0][0] < E[b][1][0] and E[b][0][0] < E[a][1][0] and E[a][0][1] < E[b][1][1] and E[b][0][1] < E[a][1][1]]
    qa(fig, "F1", [ax], skip_patch=True)
    if ov:
        CHECKS["F1"]["NO_TEXT_OVERLAP"] = f"FAIL boxes {ov}"
    save(fig, "F1", [{"box": k, "text": s.replace("\n", " ")} for k, _, _, s in BOX])


# ================================================================ F2 forest + L4
def f2():
    d = pd.read_csv(P / "outputs/t10b_figures/F2_plot_data.csv")
    su = pd.read_csv(T11A / "E5_H2_single_model_deltas.csv").set_index(["protocol", "model"])
    j = pd.read_csv(T11A / "L4_youdenJ.csv").set_index("model")
    h2 = pd.read_csv(P / "outputs/t05_leakage/h2_tests.csv").set_index(["protocol", "model"])
    rows = [("L1", "M1", "Segment-level split", "L1 record/M1"), ("L1", "M2", "", "L1 record/M2"),
            ("L2", "M1", "Oversampling before split", "L2 augmented/M1"), ("L2", "M2", "", "L2 augmented/M2"),
            ("L3", "M1", "Selection on full/test data", None), ("L3", "M2", "", None)]
    fig = plt.figure(figsize=(W2, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.75, 0.95], wspace=0.12)
    ax, tx, aj = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    tx.axis("off"); tx.set_ylim(-0.6, 5.6); tx.set_xlim(0, 1)
    plot, yt, yl = [], [], []
    for i, (pr, m, grp, desc) in enumerate(rows):
        y = 5 - i
        r = d[d.label == f"{pr}/{m}"].iloc[0]
        ci(ax, None, r.S_low, r.S_high, y + 0.12, "#0072B2", cap=0.07)
        ci(ax, None, r.L_low, r.L_high, y - 0.12, "#D55E00", cap=0.07)
        mk(ax, r.S, y + 0.12, "#0072B2", "s", "full", 4.4); mk(ax, r.L, y - 0.12, "#D55E00", "o", "full", 4.8)
        if desc:
            q = d[d.label == desc].iloc[0]
            mk(ax, q.L, y - 0.36, "#7F7F7F", "^" if pr == "L1" else "D", "none", 4.6)
            plot.append({"row": desc, "L": q.L})
        p = "< 0.001" if r.p_holm < 0.001 else f"= {r.p_holm:.3f}"
        s = su.loc[(pr, m)]
        hc = h2.loc[(pr, m)]
        tx.text(0.02, y + 0.14, f"Δ {hc.delta:+.3f} ({hc.ci95_low:.3f}, {hc.ci95_high:.3f}), p {p}".replace("-", "−"),
                fontsize=7.5, va="center")
        tx.text(0.02, y - 0.2, f"single model {s['mean']:+.3f} ± {s['sd']:.3f}".replace("-", "−"), fontsize=7.5, va="center", color="#555555")
        yt.append(y); yl.append(f"{grp}\n{MOD[m][0].split(' (')[0]}" if grp else MOD[m][0].split(" (")[0])
        plot.append({"row": f"{pr}/{m}", "S": r.S, "S_low": r.S_low, "S_high": r.S_high, "L": r.L, "L_low": r.L_low,
                     "L_high": r.L_high, "delta": r.delta, "p_holm": r.p_holm, "single_mean": s["mean"], "single_sd": s["sd"]})
    for yy in (3.5, 1.5):
        ax.axhline(yy, color="#DDDDDD", lw=0.7, zorder=0)
    ax.axvline(0.5, **REF)
    ax.set_yticks(yt); ax.set_yticklabels(yl); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.40, 1.03); ax.set_ylim(-0.6, 5.6); ax.set_xlabel("AUROC (95% CI)")
    # L4 panel: change in Youden J
    for k, (m, y) in enumerate((("M1", 1.0), ("M2", 0.0))):
        q = j.loc[m]
        ci(aj, None, q.ci95_low, q.ci95_high, y, MOD[m][1], cap=0.06); mk(aj, q.delta_J, y, MOD[m][1], MOD[m][2], "full", 4.8)
        aj.text(q.ci95_high + 0.02, y, f"{q.J_S_record:.2f}→{q.J_L_record:.2f}", va="center", fontsize=7.5)
        plot.append({"row": f"L4/{m}", "J_S": q.J_S_record, "J_L": q.J_L_record, "delta": q.delta_J, "lo": q.ci95_low, "hi": q.ci95_high})
    aj.axvline(0, **REF); aj.set_xlim(-0.05, 0.72); aj.set_ylim(-0.6, 1.6)
    aj.set_yticks([1, 0]); aj.set_yticklabels(["GBDT", "CNN"]); aj.tick_params(axis="y", length=0)
    aj.set_xlabel("ΔYouden J (95% CI)"); aj.set_title("L4 threshold on test fold", fontsize=8, pad=8)
    panel_label(ax, "(a)"); aj.text(-0.42, 1.04, "(b)", transform=aj.transAxes, fontweight="bold", fontsize=8, ha="right", va="bottom")
    h = [Line2D([0], [0], ls="none", marker="s", ms=5, mfc="#0072B2", mec="#0072B2", label="Strict"),
         Line2D([0], [0], ls="none", marker="o", ms=5, mfc="#D55E00", mec="#D55E00", label="Lenient (tested)"),
         Line2D([0], [0], ls="none", marker="^", ms=5, mfc="white", mec="#7F7F7F", label="L1, recording-level (descriptive)"),
         Line2D([0], [0], ls="none", marker="D", ms=5, mfc="white", mec="#7F7F7F", label="L2, test set with copies (descriptive)")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.45, -0.07), columnspacing=1.2, handletextpad=0.3)
    qa(fig, "F2", [ax, aj])
    save(fig, "F2", plot)


# ================================================================ F3 ROC / calibration / DCA
def recal(p, f, y):
    q = np.empty_like(p)
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))[:, None]
    for k in range(5):
        lr = LogisticRegression(C=np.inf).fit(lp[f != k], y[f != k])
        q[f == k] = lr.predict_proba(lp[f == k])[:, 1]
    return q


def bins(y, p, n=10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return np.array([p[b == i].mean() for i in range(len(edges) - 1)]), np.array([y[b == i].mean() for i in range(len(edges) - 1)])


def f3():
    ids, y, units, fold = A.load_units()
    summ = pd.read_csv(T11A / "E5_single_model_metrics.csv").set_index(["model", "metric"])
    dca = pd.read_csv(T11A / "E6_dca.csv")
    fig = plt.figure(figsize=(W2, 5.7))
    gs = fig.add_gridspec(5, 2, height_ratios=[3, 0.7, 0.75, 3, 0.7], hspace=0.12, wspace=0.3)
    a, b, bh = fig.add_subplot(gs[0:2, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    c, ch, dd = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[3:5, 1])
    plot = []
    for m, (nm, col, mkr, ls) in MOD.items():
        Pm = np.stack([u["p"] for u in units[m]])
        fpr, tpr, _ = roc_curve(y, Pm.mean(0))
        a.plot(fpr, tpr, color=col, ls=ls, lw=1.2, drawstyle="steps-post")
        raw = np.concatenate(list(Pm)); yy = np.tile(y, len(Pm))
        rc = np.concatenate([recal(u["p"], fold[u["repeat"]], y) for u in units[m]])
        for axc, axh, v, tag in ((b, bh, raw, "raw"), (c, ch, rc, "recalibrated")):
            x_, o_ = bins(yy, v)
            axc.plot(x_, o_, color=col, ls=ls, lw=1.1, marker=mkr, ms=3.6, mfc="white", mec=col, mew=0.9)
            h_, e_ = np.histogram(v, bins=np.linspace(0, 1 if tag == "raw" else 0.3, 31), density=True)
            axh.step(e_[:-1], h_, where="post", color=col, ls=ls, lw=0.9)
            plot += [{"panel": tag, "model": m, "mean_pred": xi, "obs": oi} for xi, oi in zip(x_, o_)]
        g = dca[(dca.model == m) & (dca.kind == "recalibrated")].sort_values("threshold")
        dd.plot(g.threshold * 100, g["mean"], color=col, ls=ls, lw=1.2, marker=mkr, ms=3.4, mfc="white", mec=col, markevery=3)
        plot += [{"panel": "dca", "model": m, "threshold": t, "nb": v} for t, v in zip(g.threshold, g["mean"])]
    ta = dca[dca.model == "treat_all"].sort_values("threshold")
    dd.plot(ta.threshold * 100, ta["mean"], color="#777777", lw=1.0, label="Treat all")
    dd.axhline(0, color="#000000", lw=0.9, ls=(0, (1, 1.5)), label="Treat none")
    a.plot([0, 1], [0, 1], **REF); a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)
    a.set_xlabel("False-positive rate"); a.set_ylabel("True-positive rate")
    fig.legend(handles=[Line2D([0], [0], color=MOD[m][1], ls=MOD[m][3], lw=1.2, marker=MOD[m][2], ms=3.6, mfc="white",
                               mec=MOD[m][1], label=f"{MOD[m][0]}, AUROC {summ.loc[(m, 'AUROC')]['mean']:.3f} ± {summ.loc[(m, 'AUROC')]['sd']:.3f}")
                        for m in MOD], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), handlelength=2.8)
    for axc, axh, hi in ((b, bh, 0.85), (c, ch, 0.3)):
        axc.plot([0, hi], [0, hi], **REF); axc.set_xlim(0, hi); axc.set_ylim(0, hi)
        axc.set_ylabel("Observed proportion"); axc.tick_params(labelbottom=False)
        axh.set_xlim(0, hi); axh.set_yticks([]); axh.spines["left"].set_visible(False)
        axh.set_xlabel("Predicted probability")
    b.set_title("Before recalibration", fontsize=8); c.set_title("After cross-fitted recalibration", fontsize=8)
    lo = min(-0.02, ta["mean"].min())
    dd.set_ylim(-0.02, 0.05); dd.set_xlim(1.5, 20.5); dd.set_xlabel("Threshold probability (%)"); dd.set_ylabel("Net benefit")
    dd.legend(loc="upper right", handlelength=2.4)
    for axx, s in ((a, "(a)"), (b, "(b)"), (c, "(c)"), (dd, "(d)")):
        panel_label(axx, s)
    CHECKS["F3_note"] = {"treat_all_min": f"{lo:.3f} (clipped at -0.02 by ylim)"}
    qa(fig, "F3", [a, b, c, dd])
    save(fig, "F3", plot)


# ================================================================ F4 mechanisms
def f4():
    perm = pd.read_csv(T11B / "e3_perms.csv")
    obs = pd.read_csv(T11A / "E6_recal_units.csv")
    obs = obs[(obs.kind == "raw") & (obs.repeat == 0) & (obs.seed == 0)].set_index("model").AUROC
    e9 = pd.read_csv(T11B / "e9_summary.csv").set_index(["setting", "model"])
    e8c = pd.read_csv(T11B / "e8c_summary.csv").groupby("candidates").inflation.agg(["mean", "std"])
    fig = plt.figure(figsize=(W2, 5.0))
    gs = fig.add_gridspec(2, 12, hspace=0.6, wspace=3.2)
    a = fig.add_subplot(gs[0, 0:6]); b = fig.add_subplot(gs[0, 6:9]); c = fig.add_subplot(gs[0, 9:12])
    d_ = fig.add_subplot(gs[1, 0:4]); e = fig.add_subplot(gs[1, 4:8]); f = fig.add_subplot(gs[1, 8:12])
    plot = []
    R = [("S", "M1", "S, GBDT"), ("S", "M2", "S, CNN"), ("L1", "M1", "L1, GBDT"), ("L1", "M2", "L1, CNN"),
         ("L2", "M1_SMOTE", "L2, GBDT SMOTE"), ("L2", "M1_dup", "L2, GBDT dup."), ("L2", "M2_dup", "L2, CNN dup.")]
    rng = np.random.default_rng(0)
    for i, (ctl, m, lab) in enumerate(R):
        y = len(R) - 1 - i
        v = perm[(perm.control == ctl) & (perm.model == m)]
        v = (v.AUROC_real_rows if ctl == "L2" else v.AUROC).to_numpy()
        col = MOD["M1" if m.startswith("M1") else "M2"][1]
        a.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=5, color=col, alpha=0.45, lw=0, zorder=2)
        a.plot([v.mean()], [y], marker="|", ms=10, mew=1.6, color="black", zorder=3)
        if ctl == "S":
            a.plot([obs[m]], [y], ls="none", marker="*", ms=7, mfc="white", mec="black", mew=0.9, zorder=4)
        plot += [{"panel": "a", "row": lab, "auroc": x} for x in v]
    a.axvline(0.5, **REF); a.set_xlim(0.2, 1.02); a.set_ylim(-0.6, len(R) - 0.4)
    a.set_yticks(range(len(R))[::-1]); a.set_yticklabels([r[2] for r in R]); a.tick_params(axis="y", length=0)
    a.set_xlabel("AUROC with permuted labels")
    a.legend(handles=[Line2D([0], [0], ls="none", marker="|", ms=8, mew=1.6, color="black", label="Mean of permutations"),
                      Line2D([0], [0], ls="none", marker="*", ms=7, mfc="white", mec="black", label="Observed, true labels")],
             loc="lower left", ncol=1, handletextpad=0.2)

    def dose(ax, tags, xs, xlabel, col="delta_segment", ylabel="ΔAUROC (L1 − S$_{seg}$)", logx=False):
        for m in ("M1", "M2"):
            nm, cc, mkr, ls = MOD[m]
            v = [e9.loc[(t, m), col] for t in tags]
            ax.plot(xs, v, color=cc, ls=ls, lw=1.1, marker=mkr, ms=4, mfc="white", mec=cc, mew=0.9)
            plot.extend({"panel": xlabel, "model": m, "x": x, "y": yv} for x, yv in zip(xs, v))
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if logx:
            ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([str(x) if x < 16 else "16" for x in xs])
            ax.minorticks_off()
        else:
            ax.set_xticks(xs)
    dose(b, ["len10_ov0", "len10_ov50", "len10_ov75"], [0, 50, 75], "Segment overlap (%)")
    dose(c, ["len5_ov50", "len10_ov50", "len20_ov50"], [5, 10, 20], "Segment length (min)", ylabel="")
    dose(d_, ["sub25", "sub50", "sub75", "len10_ov50"], [25, 50, 75, 100], "Recordings used (%)")
    dose(e, ["dup1", "dup2", "dup5", "dup10", "dupbal"], [1, 2, 5, 10, 16.3], "Copies of each positive",
         col="AUROC_L2_real", ylabel="AUROC under L2", logx=True)
    for ax in (b, c, d_):
        ax.axhline(0, **REF); ax.set_ylim(-0.02, 0.42)
    e.set_ylim(0.45, 1.02); e.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); e.axhline(0.5, **REF)
    f.plot(e8c.index, e8c["mean"], color=MOD["M2"][1], ls=MOD["M2"][3], lw=1.1, marker="^", ms=4, mfc="white", mec=MOD["M2"][1])
    for k, row in e8c.iterrows():
        ci(f, k, row["mean"] - row["std"], row["mean"] + row["std"], None, MOD["M2"][1], horizontal=False, cap=1.2)
    plot.extend({"panel": "f", "candidates": k, "mean": r_["mean"], "sd": r_["std"]} for k, r_ in e8c.iterrows())
    f.axhline(0, **REF); f.set_xlabel("Candidate checkpoints (epochs)"); f.set_ylabel("Inflation (ΔAUROC)")
    f.set_xticks([1, 10, 20, 30, 60]); f.set_xlim(-1, 62); f.set_ylim(-0.03, 0.11); f.set_yticks([0, 0.05, 0.10])
    for ax, s in zip((a, b, c, d_, e, f), "abcdef"):
        ax.text(-0.3 if ax is not a else -0.02, 1.06, f"({s})", transform=ax.transAxes, fontweight="bold", fontsize=8, ha="right", va="bottom")
    fig.legend(handles=[Line2D([0], [0], color=MOD[m][1], ls=MOD[m][3], marker=MOD[m][2], ms=4, mfc="white", mec=MOD[m][1],
                               label=MOD[m][0]) for m in ("M1", "M2")], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    qa(fig, "F4", [b, c, d_, e, f])
    save(fig, "F4", plot)


# ================================================================ F8 literature
def f8():
    u = pd.read_csv(P / "reports/lit/U1_extraction.csv", dtype=str)
    u = u[(u.included_main == "yes") & u.ctu_auroc_for_summary.notna()].copy()
    u["auroc"] = u.ctu_auroc_for_summary.astype(float)

    def thr(s):
        v = [float(x) for x in re.findall(r"7\.\d+", s)]
        return "pH ≤ 7.05" if v and min(v) <= 7.05 and "7.20" not in s[:20] else ("pH 7.15" if v and min(v) <= 7.15 else "pH 7.20 or composite")
    u["thr"] = [thr(s) for s in u.outcome_definition.fillna("")]
    SH = {"pH ≤ 7.05": "o", "pH 7.15": "s", "pH 7.20 or composite": "D"}
    G = [("within_CTU_segment_or_image", "Segment/image\nsplit", "#D55E00"),
         ("within_CTU_record", "Recording\nsplit", "#0072B2"),
         ("external_test_on_CTU", "External\ntest", "#009E73")]
    fig, ax = plt.subplots(figsize=(W1, 3.6)); plot = []
    ax.axhspan(0.572, 0.691, color="#E5E5E5", zorder=0)
    ax.text(2.95, 0.585, "This study, strict\n(single model)", ha="right", va="bottom", fontsize=7.5, color="#444444")
    for i, (g, lab, col) in enumerate(G):
        s = u[u.evaluation_design == g].sort_values("auroc")
        ax.plot([i - 0.12, i - 0.12], [s.auroc.min(), s.auroc.max()], color=col, lw=1.2, zorder=1)
        ly = s.auroc.to_numpy().copy()
        for k in range(1, len(ly)):  # spread labels at least 0.026 apart, then keep inside the axis
            ly[k] = max(ly[k], ly[k - 1] + 0.026)
        ly -= max(0, ly[-1] - 1.0)
        for (_, r), yl in zip(s.iterrows(), ly):
            ax.plot([i - 0.12], [r.auroc], ls="none", marker=SH[r.thr], ms=4.6, mfc="white", mec=col, mew=1.0, zorder=3)
            name = r.first_author.split("/")[0].split(" ")[-1] if "M'Barek" not in r.first_author else "Ben M'Barek"
            name = {"S006": "Zhao (RP)", "S007": "Zhao (DeepFHR)"}.get(r.id, name)
            if abs(yl - r.auroc) > 0.006:
                ax.plot([i - 0.09, i - 0.04], [r.auroc, yl], color="#AAAAAA", lw=0.5, zorder=1)
            ax.text(i - 0.03, yl, f"{name} {r.year}", fontsize=7.0, va="center", ha="left")
            plot.append({"id": r.id, "design": g, "auroc": r.auroc, "threshold_group": r.thr})
        ax.text(i, 0.505, f"n = {len(s)}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([g[1] for g in G]); ax.set_xlim(-0.45, 3.0)
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("Reported AUROC on CTU-UHB")
    ax.legend(handles=[Line2D([0], [0], ls="none", marker=v, ms=4.6, mfc="white", mec="black", label=k) for k, v in SH.items()],
              loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, handletextpad=0.2, columnspacing=0.8)
    qa(fig, "F8", [ax], skip_patch=True)
    save(fig, "F8", plot)


if __name__ == "__main__":
    for fn in (f1, f2, f3, f4, f8):
        fn()
    with open(OUT / "OUTPUT_SHA256.txt", "w") as fh:
        for p in sorted(OUT.glob("*")):
            if p.name != "OUTPUT_SHA256.txt":
                fh.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
    for k, v in CHECKS.items():
        print(k, v)

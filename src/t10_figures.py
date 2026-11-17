"""P21-T10: redraw F1-F7 (+ F8 literature audit) to IEEE figure rules.
Presentation only: every plotted value is read from SHA-locked plot_data CSVs.
Usage: python t10_figures.py <input_root> <output_dir>
  input_root must contain outputs/t06d_figures/F{1..5}_plot_data.csv,
  outputs/t09a_synthesis/figures/F{6,7}_plot_data.csv, reports/lit/U1_extraction.csv
"""
import sys, hashlib, shutil, textwrap, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PathCollection, LineCollection, PolyCollection
from matplotlib.transforms import blended_transform_factory
from PIL import Image

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
INPUTS = {
    "F1": ("outputs/t06d_figures/F1_plot_data.csv", "b3d419e02028161074170b2cc898e2ee765c774adf23b4ed6a94b2d5bb0dc808"),
    "F2": ("outputs/t06d_figures/F2_plot_data.csv", "f1885f1d989dee9779d7f339c45d006747c54763b4b32c592302df60cf73f0ee"),
    "F3": ("outputs/t06d_figures/F3_plot_data.csv", "ffe9b97340ce288ad637c99424f550005c49124d67f84749c4f37ae1195a584d"),
    "F4": ("outputs/t06d_figures/F4_plot_data.csv", "3bfb1598ac152210c5d946a35942f0d3e377f88945feb17a82bbea099fcae141"),
    "F5": ("outputs/t06d_figures/F5_plot_data.csv", "07aded403befb69330cfe34f942d24d7a3f649a31878fbe202416f970e751a25"),
    "F6": ("outputs/t09a_synthesis/figures/F6_plot_data.csv", "7d7177968bc901b345d0886f992711c1d863d0aece761dfd744eea73a47e1d22"),
    "F7": ("outputs/t09a_synthesis/figures/F7_plot_data.csv", "0ce2d20f43c3595037796efb09082c85979383a07a49539bb22168a5faad74d4"),
}
U1 = "reports/lit/U1_extraction.csv"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for k, (rel, h) in INPUTS.items():
    if sha(ROOT / rel) != h: sys.exit(f"INPUT_SHA_MISMATCH:{rel}")
if OUT.exists() and any(OUT.iterdir()): sys.exit("OUTPUT_DIR_NOT_EMPTY")
OUT.mkdir(parents=True, exist_ok=True)
D = {k: pd.read_csv(ROOT / rel) for k, (rel, _) in INPUTS.items()}

plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False, "lines.solid_capstyle": "butt"})
FONT_USED = matplotlib.font_manager.findfont(matplotlib.font_manager.FontProperties(family=plt.rcParams["font.family"]))

W1, W2 = 3.5, 7.16
REF = dict(color="#999999", lw=0.8, ls=(0, (1, 1.5)), zorder=0)
MOD = {  # name, colour, marker, linestyle
    "M0": ("Logistic (features)", "#000000", "o", "-"),
    "M1": ("GBDT (features)", "#0072B2", "s", (0, (4, 2))),
    "M2": ("CNN", "#E69F00", "^", (0, (1, 1.5))),
    "M3": ("ResNet", "#009E73", "D", (0, (4, 1.5, 1, 1.5))),
}
GRP = {  # F2 encodings
    "strict": ("Strict (record-level split)", "#0072B2", "s", "full"),
    "leaky": ("Leaky (tested)", "#D55E00", "o", "full"),
    "record": ("Record-level AUROC under segment split (descriptive)", "#E69F00", "^", "none"),
    "synth": ("Test set containing duplicated samples (descriptive)", "#CC79A7", "D", "none"),
}
FEAT = {"orig": ("Original", "#0072B2", "s", "////"), "wmfb": ("WMFB", "#009E73", "o", "....")}
REPORT, CHECKS = [], {}

def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=8, ha="right", va="bottom")

def ci(ax, x, lo, hi, y, color, ls="-", horizontal=True, cap=0.08):
    """CI bar with caps; horizontal: x-range at y, else y-range at x."""
    kw = dict(color=color, lw=1.0, ls=ls, clip_on=False, zorder=2)
    if horizontal:
        ax.plot([lo, hi], [y, y], **kw)
        for v in (lo, hi): ax.plot([v, v], [y - cap, y + cap], color=color, lw=0.9, clip_on=False, zorder=2)
    else:
        ax.plot([x, x], [lo, hi], **kw)
        for v in (lo, hi): ax.plot([x - cap, x + cap], [v, v], color=color, lw=0.9, clip_on=False, zorder=2)

def mk(ax, x, y, color, marker, fill="full", ms=5, **kw):
    fc = color if fill == "full" else "white"
    return ax.plot([x], [y], ls="none", marker=marker, ms=ms, mfc=fc, mec=color, mew=0.9, clip_on=False, zorder=4, **kw)[0]

# ---------------------------------------------------------------- QA ----
def _pix_points(artist, ax, renderer):
    """Return (N,2) display coords sampled densely along a data artist, plus marker radius."""
    pts, rad = [], 0.0
    if isinstance(artist, Line2D):
        xy = artist.get_transform().transform(np.column_stack([artist.get_xdata(orig=False), artist.get_ydata(orig=False)]).astype(float))
        if artist.get_linestyle() not in ("None", "none", "", " ") and len(xy) > 1:
            for a, b in zip(xy[:-1], xy[1:]):
                n = int(max(2, np.hypot(*(b - a)) // 0.7 + 2)); pts.append(np.linspace(a, b, n))
        if artist.get_marker() not in (None, "None", "none", "", " "):
            pts.append(xy); rad = artist.get_markersize() / 72 * artist.figure.dpi / 2
        elif not pts: pts.append(xy)
    elif isinstance(artist, Rectangle):
        bb = artist.get_window_extent(renderer); x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
        gx, gy = np.meshgrid(np.linspace(x0, x1, 12), np.linspace(y0, y1, 12)); pts.append(np.column_stack([gx.ravel(), gy.ravel()]))
    elif isinstance(artist, LineCollection):
        for seg in artist.get_segments():
            seg = artist.get_transform().transform(np.asarray(seg))
            for a, b in zip(seg[:-1], seg[1:]): pts.append(np.linspace(a, b, int(np.hypot(*(b - a)) // 0.7 + 2)))
    elif isinstance(artist, PathCollection):
        off = artist.get_offset_transform().transform(artist.get_offsets()); pts.append(off)
        rad = np.sqrt(np.max(artist.get_sizes())) / 72 * artist.figure.dpi / 2 if len(artist.get_sizes()) else 0
    return (np.vstack(pts) if pts else np.zeros((0, 2))), rad

def qa(fig, name, data_axes, allow_text_in_patch=False, ci_list=(), extra_skip=()):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); res = {}
    texts = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip() or t in extra_skip: continue
        texts.append((t, t.get_window_extent(r)))
    # text-text overlap
    tt = [(a.get_text()[:25], b.get_text()[:25]) for i, (a, ba) in enumerate(texts) for (b, bb) in texts[i + 1:]
          if ba.x0 < bb.x1 - 1 and bb.x0 < ba.x1 - 1 and ba.y0 < bb.y1 - 1 and bb.y0 < ba.y1 - 1]
    # text-data overlap
    td = []
    for ax in data_axes:
        arts = [a for a in ax.lines + ax.collections + ax.patches if a.get_visible()]
        for a in arts:
            if allow_text_in_patch and isinstance(a, Rectangle): continue
            P, rad = _pix_points(a, ax, r)
            if not len(P): continue
            if a.get_clip_on():
                pb = ax.get_window_extent(r); P = P[(P[:, 0] >= pb.x0 - 0.5) & (P[:, 0] <= pb.x1 + 0.5) & (P[:, 1] >= pb.y0 - 0.5) & (P[:, 1] <= pb.y1 + 0.5)]
                if not len(P): continue
            for t, bb in texts:
                inside = (P[:, 0] > bb.x0 - rad + 1) & (P[:, 0] < bb.x1 + rad - 1) & (P[:, 1] > bb.y0 - rad + 1) & (P[:, 1] < bb.y1 + rad - 1)
                if inside.any(): td.append(t.get_text()[:25])
    res["NO_TEXT_OVERLAP"] = "PASS" if not tt and not td else f"FAIL tt={tt[:4]} td={sorted(set(td))[:6]}"
    # CI inside axes with >=2% margin
    bad = []
    for ax, axis, vals in ci_list:
        lo, hi = (ax.get_xlim() if axis == "x" else ax.get_ylim()); span = hi - lo
        v = np.asarray([x for x in vals if np.isfinite(x)])
        if len(v) and (v.min() < lo + 0.02 * span - 1e-12 or v.max() > hi - 0.02 * span + 1e-12): bad.append((axis, float(v.min()), float(v.max()), lo, hi))
    res["CI_INSIDE_AXES"] = "PASS" if not bad else f"FAIL {bad}"
    # legend outside data
    lb = []
    for leg in [a.get_legend() for a in fig.axes] + fig.legends:
        if leg is None: continue
        b = leg.get_window_extent(r)
        for ax in data_axes:
            p = ax.get_window_extent(r)
            if b.x0 < p.x1 - 1 and p.x0 < b.x1 - 1 and b.y0 < p.y1 - 1 and p.y0 < b.y1 - 1: lb.append(ax.get_label())
    res["LEGEND_OUTSIDE_DATA"] = "PASS" if not lb else f"FAIL {lb}"
    sizes = sorted({round(t.get_fontsize(), 2) for t, _ in texts})
    res["FONT_RANGE"] = "PASS" if min(sizes) >= 7.0 and max(sizes) <= 8.5 else f"FAIL {sizes}"
    res["SPINES"] = "PASS" if all(not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible() for ax in data_axes) else "FAIL"
    CHECKS[name] = res
    return res

def save(fig, name, plotted, csv_src=None):
    for ext in ("svg", "pdf"): fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    dpi = 800 if fig.get_figwidth() < 4 else 600  # one-column figures: keep PNG width > 2000 px
    fig.savefig(OUT / f"{name}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.tif", dpi=dpi, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    s = (OUT / f"{name}.svg").read_text(encoding="utf-8")
    s = re.sub(r"font-family:\s*'?Liberation Sans'?", "font-family: Arial, 'Liberation Sans', Helvetica, sans-serif", s)
    s = s.replace("font: ", "font: ").replace("'Liberation Sans'", "Arial, 'Liberation Sans', Helvetica, sans-serif") if "Arial," not in s else s
    (OUT / f"{name}.svg").write_text(s, encoding="utf-8")
    with Image.open(OUT / f"{name}.png") as im:
        im.convert("L").save(OUT / f"{name}_gray.png"); w, h = im.size
    if csv_src is not None: shutil.copyfile(csv_src, OUT / f"{name}_plot_data.csv")
    CHECKS[name]["VALUES_UNCHANGED"] = plotted
    CHECKS[name]["SIZE"] = f"{w}x{h}px @{dpi} dpi"
    plt.close(fig)

def same(a, b): return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-12, equal_nan=True)

# ================================================================ F1 ====
def f1():
    d = D["F1"]; lab = " ".join(d.label.astype(str)).replace(",", "")
    for v in ["552", "518", "30", "34", "10", "24", "5224", "499"]:
        if v not in lab: sys.exit(f"F1_COUNT_MISMATCH:{v}")
    BOX = (("A", .40, .92, "CTU-UHB\nrecordings\nn = 552"),
           ("B", .40, .62, "Analysis cohort\nn = 518 (pH < 7.05: 30)"),
           ("E", .75, .77, "Excluded: > 50% FHR signal loss in last 30 min of stage 1\nn = 34 (acidaemic 10; non-acidaemic 24)"),
           ("S", .78, .92, "Sensitivity analysis:\nno exclusion, n = 552\n(pH < 7.05: 40)"),
           ("C", .22, .30, "Strict evaluation\nrepeated nested CV\n(10 × 5 × 5)"),
           ("D", .65, .30, "Leakage audit\n10-min segments\n5,224 retained; 499 rejected"))
    fig, ax = plt.subplots(figsize=(W1, 5.0)); ax.set_xlim(0, 1); ax.set_ylim(0.18, 1.02); ax.axis("off")
    T = {}
    for k, x, y, s in BOX:
        s = "\n".join(textwrap.fill(line, 24, break_on_hyphens=False) for line in s.split("\n"))
        T[k] = ax.text(x, y, s, ha="center", va="center", fontsize=7.5, linespacing=1.3,
                       bbox=dict(boxstyle="square,pad=0.5", fc="#F2F2F2", ec="#4D4D4D", lw=0.8, ls="--" if k == "S" else "-"))
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    E = {k: inv.transform(t.get_bbox_patch().get_window_extent(r)) for k, t in T.items()}  # [[x0,y0],[x1,y1]]
    ed = dict(color="#4D4D4D", lw=0.8)
    def arrow(p, q, ls="-"):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", mutation_scale=7, shrinkA=0, shrinkB=0, ls=ls, **ed))
    xa = .40
    arrow((xa, E["A"][0][1]), (xa, E["B"][1][1]))
    ye = .77; ax.plot([xa, xa], [ye, ye], **ed); arrow((xa, ye), (E["E"][0][0], ye))
    arrow((E["A"][1][0], .92), (E["S"][0][0], .92), ls="--")
    yb = .45; ax.plot([xa, xa], [E["B"][0][1], yb], **ed); ax.plot([.22, .65], [yb, yb], **ed)
    arrow((.22, yb), (.22, E["C"][1][1])); arrow((.65, yb), (.65, E["D"][1][1]))
    # QA: boxes must not overlap each other; text overlap with connectors checked via boxes
    boxes = list(E.values()); ov = [(a, b) for i, a in enumerate(E) for b in list(E)[i + 1:]
                                     if E[a][0][0] < E[b][1][0] and E[b][0][0] < E[a][1][0] and E[a][0][1] < E[b][1][1] and E[b][0][1] < E[a][1][1]]
    res = qa(fig, "F1", [ax], allow_text_in_patch=True)
    # connector lines may touch box edges (endpoints), so test connectors against text glyph bboxes only
    if ov: CHECKS["F1"]["NO_TEXT_OVERLAP"] = f"FAIL boxes {ov}"
    CHECKS["F1"]["SPINES"] = "N/A (diagram)"
    save(fig, "F1", "PASS (counts matched F1_plot_data labels)", ROOT / INPUTS["F1"][0])

# ================================================================ F2 ====
def f2():
    d = D["F2"].copy()
    rows = [("L1/M1", "L1 record/M1", "Segment-level split\nGBDT (features)"),
            ("L1/M2", "L1 record/M2", "Segment-level split\nCNN"),
            ("L2/M1", "L2 augmented/M1", "Oversampling before split\nGBDT (features)"),
            ("L2/M2", "L2 augmented/M2", "Oversampling before split\nCNN"),
            ("L3/M1", None, "Tuning on full data\nGBDT (features)"),
            ("L3/M2", None, "Checkpoint selected on test\nCNN")]
    if sorted(sum([[a] + ([b] if b else []) for a, b, _ in rows], [])) != sorted(d.label): sys.exit("F2_ROWS_MISMATCH")
    fig = plt.figure(figsize=(W2, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.15], wspace=0.04)
    ax = fig.add_subplot(gs[0]); tx = fig.add_subplot(gs[1], sharey=ax); tx.axis("off")
    yt, labels, plotted, civals = [], [], {"S": [], "L": []}, []
    y = 0.0; ypos = {}
    for tested, desc, lab in rows[::-1]:
        ypos[tested] = y
        if desc: ypos[desc] = y + 0.34
        yt.append(y + (0.17 if desc else 0)); labels.append(lab)
        y += 1.25 if desc else 1.0
    for _, r in d.iterrows():
        yy = ypos[r.label]; desc = bool(r.descriptive) if not isinstance(r.descriptive, str) else r.descriptive == "True"
        g = "leaky" if not desc else ("synth" if "augmented" in r.label else "record")
        ax.plot([r.S, r.L], [yy, yy], color="#BBBBBB", lw=0.6, zorder=1)
        if np.isfinite(r.S_low): ci(ax, None, r.S_low, r.S_high, yy, GRP["strict"][1], cap=0.07); civals += [r.S_low, r.S_high]
        if np.isfinite(r.L_low): ci(ax, None, r.L_low, r.L_high, yy, GRP[g][1], ls="-" if g == "leaky" else (0, (3, 1.5)), cap=0.07); civals += [r.L_low, r.L_high]
        a = mk(ax, r.S, yy, GRP["strict"][1], "s", "full", 4.6); b = mk(ax, r.L, yy, GRP[g][1], GRP[g][2], GRP[g][3], 4.8 if g != "leaky" else 5)
        plotted["S"].append((a.get_xdata()[0], r.S)); plotted["L"].append((b.get_xdata()[0], r.L)); civals += [r.S, r.L]
        if desc:
            txt = f"Δ = {r.delta:.3f} (descriptive)" if np.isfinite(r.delta) else "descriptive"
        else:
            p = r.p_holm; ps = "< 0.001" if p < 0.001 else f"= {p:.3f}"
            txt = f"Δ = {r.delta:.3f}; Holm p {ps}"
        tx.text(0.03, yy, txt.replace("Δ = -", "Δ = −"), va="center", ha="left", fontsize=7.5, color="#222222" if not desc else "#555555")
    tops = sorted(ypos.values())
    for cut in (1.0 + 1.0 - 0.5, 2.0 + 2.5 - 0.5 - 0.0):
        pass
    sep = [ (ypos["L3/M1"] + ypos["L2/M2"]) / 2 + 0.0 , (ypos["L2/M1"] + 0.34 + ypos["L1/M2"]) / 2]
    for s_ in sep: ax.axhline(s_ - 0.0, color="#DDDDDD", lw=0.7, zorder=0)
    ax.axvline(0.5, **REF)
    ax.set_yticks(yt); ax.set_yticklabels(labels); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.40, 1.03); ax.set_xticks(np.arange(0.4, 1.01, 0.1)); ax.set_ylim(-0.45, y - 0.5)
    ax.set_xlabel("AUROC (95% CI)")
    h = [Line2D([0], [0], ls="none", marker=GRP[k][2], ms=5, mfc=GRP[k][1] if GRP[k][3] == "full" else "white", mec=GRP[k][1], mew=0.9, label=GRP[k][0]) for k in GRP]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.45, -0.075), columnspacing=1.5, handletextpad=0.4)
    qa(fig, "F2", [ax], ci_list=[(ax, "x", civals)])
    ok = all(same(a, b) for a, b in plotted["S"] + plotted["L"])
    save(fig, "F2", "PASS" if ok else "FAIL", ROOT / INPUTS["F2"][0])

# ================================================================ F3 ====
def f3():
    d = D["F3"]; fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 3.35)); fig.subplots_adjust(wspace=0.32, bottom=0.25)
    ok = True; civals = []
    for i, m in enumerate(["M0", "M1", "M2", "M3"]):
        nm, c, mkr, ls = MOD[m]
        roc = d[(d.panel == "ROC") & (d.model == m)]; x, y = roc.x.values, roc.y.values
        ln = a1.plot(x, y, color=c, ls=ls, lw=1.2, drawstyle="steps-post", zorder=3)[0]
        ok &= same(ln.get_xdata(), x) and same(ln.get_ydata(), y)
        idx = [int(np.argmax(x >= f + 0.03 * i)) for f in (0.12, 0.32, 0.52, 0.72)]
        a1.plot(x[idx], y[idx], ls="none", marker=mkr, ms=4, mfc="white", mec=c, mew=0.9, zorder=4)
        cal = d[(d.panel == "calibration") & (d.model == m)]
        l2 = a2.plot(cal.x, cal.y, color=c, ls=ls, lw=1.1, marker=mkr, ms=4, mfc="white", mec=c, mew=0.9, zorder=3)[0]
        ok &= same(l2.get_xdata(), cal.x) and same(l2.get_ydata(), cal.y); civals += list(cal.x) + list(cal.y)
    a1.plot([0, 1], [0, 1], **REF); a1.set_xlim(-0.02, 1.02); a1.set_ylim(-0.02, 1.02)
    a1.set_xlabel("False-positive rate"); a1.set_ylabel("True-positive rate"); a1.set_aspect("equal")
    hi = max(1.0, max(civals)) * 1.03
    a2.plot([0, hi], [0, hi], **REF); a2.set_xlim(-0.04 * hi, hi); a2.set_ylim(-0.04 * hi, hi); a2.set_aspect("equal")
    a2.set_xlabel("Mean predicted probability"); a2.set_ylabel("Observed proportion acidaemic")
    panel_label(a1, "(a)"); panel_label(a2, "(b)")
    h = [Line2D([0], [0], color=MOD[m][1], ls=MOD[m][3], lw=1.2, marker=MOD[m][2], ms=4, mfc="white", mec=MOD[m][1], label=MOD[m][0]) for m in MOD]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), handlelength=3.2)
    qa(fig, "F3", [a1, a2], ci_list=[(a2, "x", civals), (a2, "y", civals)])
    save(fig, "F3", "PASS" if ok else "FAIL", ROOT / INPUTS["F3"][0])

# ================================================================ F4 ====
def f4():
    d = D["F4"]; fig, ax = plt.subplots(figsize=(W1, 2.9)); ok = True
    sty = {"M1": MOD["M1"], "M3": MOD["M3"], "treat_all": ("Treat all", "#777777", None, "-"), "treat_none": ("Treat none", "#000000", None, (0, (1, 1.5)))}
    ymax = 0.02; ylo = 1.05 * d[d.strategy.isin(["M1", "M3"])].net_benefit.min()
    for k, (nm, c, mkr, ls) in sty.items():
        s = d[d.strategy == k].sort_values("threshold"); x = s.threshold.values * 100; y = s.net_benefit.values
        ln = ax.plot(x, y, color=c, ls=ls, lw=1.2 if mkr else 1.0, marker=mkr, ms=3.6, mfc="white", mec=c, mew=0.8,
                     markevery=(0 if k == "M1" else 2, 5) if mkr else None, zorder=3 if mkr else 2, label=nm)[0]
        ok &= same(ln.get_xdata(), x) and same(ln.get_ydata(), y)
    below = int((d[d.strategy == "treat_all"].net_benefit < ylo).sum())
    ax.set_ylim(ylo, ymax); ax.set_xlim(4, 31); ax.set_xlabel("Threshold probability (%)"); ax.set_ylabel("Net benefit")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, handlelength=2.6, columnspacing=1.0)
    qa(fig, "F4", [ax])
    CHECKS["F4"]["CI_INSIDE_AXES"] = f"N/A (no CI); treat_all points below ylim = {below}"
    save(fig, "F4", "PASS" if ok else "FAIL", ROOT / INPUTS["F4"][0]); REPORT.append(f"F4_treat_all_points_below_ylim={below} (ylim lower = 1.05 x min of M1/M3 curves)")

# ================================================================ F5 ====
def f5():
    d = D["F5"]; fig, ax = plt.subplots(figsize=(W1, 2.9)); fig.subplots_adjust(bottom=0.3, right=0.75); ok = True; civ = []
    for _, r in d.iterrows():
        nm, c, mkr, _ = MOD[r.model]
        ci(ax, r.x, r.ci95_low, r.ci95_high, None, c, horizontal=False, cap=0.04)
        p = mk(ax, r.x, r.auroc, c, mkr, "full", 5); ok &= same(p.get_ydata(), [r.auroc]); civ += [r.ci95_low, r.ci95_high]
    ax.axhline(0.5, **REF)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Low signal loss\n(≤ median)", "High signal loss\n(> median)"]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.2, 0.93); ax.set_ylabel("AUROC (95% CI)")
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for g, xg in (("le_median", 0), ("gt_median", 1)):
        s = d[d.group == g].iloc[0]; ax.text(xg, -0.30, f"n = {s.n}; positives: {s.positives}", transform=tr, ha="center", va="top", fontsize=7.5)
    h = [Line2D([0], [0], ls="none", marker=MOD[m][2], ms=5, mfc=MOD[m][1], mec=MOD[m][1], label=MOD[m][0]) for m in ("M1", "M3")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.02, 1.0), handletextpad=0.3)
    qa(fig, "F5", [ax], ci_list=[(ax, "y", civ)])
    save(fig, "F5", "PASS" if ok else "FAIL", ROOT / INPUTS["F5"][0])

# ================================================================ F6 ====
def f6():
    d = D["F6"]; prot = list(dict.fromkeys(d.protocol)); fig, ax = plt.subplots(figsize=(W1, 3.0)); ok = True; civ = []
    ypos = {p: len(prot) - 1 - i for i, p in enumerate(prot)}
    for _, r in d.iterrows():
        k = "orig" if r.features.startswith("Original") else "wmfb"; nm, c, mkr, _ = FEAT[k]
        yy = ypos[r.protocol] + (0.13 if k == "orig" else -0.13)
        ci(ax, None, r.lo, r.hi, yy, c, cap=0.06); p = mk(ax, r.delta, yy, c, mkr, "full", 4.8)
        ok &= same(p.get_xdata(), [r.delta]); civ += [r.lo, r.hi]
    ax.axvline(0, **REF)
    ax.set_yticks(list(ypos.values())); ax.set_yticklabels([textwrap.fill(p, 22) for p in ypos]); ax.tick_params(axis="y", length=0)
    lo, hi = min(civ), max(civ); sp = hi - lo; ax.set_xlim(lo - 0.06 * sp, hi + 0.06 * sp); ax.set_ylim(-0.6, len(prot) - 0.4)
    ax.set_xlabel("ΔAUROC, leaky − strict (95% CI)")
    h = [Line2D([0], [0], ls="none", marker=FEAT[k][2], ms=5, mfc=FEAT[k][1], mec=FEAT[k][1], label=f"{FEAT[k][0]} features") for k in FEAT]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2)
    qa(fig, "F6", [ax], ci_list=[(ax, "x", civ)])
    save(fig, "F6", "PASS" if ok else "FAIL", ROOT / INPUTS["F6"][0])

# ================================================================ F7 ====
def f7():
    d = D["F7"].set_index("extractor"); fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.8), gridspec_kw=dict(width_ratios=[1, 1.6]))
    fig.subplots_adjust(wspace=0.35, bottom=0.27); ok = True
    keys = [("Original extractor", "orig"), ("WMFB", "wmfb")]; bw = 0.36
    civ1 = []
    for i, (row, k) in enumerate(keys):
        nm, c, _, hatch = FEAT[k]; v = d.loc[row]
        b = a1.bar(i, v.MAE_bpm, width=0.55, color=c, alpha=0.85, edgecolor="black", lw=0.8, hatch=hatch, zorder=2)
        ok &= same(b.patches[0].get_height(), v.MAE_bpm)
        ci(a1, i, v.MAE_bpm_lo, v.MAE_bpm_hi, None, "black", horizontal=False, cap=0.07); civ1 += [v.MAE_bpm_lo, v.MAE_bpm_hi]
    a1.axhline(5, color="#999999", lw=0.8, ls=(0, (4, 2)), zorder=1)
    a1.set_xlim(-0.6, 2.0); a1.text(1.98, 5, "5 bpm", ha="right", va="bottom", fontsize=7.5, color="#555555")
    a1.set_ylim(0, max(civ1 + [5]) * 1.12); a1.set_xticks([0, 1]); a1.set_xticklabels(["Original", "WMFB"]); a1.set_ylabel("Baseline MAE (bpm)")
    civ2 = []
    for j, (met, lab) in enumerate([("Acc_F1", "Accelerations"), ("Dec_F1", "Decelerations")]):
        for i, (row, k) in enumerate(keys):
            nm, c, _, hatch = FEAT[k]; v = d.loc[row]; x = j + (i - 0.5) * bw
            b = a2.bar(x, v[met], width=bw * 0.92, color=c, alpha=0.85, edgecolor="black", lw=0.8, hatch=hatch, zorder=2)
            ok &= same(b.patches[0].get_height(), v[met])
            ci(a2, x, v[met + "_lo"], v[met + "_hi"], None, "black", horizontal=False, cap=0.05); civ2 += [v[met + "_lo"], v[met + "_hi"]]
    a2.axhline(0.60, color="#999999", lw=0.8, ls=(0, (4, 2)), zorder=1)
    a2.set_xlim(-0.5, 2.05); a2.text(2.03, 0.60, "0.60", ha="right", va="bottom", fontsize=7.5, color="#555555")
    a2.set_xticks([0, 1]); a2.set_xticklabels(["Accelerations", "Decelerations"]); a2.set_ylim(0, 1.05); a2.set_ylabel("F1 score")
    panel_label(a1, "(a)"); panel_label(a2, "(b)")
    h = [Patch(facecolor=FEAT[k][1], alpha=0.85, edgecolor="black", hatch=FEAT[k][3], label=("Original extractor" if k == "orig" else "WMFB")) for _, k in keys]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    qa(fig, "F7", [a1, a2], ci_list=[(a1, "y", civ1), (a2, "y", civ2)])
    save(fig, "F7", "PASS" if ok else "FAIL", ROOT / INPUTS["F7"][0])

# ================================================================ F8 ====
def f8():
    u = pd.read_csv(ROOT / U1, dtype=str); u = u[(u.included_main == "yes") & u.ctu_auroc_for_summary.notna()].copy()
    u["auroc"] = u.ctu_auroc_for_summary.astype(float)
    G = [("within_CTU_segment_or_image", "Segment/image\nsplit", "#D55E00", "o"),
         ("within_CTU_record", "Record\nsplit", "#0072B2", "s"),
         ("external_test_on_CTU", "External\ntest", "#009E73", "D")]
    fig, ax = plt.subplots(figsize=(W1, 2.9)); fig.subplots_adjust(bottom=0.24); rows = []; ok = True
    for i, (g, lab, c, mkr) in enumerate(G):
        s = u[u.evaluation_design == g].sort_values("auroc"); n = len(s)
        xs = i + (np.arange(n) - (n - 1) / 2) * 0.09
        p = ax.plot(xs, s.auroc, ls="none", marker=mkr, ms=4.6, mfc="white", mec=c, mew=1.0, zorder=3)[0]
        ok &= same(p.get_ydata(), s.auroc)
        med = s.auroc.median(); ax.plot([i - 0.3, i + 0.3], [med, med], color=c, lw=1.4, zorder=2)
        ax.text(i, 0.53, f"n = {n}", ha="center", va="bottom", fontsize=7.5)
        rows += [dict(study_id=r.id, first_author=r.first_author, year=r.year, evaluation_design=g, ctu_auroc=r.auroc) for r in s.itertuples()]
    ax.axhline(0.5, **REF); ax.set_xticks(range(3)); ax.set_xticklabels([g[1] for g in G]); ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0.5, 1.01); ax.set_ylabel("Reported AUROC on CTU-UHB"); ax.set_xlabel("Evaluation design")
    pd.DataFrame(rows).to_csv(OUT / "F8_plot_data.csv", index=False)
    qa(fig, "F8", [ax], ci_list=[(ax, "y", u.auroc.values)])
    save(fig, "F8", "PASS" if ok else "FAIL", None)

for f in (f1, f2, f3, f4, f5, f6, f7, f8): f()
# grayscale distinctness (encoding tables)
def distinct(tab):
    v = list(tab.values()); return all((a[2], a[3]) != (b[2], b[3]) for i, a in enumerate(v) for b in v[i + 1:])
gs_ok = distinct(MOD) and distinct({k: (0, 0, v[2], v[3]) for k, v in GRP.items()}) and distinct({k: (0, 0, v[2], v[3]) for k, v in FEAT.items()})
for k in CHECKS: CHECKS[k]["GRAYSCALE_DISTINCT"] = "PASS" if gs_ok else "FAIL"
files = sorted(p for p in OUT.iterdir() if p.name != "OUTPUT_SHA256.txt")
(OUT / "OUTPUT_SHA256.txt").write_text("".join(f"{sha(p)}  {p.name}\n" for p in files))
lines = ["=== REPORT P21-T10 ===", "status=PASS" if all(all(str(v).startswith(("PASS", "N/A")) or kk == "SIZE" for kk, v in c.items()) for c in CHECKS.values()) else "status=CHECK_FAIL",
         "presentation-only; no-new-numbers; F8 = U1 literature audit (published AUROCs)", f"font_used={Path(FONT_USED).name}", "input_sha=PASS (7/7)"]
for k, c in CHECKS.items(): lines.append(f"{k}: {c.pop('SIZE')}; " + " ".join(f"{a}:{b}" for a, b in c.items()))
lines += REPORT + [f"files_written={len(files) + 1}", "=== END ==="]
(OUT / "report.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

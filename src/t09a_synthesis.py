"""P21-T09a synthesis: tables T6-T8, S-T1, figures F6-F7, NUMBERS_TRACE_T09a. Reads frozen outputs only."""
import csv, hashlib, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from scipy.stats.contingency import odds_ratio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1]            # root that mirrors D:\Tuan\AI_training
OUT = sys.argv[2]            # output dir (t09a_synthesis)
GATE = json.loads(sys.argv[3])  # gate counts recomputed on PC-A from units/
P = "P21_CTG_Acidemia/outputs"
os.makedirs(OUT, exist_ok=False); os.makedirs(f"{OUT}/tables"); os.makedirs(f"{OUT}/figures")

def sha8(rel): return hashlib.sha256(open(f"{SRC}/{rel}", "rb").read()).hexdigest()[:8]
def jload(rel): return json.load(open(f"{SRC}/{rel}"))
def get(d, path):
    for k in path.split("."): d = d[k] if not isinstance(d, list) else d[int(k)]
    return d

trace = []   # (table,row,column,value,source_file,source_sha8,source_field)
def T(table, row, col, value, src, field):
    trace.append([table, row, col, value, src, sha8(src) if src else "", field]); return value

# ---------- T6: QC exclusion bias (post hoc) ----------
LAB = "data_proc/ctu-uhb-ctgdb/1.0.0/labels.parquet"
lab = pd.read_parquet(f"{SRC}/{LAB}")
t6 = []
for y in ["y705", "y715", "y720"]:
    pos, ex = lab[y].astype(bool), lab["excluded"].astype(bool)
    a, b = int((ex & pos).sum()), int((~ex & pos).sum())
    c, d = int((ex & ~pos).sum()), int((~ex & ~pos).sum())
    p = fisher_exact([[a, b], [c, d]], alternative="two-sided").pvalue
    orr = odds_ratio([[a, b], [c, d]], kind="conditional")
    ci = orr.confidence_interval(0.95)
    r = dict(outcome=y, excluded_pos=a, total_pos=a + b, pct_excl_pos=100 * a / (a + b),
             excluded_neg=c, total_neg=c + d, pct_excl_neg=100 * c / (c + d),
             conditional_OR=orr.statistic, OR_ci_low=ci.low, OR_ci_high=ci.high, fisher_p=p, note="post hoc")
    for k, v in r.items():
        if k not in ("outcome", "note"): T("T6", y, k, v, LAB, f"computed:{k} from columns {y},excluded")
    t6.append(r)
pd.DataFrame(t6).to_csv(f"{OUT}/tables/T6a_QC_exclusion.csv", index=False)

rows = []
for m, src, path in [("Logistic (features)", "t03_baselines", "M0"), ("GBDT (features)", "t03_baselines", "M1"),
                     ("CNN", "t04_dl", "models.M2"), ("ResNet", "t04_dl", "models.M3")]:
    r = {"model": m}
    for cfg, tag in [("A_primary", "A"), ("B_noexcl", "B")]:
        rel = f"{P}/{src}/{cfg}/metrics.json"; d = jload(rel)
        if src == "t03_baselines":
            fa, fc = f"{path}.mean_probability_metrics.auroc", f"{path}.auroc_ci95"
        else:
            fa, fc = f"{path}.mean_probability_over_repeats_and_seeds.auroc", f"{path}.mean_probability_over_repeats_and_seeds.auroc_ci95"
        r[f"AUROC_{tag}"] = T("T6", m, f"AUROC_{tag}", get(d, fa), rel, fa)
        ci = get(d, fc)
        r[f"CI_low_{tag}"] = T("T6", m, f"CI_low_{tag}", ci[0], rel, fc + ".0")
        r[f"CI_high_{tag}"] = T("T6", m, f"CI_high_{tag}", ci[1], rel, fc + ".1")
    rows.append(r)
EXP = f"{P}/t06a_stats/exploratory.csv"
ex = pd.read_csv(f"{SRC}/{EXP}")
for cfg, tag in [("A_primary", "A"), ("B_noexcl", "B")]:
    e = ex[(ex.analysis == "E1") & (ex.config == cfg)].iloc[0]
    rows_e = rows[-1] if rows[-1]["model"] == "Signal loss only (E1)" else None
    if rows_e is None: rows.append({"model": "Signal loss only (E1)"}); rows_e = rows[-1]
    for col, fld in [(f"AUROC_{tag}", "estimate"), (f"CI_low_{tag}", "ci95_low"), (f"CI_high_{tag}", "ci95_high")]:
        rows_e[col] = T("T6", "Signal loss only (E1)", col, float(e[fld]), EXP, f"row E1/{cfg}:{fld}")
pd.DataFrame(rows).to_csv(f"{OUT}/tables/T6b_AUROC_A_vs_B.csv", index=False)

# ---------- T7: FHRMA validation ----------
T07, T07C = f"{P}/t07_fhrma/metrics.json", f"{P}/t07c_wmfb/metrics.json"
d7, d7c = jload(T07), jload(T07C)
t7 = []
for name, d, base, rel in [("Original extractor", d7, "splits.train.metrics", T07), ("WMFB", d7c, "train_metrics", T07C)]:
    r = {"extractor": name}
    for mk, lab_ in [("baseline_mae_bpm", "MAE_bpm"), ("accel_f1", "Acc_F1"), ("decel_f1", "Dec_F1")]:
        r[lab_] = T("T7", name, lab_, get(d, f"{base}.{mk}.estimate"), rel, f"{base}.{mk}.estimate")
        ci = get(d, f"{base}.{mk}.ci95_cluster_record")
        r[lab_ + "_lo"] = T("T7", name, lab_ + "_lo", ci[0], rel, f"{base}.{mk}.ci95_cluster_record.0")
        r[lab_ + "_hi"] = T("T7", name, lab_ + "_hi", ci[1], rel, f"{base}.{mk}.ci95_cluster_record.1")
    if name == "WMFB":
        for k in ["MADI", "RMSD_bpm"]:
            r[k] = T("T7", name, k, get(d, f"official_evaluateaam.median_metrics.{k}"), rel, f"official_evaluateaam.median_metrics.{k}")
    r["note"] = "FHRMA training set (n = 66); test-set expert labels not publicly available"
    t7.append(r)
pd.DataFrame(t7).to_csv(f"{OUT}/tables/T7_FHRMA_validation.csv", index=False)

# ---------- T8: feature v2 sensitivity ----------
S7D, C7D = f"{P}/t07d_featv2/summary.json", f"{P}/t07d_featv2/comparisons.csv"
s7d = jload(S7D)
t8a = []
for m in ["M0", "M0-v2", "M1", "M1-v2", "M2", "M3"]:
    f = f"comparisons.model_metrics.{m}"
    t8a.append({"model": m, "AUROC": T("T8", m, "AUROC", get(s7d, f + ".auroc"), S7D, f + ".auroc"),
                "CI_low": T("T8", m, "CI_low", get(s7d, f + ".ci95")[0], S7D, f + ".ci95.0"),
                "CI_high": T("T8", m, "CI_high", get(s7d, f + ".ci95")[1], S7D, f + ".ci95.1")})
pd.DataFrame(t8a).to_csv(f"{OUT}/tables/T8a_models_v2.csv", index=False)
cmp = pd.read_csv(f"{SRC}/{C7D}")
t8b = []
for i, r in cmp.iterrows():
    row = {"analysis": r.analysis, "contrast": r.contrast}
    for col in ["delta_AUROC", "ci95_low", "ci95_high", "p_raw", "n"]:
        row[col] = T("T8", f"{r.analysis}|{r.contrast}", col, r[col].item() if hasattr(r[col], "item") else r[col], C7D, f"row {i}:{col}")
    row["note"] = "sensitivity, not Holm" if r.analysis != "EXPLORATORY_E2" else "exploratory, post hoc"
    t8b.append(row)
pd.DataFrame(t8b).to_csv(f"{OUT}/tables/T8b_comparisons_v2.csv", index=False)

# ---------- S-T1: image-based reproduction ----------
st = []
for run, label, rel, grid in [("T08", "original", f"{P}/t08_deepfhr/summary.json", GATE["t08_deepfhr"]),
                              ("T08b", "corrected", f"{P}/t08b_deepfhr/summary.json", GATE["t08b_deepfhr"])]:
    d = jload(rel)
    for y in ["y705", "y715"]:
        r = {"run": run, "outcome": y, "images_total": T("S-T1", f"{run}/{y}", "images_total", d["images"]["images_total"], rel, "images.images_total"),
             "seed_x_repeat_grid": f"{grid['seed_repeat_combos']}/9"}
        for k in ["S_image_auroc", "L_image_auroc", "S_record_auroc", "L_record_auroc", "delta_image", "delta_record"]:
            b = f"metrics.{y}.aggregate.{k}"
            r[k] = T("S-T1", f"{run}/{y}", k, get(d, b + ".mean"), rel, b + ".mean")
            r[k + "_lo"] = T("S-T1", f"{run}/{y}", k + "_lo", get(d, b + ".ci95_cluster_record")[0], rel, b + ".ci95_cluster_record.0")
            r[k + "_hi"] = T("S-T1", f"{run}/{y}", k + "_hi", get(d, b + ".ci95_cluster_record")[1], rel, b + ".ci95_cluster_record.1")
        if run == "T08b":
            a = d["assay_sensitivity"]
            r["gate_train_auroc_gt_0_75"] = f"{T('S-T1', 'T08b', 'S_img_y705_train_auroc_gt_0_75', a['S_img_y705_train_auroc_gt_0_75'], rel, 'assay_sensitivity.S_img_y705_train_auroc_gt_0_75')}/{a['n_main_S_img_y705']} (required >= 36)"
            r["near_constant_units"] = f"{T('S-T1', 'T08b', 'near_constant_probability_count', a['near_constant_probability_count'], rel, 'assay_sensitivity.near_constant_probability_count')}/{a['n_main_units']}"
            assert a["S_img_y705_train_auroc_gt_0_75"] == grid["S_y705_train_auroc_gt075"] and a["near_constant_probability_count"] == grid["near_constant_units"], "gate mismatch vs units/"
        else:
            r["gate_train_auroc_gt_0_75"] = "not recorded"; r["near_constant_units"] = f"{grid['near_constant_units']}/{grid['main_units']}"
        r["conclusion"] = "inconclusive (assay sensitivity not met)" if run == "T08b" else "uninformative run (incomplete grid, model did not learn)"
        st.append(r)
pd.DataFrame(st).to_csv(f"{OUT}/tables/ST1_image_reproduction.csv", index=False)

# ---------- LaTeX booktabs ----------
def tex(df, path, cols, fmt=3):
    with open(path, "w", encoding="utf8") as f:
        f.write("\\begin{tabular}{" + "l" * len(cols) + "}\n\\toprule\n" + " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\\n\\midrule\n")
        for _, r in df[cols].iterrows():
            cells = []
            for v in r:
                if isinstance(v, (float, np.floating)): cells.append(f"{v:.{fmt}f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}")
                else: cells.append(str(v).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&"))
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
for name in ["T6a_QC_exclusion", "T6b_AUROC_A_vs_B", "T7_FHRMA_validation", "T8a_models_v2", "T8b_comparisons_v2", "ST1_image_reproduction"]:
    df = pd.read_csv(f"{OUT}/tables/{name}.csv")
    tex(df, f"{OUT}/tables/{name}.tex", [c for c in df.columns if c != "note"])

# ---------- Figures ----------
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "svg.fonttype": "none", "axes.linewidth": 0.8})
OI = {"blue": "#0072B2", "verm": "#D55E00", "orange": "#E69F00", "grey": "#999999"}
def save(fig, name):
    for ext in ["png", "svg"]: fig.savefig(f"{OUT}/figures/{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

# F6
items = [("Segment-level split\n(segment AUROC)", "L1 - S_original (segment)", "L1 - S_seg_v2 (segment)"),
         ("Segment-level split\n(record AUROC)", "L1 - S_seg_original (record)", "L1 - S_seg_v2 (record)"),
         ("Oversampling before split", "L2 - S_original (real record)", "L2 - S_v2 (real record)"),
         ("Tuning on full data", "L3 - S_original (real record)", "L3 - S_v2 (real record)")]
fig, ax = plt.subplots(figsize=(3.5, 2.8)); pdat = []
for i, (lbl, c0, c1) in enumerate(items):
    for j, (c, col, mk, nm) in enumerate([(c0, OI["blue"], "s", "Original features"), (c1, OI["verm"], "o", "WMFB features")]):
        r = cmp[cmp.contrast == c].iloc[0]; y = i + (-0.15 if j == 0 else 0.15)
        ax.errorbar(r.delta_AUROC, y, xerr=[[r.delta_AUROC - r.ci95_low], [r.ci95_high - r.delta_AUROC]], fmt=mk, color=col, ms=4, capsize=2, lw=1, label=nm if i == 0 else None)
        pdat.append({"protocol": lbl.replace("\n", " "), "features": nm, "delta": r.delta_AUROC, "lo": r.ci95_low, "hi": r.ci95_high})
ax.axvline(0, color=OI["grey"], ls=":", lw=0.8)
ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items]); ax.invert_yaxis()
ax.set_xlabel("ΔAUROC, leaky − strict (95% CI)"); ax.legend(frameon=False, loc="lower right")
pd.DataFrame(pdat).to_csv(f"{OUT}/figures/F6_plot_data.csv", index=False); save(fig, "F6")

# F7
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 2.2), gridspec_kw={"width_ratios": [1, 2]}); pdat = []
for k, (r, col) in enumerate(zip(t7, [OI["blue"], OI["verm"]])):
    a1.bar(k, r["MAE_bpm"], yerr=[[r["MAE_bpm"] - r["MAE_bpm_lo"]], [r["MAE_bpm_hi"] - r["MAE_bpm"]]], color=col, capsize=2, width=0.6)
    for m, (key) in enumerate(["Acc_F1", "Dec_F1"]):
        x = m + (-0.18 if k == 0 else 0.18)
        a2.bar(x, r[key], yerr=[[r[key] - r[key + "_lo"]], [r[key + "_hi"] - r[key]]], color=col, capsize=2, width=0.34, label=r["extractor"] if m == 0 else None)
    pdat.append({k2: r[k2] for k2 in r if k2 != "note"})
a1.axhline(5, color=OI["grey"], ls="--", lw=0.8); a1.set_xticks([0, 1]); a1.set_xticklabels(["Original", "WMFB"]); a1.set_ylabel("Baseline MAE (bpm)")
a2.axhline(0.60, color=OI["grey"], ls="--", lw=0.8); a2.set_xticks([0, 1]); a2.set_xticklabels(["Accelerations", "Decelerations"]); a2.set_ylabel("F1 score"); a2.set_ylim(0, 1)
a2.legend(frameon=False, loc="upper left", fontsize=7)
fig.tight_layout(); pd.DataFrame(pdat).to_csv(f"{OUT}/figures/F7_plot_data.csv", index=False); save(fig, "F7")

# ---------- NUMBERS_TRACE + automatic match ----------
tr = pd.DataFrame(trace, columns=["table", "row", "column", "value", "source_file", "source_sha8", "source_field"])
ok = 0
for _, r in tr.iterrows():
    f = r.source_field
    if f.startswith("computed:"): ok += 1; continue          # recomputed above from labels.parquet
    if r.source_file.endswith(".json"): v = get(jload(r.source_file), f)
    elif f.startswith("row E1/"):
        cfg, col = f[7:].split(":"); v = float(ex[(ex.analysis == "E1") & (ex.config == cfg)].iloc[0][col])
    else:
        i, col = f[4:].split(":"); v = cmp.iloc[int(i)][col]
    ok += int(float(v) == float(r.value))
tr.to_csv(f"{OUT}/NUMBERS_TRACE_T09a.csv", index=False)
res = {"trace_rows": len(tr), "trace_matched": ok, "match_pct": 100 * ok / len(tr), "fisher": t6,
       "font_note": "SVG text keeps font-family Arial; PNG rendered with Liberation Sans (Arial-metric) because Arial is not installed on the render host"}
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ["trace_rows", "trace_matched", "match_pct"]}))
for r in t6: print(r["outcome"], r["excluded_pos"], r["total_pos"], r["excluded_neg"], r["total_neg"], round(r["conditional_OR"], 3), round(r["OR_ci_low"], 3), round(r["OR_ci_high"], 3), f"{r['fisher_p']:.3g}")

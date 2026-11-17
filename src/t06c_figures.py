from __future__ import annotations
import argparse, hashlib, os, re, shutil, time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import pandas as pd
import yaml

PREREG = '40573556ecf4664c430430eb99eaaa20d1d0958aa32647cd30b7144434d048eb'
PLOT_FILES = [f'F{i}_plot_data.csv' for i in range(1, 6)]
NAME_MAP = {
    'M0': 'Logistic (features)',
    'M1': 'GBDT (features)',
    'M2': 'CNN',
    'M3': 'ResNet',
    'treat_all': 'Treat all',
    'treat_none': 'Treat none',
}

def require(test, msg):
    if not test:
        raise ValueError(msg)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def parse_manifest(path: Path) -> dict[str, str]:
    out = {}
    for raw in path.read_text(encoding='utf-8-sig').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        m = re.fullmatch(r'([0-9a-fA-F]{8}|[0-9a-fA-F]{64})\s+(.+)', line)
        if not m:
            continue
        digest, name = m.group(1).lower(), m.group(2).strip().replace('\\', '/')
        name = name.removeprefix('./')
        if name.lower().startswith('d:/tuan/ai_training/'):
            name = name[len('D:/Tuan/AI_training/'):]
        out[name] = digest
    return out


def verify_inputs(root: Path, manifest_rel: str, plot_folder_rel: str):
    manifest = root / manifest_rel
    require(manifest.is_file(), f'Missing frozen manifest: {manifest}')
    declared = parse_manifest(manifest)
    input_dir = root / plot_folder_rel
    require(input_dir.is_dir(), f'Missing input figure folder: {input_dir}')
    files = []
    report = {}
    for name in PLOT_FILES:
        path = input_dir / name
        require(path.is_file(), f'Missing plot-data file: {path}')
        key = rel(path, root)
        require(key in declared, f'MISSING_FROZEN_SHA for {key}')
        current = sha(path)
        require(current.startswith(declared[key]), f'FROZEN_SHA_MISMATCH {key}: locked={declared[key]} actual={current[:8]}')
        files.append(path)
        report[key] = current[:8]
    return files, report


def save_fig(fig, outdir: Path, name: str, w_in: float, h_in: float, tight=False):
    fig.set_size_inches(w_in, h_in)
    png = outdir / f'{name}.png'
    svg = outdir / f'{name}.svg'
    bbox = 'tight' if tight else None
    fig.savefig(png, dpi=600, bbox_inches=bbox)
    fig.savefig(svg, bbox_inches=bbox)
    plt.close(fig)
    with Image.open(png) as im:
        w, h = im.size
    require(w >= 2000, f'{name}.png width too small: {w}')
    require(svg.stat().st_size > 100, f'{name}.svg appears empty')
    return w, h


def auc_from_plot(sub: pd.DataFrame) -> float:
    x = sub['x'].to_numpy(dtype=float)
    y = sub['y'].to_numpy(dtype=float)
    order = np.argsort(x, kind='mergesort')
    x = x[order]
    y = y[order]
    return float(np.trapezoid(y, x))


def copy_plot_data(input_files: list[Path], outdir: Path):
    for p in input_files:
        shutil.copy2(p, outdir / p.name)


def make_f1(outdir: Path, cfg: dict):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    edge = '#4D4D4D'
    fill = '#F7F7F7'
    boxes = {
        'top': (0.26, 0.80, 0.48, 0.11, 'CTU-UHB recordings, n = 552'),
        'analysis': (0.26, 0.58, 0.48, 0.11, 'Analysis cohort, n = 518\n(pH < 7.05: 30)'),
        'strict': (0.07, 0.25, 0.38, 0.12, 'Strict evaluation:\nrepeated nested CV (10 × 5 × 5)'),
        'leak': (0.55, 0.25, 0.38, 0.12, 'Leakage audit:\n10-min segments, 5,224 retained, 499 rejected'),
        'excluded': (0.01, 0.62, 0.22, 0.15, 'Excluded: > 50% FHR signal loss in\nlast 30 min of stage 1, n = 34\n(acidaemic 10; non-acidaemic 24)'),
        'sens': (0.77, 0.79, 0.21, 0.11, 'Sensitivity:\nno exclusion, n = 552\n(pH < 7.05: 40)'),
    }
    for key, (x, y, w, h, txt) in boxes.items():
        rect = Rectangle((x, y), w, h, ec=edge, fc=fill, lw=1.0, ls='--' if key == 'sens' else '-')
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=8)
    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', lw=1.0, color=edge))
    arrow(0.50, 0.80, 0.50, 0.69)
    arrow(0.26, 0.69, 0.12, 0.69)
    ax.plot([0.74, 0.77], [0.855, 0.855], ls='--', lw=1.0, color=edge)
    ax.plot([0.50, 0.50], [0.58, 0.45], color=edge, lw=1.0)
    ax.plot([0.26, 0.74], [0.45, 0.45], color=edge, lw=1.0)
    arrow(0.26, 0.45, 0.26, 0.37)
    arrow(0.74, 0.45, 0.74, 0.37)
    return save_fig(fig, outdir, 'F1', cfg['figure_specs']['F1']['width_in'], cfg['figure_specs']['F1']['height_in'], tight=True)


def f2_label(lbl: str) -> str:
    if lbl == 'L1/M1':
        return 'Segment-level split — GBDT (features)'
    if lbl == 'L1 record/M1':
        return 'Record-level (descriptive) — GBDT (features)'
    if lbl == 'L1/M2':
        return 'Segment-level split — CNN'
    if lbl == 'L1 record/M2':
        return 'Record-level (descriptive) — CNN'
    if lbl == 'L2/M1':
        return 'Oversampling before split — GBDT (features)'
    if lbl == 'L2 augmented/M1':
        return 'Test incl. synthetic/duplicated samples — GBDT (features)'
    if lbl == 'L2/M2':
        return 'Oversampling before split — CNN'
    if lbl == 'L2 augmented/M2':
        return 'Test incl. synthetic/duplicated samples — CNN'
    if lbl == 'L3/M1':
        return 'Tuning on full data — GBDT (features)'
    if lbl == 'L3/M2':
        return 'Checkpoint selected on test — CNN'
    return lbl


def make_f2(df: pd.DataFrame, outdir: Path, cfg: dict):
    fig, ax = plt.subplots()
    blue = cfg['palette']['blue']
    green = cfg['palette']['green']
    orange = cfg['palette']['orange']
    purple = cfg['palette']['purple']
    # reorder only; do not add or remove rows
    order = ['L1/M1', 'L1 record/M1', 'L1/M2', 'L1 record/M2',
             'L2/M1', 'L2 augmented/M1', 'L2/M2', 'L2 augmented/M2',
             'L3/M1', 'L3/M2']
    plot = df.copy()
    plot['label'] = plot['label'].astype(str)
    plot = plot.set_index('label').loc[order].reset_index()
    y = np.arange(len(plot))[::-1]
    ax.axvline(0.5, color='#888888', ls=':', lw=0.8)
    labels = []
    for i, r in plot.iterrows():
        yy = y[i]
        s = float(r['S']); l = float(r['L'])
        desc = str(r['descriptive']).lower() in ('true', '1')
        if str(r['S_low']) not in ('', 'nan'):
            ax.plot([float(r['S_low']), float(r['S_high'])], [yy, yy], c=blue, lw=1.25)
        if str(r['L_low']) not in ('', 'nan'):
            ax.plot([float(r['L_low']), float(r['L_high'])], [yy, yy], c=(green if not desc else (purple if 'augmented' in r['label'] else orange)), lw=1.25)
        ax.annotate('', xy=(l, yy), xytext=(s, yy), arrowprops={'arrowstyle': '->', 'lw': .7, 'color': '#666666'})
        ax.plot(s, yy, marker='s', ms=4.5, c=blue, clip_on=False)
        if desc:
            mk = '^'
            col = purple if 'augmented' in r['label'] else orange
            ax.plot(l, yy, marker=mk, ms=5.0, c=col, clip_on=False)
            annot = 'descriptive'
        else:
            col = blue if '/M1' in r['label'] else green
            ax.plot(l, yy, marker='o', ms=4.5, c=col, clip_on=False)
            annot = f"Δ = {float(r['delta']):.3f}; Holm p = {float(r['p_holm']):.4f}"
        ax.text(1.025, yy, annot, ha='left', va='center', fontsize=8)
        labels.append(f2_label(str(r['label'])))
    for cut in [5.5, 1.5]:
        ax.axhline(cut, color='#DDDDDD', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(-0.7, len(plot)-0.3)
    ax.set_xlabel('AUROC (95% CI)')
    handles = [
        Line2D([0],[0], marker='s', color='none', markerfacecolor=blue, markeredgecolor=blue, label='Strict (record-level)'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor=green, markeredgecolor=green, label='Leaky (segment/row-level tested)'),
        Line2D([0],[0], marker='^', color='none', markerfacecolor=orange, markeredgecolor=orange, label='Record-level (descriptive)'),
        Line2D([0],[0], marker='^', color='none', markerfacecolor=purple, markeredgecolor=purple, label='Test set including synthetic/duplicated samples (descriptive)'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.31, right=0.74, bottom=0.22, top=0.98)
    return save_fig(fig, outdir, 'F2', cfg['figure_specs']['F2']['width_in'], cfg['figure_specs']['F2']['height_in'], tight=False)


def make_f3(df: pd.DataFrame, outdir: Path, cfg: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    colors = {'M0': '#000000', 'M1': cfg['palette']['blue'], 'M2': cfg['palette']['orange'], 'M3': cfg['palette']['green']}
    for model in ['M0', 'M1', 'M2', 'M3']:
        roc = df[(df['panel'] == 'ROC') & (df['model'] == model)].copy()
        auc = auc_from_plot(roc)
        ax1.plot(roc['x'].astype(float), roc['y'].astype(float), lw=1.4, color=colors[model], label=f"{NAME_MAP[model]} (AUROC {auc:.3f})")
        cal = df[(df['panel'] == 'calibration') & (df['model'] == model)].copy()
        ax2.plot(cal['x'].astype(float), cal['y'].astype(float), marker='o', ms=3, lw=1.2, color=colors[model], label=NAME_MAP[model])
    ax1.plot([0, 1], [0, 1], ls=':', c='#777777', lw=0.8)
    ax2.plot([0, 1], [0, 1], ls=':', c='#777777', lw=0.8)
    ax1.set(xlim=(0,1), ylim=(0,1), xlabel='1 − specificity', ylabel='Sensitivity')
    ax2.set(xlim=(0,1), ylim=(0,1), xlabel='Mean predicted probability', ylabel='Observed event frequency')
    ax1.legend(loc='lower right', frameon=False, fontsize=8)
    ax2.legend(loc='upper left', frameon=False, fontsize=8)
    ax1.text(-0.15, 1.03, '(a)', transform=ax1.transAxes, fontweight='bold', fontsize=8)
    ax2.text(-0.15, 1.03, '(b)', transform=ax2.transAxes, fontweight='bold', fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.98, wspace=0.30, bottom=0.18, top=0.92)
    return save_fig(fig, outdir, 'F3', cfg['figure_specs']['F3']['width_in'], cfg['figure_specs']['F3']['height_in'], tight=False)


def make_f4(df: pd.DataFrame, outdir: Path, cfg: dict):
    fig, ax = plt.subplots()
    line_styles = {'M1': '-', 'M3': '-', 'treat_all': '--', 'treat_none': ':'}
    colors = {'M1': cfg['palette']['blue'], 'M3': cfg['palette']['green'], 'treat_all': '#777777', 'treat_none': '#000000'}
    for strat in ['M1', 'M3', 'treat_all', 'treat_none']:
        sub = df[df['strategy'] == strat]
        ax.plot(sub['threshold'].astype(float) * 100, sub['net_benefit'].astype(float), label=NAME_MAP.get(strat, strat), color=colors[strat], lw=1.4, ls=line_styles[strat])
    ax.set(xlabel='Threshold probability (%)', ylabel='Net benefit', xlim=(5, 30))
    ax.legend(frameon=False, fontsize=8, loc='best')
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.98)
    return save_fig(fig, outdir, 'F4', cfg['figure_specs']['F4']['width_in'], cfg['figure_specs']['F4']['height_in'], tight=False)


def make_f5(df: pd.DataFrame, outdir: Path, cfg: dict):
    fig, ax = plt.subplots()
    colors = {'M1': cfg['palette']['blue'], 'M3': cfg['palette']['green']}
    x_positions = {'le_median': 0, 'gt_median': 1}
    offsets = {'M1': -0.13, 'M3': 0.13}
    for _, r in df.iterrows():
        model = str(r['model']); group = str(r['group'])
        x = x_positions[group] + offsets[model]
        est, lo, hi = float(r['auroc']), float(r['ci95_low']), float(r['ci95_high'])
        ax.errorbar([x], [est], yerr=[[est-lo], [hi-est]], fmt='o', ms=4, capsize=3, color=colors[model], lw=1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Low signal loss\n(≤ median)', 'High signal loss\n(> median)'])
    ax.set_xlim(-0.38, 1.38)
    ax.set_ylim(0, 1)
    ax.set_ylabel('AUROC (95% CI)')
    ax.axhline(0.5, ls=':', color='#888888', lw=0.8)
    for group, x in [('le_median', 0), ('gt_median', 1)]:
        sub = df[(df['group'] == group) & (df['model'] == 'M1')]
        require(len(sub) == 1, f'F5 missing M1 row for {group}')
        pos = str(sub.iloc[0]['positives'])
        label = 'positives < 10' if pos == '<10' else f'positives = {pos}'
        ax.text(x, 0.05, label, ha='center', va='center', fontsize=8)
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[m], markeredgecolor=colors[m], label=NAME_MAP[m]) for m in ['M1', 'M3']]
    ax.legend(handles=handles, frameon=False, loc='upper right', fontsize=8)
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.25, top=0.98)
    return save_fig(fig, outdir, 'F5', cfg['figure_specs']['F5']['width_in'], cfg['figure_specs']['F5']['height_in'], tight=False)


def emit_output_manifest(outdir: Path) -> str:
    manifest = outdir / 'OUTPUT_SHA256.txt'
    lines = []
    for p in sorted(outdir.glob('*')):
        if p.is_file() and p.name != 'OUTPUT_SHA256.txt':
            lines.append(f'{sha(p)} {p.name}')
    manifest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return sha(manifest)[:8]


def run(cfg: dict, stage: str) -> int:
    root = Path(cfg['work_root'])
    require(root.is_dir(), f'work_root not found: {root}')
    input_files, status = verify_inputs(root, cfg['input_manifest'], cfg['input_plot_folder'])
    if stage == 'inspect':
        for k, v in status.items():
            print(f'PLOT_DATA_SHA_{Path(k).name}={v} PASS')
        print('P21_T06C_INSPECT_PASS')
        return 0

    outdir = root / cfg['output_folder']
    require(not outdir.exists(), f'STOP: output directory exists: {outdir}')
    outdir.mkdir(parents=True)
    plt.rcParams.update({'font.family': cfg.get('font_family', 'Arial'), 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
    dfs = {p.name: pd.read_csv(p) for p in input_files}
    dims = {}
    dims['F1'] = make_f1(outdir, cfg)
    dims['F2'] = make_f2(dfs['F2_plot_data.csv'], outdir, cfg)
    dims['F3'] = make_f3(dfs['F3_plot_data.csv'], outdir, cfg)
    dims['F4'] = make_f4(dfs['F4_plot_data.csv'], outdir, cfg)
    dims['F5'] = make_f5(dfs['F5_plot_data.csv'], outdir, cfg)
    copy_plot_data(input_files, outdir)
    out_sha = {name: sha(outdir / f'{name}.png')[:8] for name in ['F1', 'F2', 'F3', 'F4', 'F5']}
    lines = ['=== REPORT P21-T06c ===', 'status=PASS; PC-A CPU; no-new-numbers; no-data-changes', f'PREREG_v1.3_SHA256={cfg["prereg_sha256"]}', f'plot_data_frozen_sha=PASS; files={len(input_files)}; manifest={cfg["input_manifest"]}']
    for name in ['F1', 'F2', 'F3', 'F4', 'F5']:
        w, h = dims[name]
        lines.append(f'{name}: {w}x{h}px; png_sha8={out_sha[name]}; svg_sha8={sha(outdir/(name+".svg"))[:8]}')
    lines.append(f'output_sha8_manifest={emit_output_manifest(outdir)}')
    lines.append('=== END ===')
    (outdir / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--stage', default='run', choices=['inspect', 'run'])
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    require(cfg.get('prereg_sha256') == PREREG, 'PREREG SHA mismatch in config')
    t0 = time.time()
    try:
        code = run(cfg, args.stage)
    except Exception as e:
        print(f'P21_T06C_STOP: {type(e).__name__}: {e}')
        print('P21_T06C_EXIT_CODE=2')
        return 2
    if args.stage == 'run':
        print(f'runtime_s={time.time()-t0:.2f}')
    print('P21_T06C_EXIT_CODE=0')
    return code

if __name__ == '__main__':
    raise SystemExit(main())

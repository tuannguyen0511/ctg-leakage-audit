"""P21-T06d. Presentation-only plots from frozen T06b plot_data; byte-copy T06c F3-F5.

No training, reestimation, statistical tests, or modification of prior outputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np
import pandas as pd
import yaml

PREREG = '40573556ecf4664c430430eb99eaaa20d1d0958aa32647cd30b7144434d048eb'
PLOT_FILES = tuple(f'F{i}_plot_data.csv' for i in range(1, 6))
COPY_FILES = tuple(f'F{i}.{ext}' for i in (3, 4, 5) for ext in ('png', 'svg'))
STYLE = {
    'strict': '#0072B2',
    'leaky': '#D55E00',
    'record': '#E69F00',
    'synthetic': '#CC79A7',
    'grey': '#808080',
}
MODELS = {'M1': 'GBDT (features)', 'M2': 'CNN'}
PROTOCOLS = {
    'L1': 'Segment-level split',
    'L2': 'Oversampling before split',
    'L3/M1': 'Tuning on full data',
    'L3/M2': 'Checkpoint selected on test',
}


def require(ok: bool, reason: str):
    if not ok:
        raise ValueError(reason)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def parse_sha_manifest(path: Path) -> dict[str, str]:
    """Accept only explicit SHA8/SHA256 + root-relative path records, ignore #."""
    require(path.is_file(), f'MISSING_LOCK_MANIFEST: {path}')
    records: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([0-9a-fA-F]{8}|[0-9a-fA-F]{64})\s+\*?(.+?)\s*', line)
        if not match:
            # This manifest is an explicit SHA listing, not a free-form report.
            raise ValueError(f'MALFORMED_FROZEN_SHA line {lineno}: {line[:70]}')
        digest, filename = match.group(1).lower(), match.group(2).replace('\\', '/').removeprefix('./')
        if filename.lower().startswith('d:/tuan/ai_training/'):
            filename = filename[len('D:/Tuan/AI_training/'):]
        require(filename not in records or records[filename] == digest,
                f'DUPLICATE_CONFLICTING_LOCK: {filename}')
        records[filename] = digest
    return records


def frozen_plot_data_preflight(root: Path, cfg: dict):
    manifest = parse_sha_manifest(root / cfg['frozen_manifest'])
    source_dir = root / cfg['frozen_plot_dir']
    files: dict[str, Path] = {}
    values: dict[str, str] = {}
    for filename in PLOT_FILES:
        path = source_dir / filename
        require(path.is_file(), f'MISSING_INPUT: {path}')
        key = rel(path, root)
        # For compatibility with project-relative lock paths, no guessing of hashes.
        expected = manifest.get(key) or manifest.get(key.removeprefix('P21_CTG_Acidemia/'))
        require(expected is not None, f'MISSING_FROZEN_SHA: {key}')
        got = sha256(path)
        require(got.startswith(expected), f'FROZEN_SHA_MISMATCH: {filename}: locked={expected} actual={got[:8]}')
        files[filename] = path
        values[filename] = got[:8]
    return files, values


def approved_copy_preflight(root: Path, cfg: dict):
    srcdir = root / cfg['approved_t06c_dir']
    pins = cfg['approved_t06c_sha8']
    require(set(pins) == set(COPY_FILES), 'APPROVED_SHA_LIST_INVALID')
    files: dict[str, Path] = {}
    for filename in COPY_FILES:
        path = srcdir / filename
        require(path.is_file(), f'MISSING_APPROVED_FIGURE: {path}')
        digest = sha256(path)
        require(digest.startswith(str(pins[filename]).lower()),
                f'APPROVED_FIGURE_SHA_MISMATCH: {filename}: expected={pins[filename]} actual={digest[:8]}')
        files[filename] = path
    return files


def set_style(cfg: dict):
    plt.rcParams.update({
        'font.family': cfg['figure']['font'],
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'svg.fonttype': 'none',  # retains editable text labels if font available
    })


def save_figure(fig, outdir: Path, name: str, cfg: dict, extra_artists=None):
    options = dict(bbox_inches='tight', pad_inches=float(cfg['figure']['pad_inches']))
    if extra_artists is not None:
        options['bbox_extra_artists'] = list(extra_artists)
    fig.savefig(outdir / f'{name}.png', dpi=int(cfg['figure']['dpi']), **options)
    fig.savefig(outdir / f'{name}.svg', **options)
    plt.close(fig)
    with Image.open(outdir / f'{name}.png') as img:
        size = img.size
    require(size[0] >= 2000, f'{name} width below 2000 px: {size}')
    require((outdir / f'{name}.svg').stat().st_size > 300, f'{name} SVG empty')
    return size


def wrapped(text: str, width: int = 26) -> str:
    """Wrap each explicit paragraph; no line may exceed 26 characters."""
    lines = []
    for para in text.split('\n'):
        if not para.strip():
            lines.append('')
            continue
        lines.extend(textwrap.wrap(para, width=width, break_long_words=False, break_on_hyphens=False))
    require(all(len(line) <= width for line in lines), 'F1 line exceeds 26 characters')
    return '\n'.join(lines)


def validate_f1_source(source: pd.DataFrame):
    """Match Director-approved F1 count labels to the frozen T06b plot rows.

    The sensitivity positive count of 40 comes from the Director's directive,
    since the original F1 plotting CSV does not contain this sensitivity box.
    """
    require(len(source) == 5 and {'stage', 'label'}.issubset(source.columns),
            'F1 frozen plot_data must contain five flow rows')
    require(list(source['stage'].astype(int)) == list(range(5)),
            'F1 frozen flow stage identifiers changed')
    expected_counts = [
        ('552',), ('34', '10', '24'), ('518', '30'),
        ('10', '5'), ('5224', '499')
    ]
    for stage, required_numbers in enumerate(expected_counts):
        text = str(source.iloc[stage]['label']).replace(',', '')
        for value in required_numbers:
            require(re.search(r'(?<!\d)' + value + r'(?!\d)', text) is not None,
                    f'F1 frozen stage {stage} is missing the approved count {value}')


def draw_f1(cfg: dict, outdir: Path, source: pd.DataFrame):
    validate_f1_source(source)
    spec = cfg['figure']
    fig, ax = plt.subplots(figsize=(float(spec['F1_width_in']), float(spec['F1_height_in'])))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')


    boxes = [
        ('A', .40, .92, 'CTU-UHB recordings\nn = 552'),
        ('B', .40, .62, 'Analysis cohort\nn = 518 (pH < 7.05: 30)'),
        ('E', .78, .77,
         'Excluded: > 50% FHR signal loss in last 30 min of stage 1\n'
         'n = 34 (acidaemic 10; non-acidaemic 24)'),
        ('S', .78, .92, 'Sensitivity analysis:\nno exclusion, n = 552\n(pH < 7.05: 40)'),
        ('C', .22, .25, 'Strict evaluation\nrepeated nested CV\n(10 × 5 × 5)'),
        ('D', .65, .25, 'Leakage audit\n10-min segments\n5,224 retained; 499 rejected'),
    ]
    artists = {}
    for label, x, y, content in boxes:
        # A long first line is intentionally wrapped more tightly to avoid S.
        if label == 'A':
            content = 'CTU-UHB\nrecordings\nn = 552'
        # Pre-wrap the required sentences with the same textwrap semantics.
        if label == 'S':
            content = 'Sensitivity analysis:\nno exclusion, n = 552\n(pH < 7.05: 40)'
        text = ax.text(x, y, wrapped(content), ha='center', va='center',
                       fontsize=8, linespacing=0.84, zorder=4,
                       bbox=dict(boxstyle='square,pad=0.5', fc='#F2F2F2',
                                 ec='#4D4D4D', lw=0.9))
        if label == 'S':
            text.get_bbox_patch().set_linestyle('--')
        artists[label] = text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Use window extents on the actual renderer, never width/height approximations.
    text_extents = {label: a.get_window_extent(renderer=renderer) for label, a in artists.items()}
    patch_extents = {label: a.get_bbox_patch().get_window_extent(renderer=renderer)
                     for label, a in artists.items()}
    overlapping_text = []
    overlapping_boxes = []
    keys = list(artists)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            if text_extents[a].overlaps(text_extents[b]):
                overlapping_text.append((a, b))
            if patch_extents[a].overlaps(patch_extents[b]):
                overlapping_boxes.append((a, b))
        # text is fully enclosed by its own bounding box
        require(patch_extents[a].contains(*text_extents[a].corners()[0]) and
                patch_extents[a].contains(*text_extents[a].corners()[2]),
                f'F1_TEXT_OUTSIDE_BOX: {a}')
    # Connect at the *rendered* patch edges, so arrowheads are visible and
    # no connector runs through the automatically sized labels.
    axes_inverse = ax.transAxes.inverted()
    edges = {}
    for label, bbox in patch_extents.items():
        x0, y0 = axes_inverse.transform((bbox.x0, bbox.y0))
        x1, y1 = axes_inverse.transform((bbox.x1, bbox.y1))
        edges[label] = (x0, y0, x1, y1)
    def arrow(p0, p1, dashed=False):
        ax.annotate('', xy=p1, xytext=p0, zorder=2,
                    arrowprops=dict(arrowstyle='-|>', lw=.9,
                                    linestyle='--' if dashed else '-',
                                    color='#4D4D4D', shrinkA=0, shrinkB=0))
    def segment(p0, p1):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                c='#4D4D4D', lw=.9, zorder=2)
    arrow((.40, edges['A'][1]), (.40, edges['B'][3]))
    arrow((.40, .77), (edges['E'][0], .77))
    arrow((edges['A'][2], .92), (edges['S'][0], .92), dashed=True)
    segment((.40, edges['B'][1]), (.40, .42))
    segment((.22, .42), (.65, .42))
    arrow((.22, .42), (.22, edges['C'][3]))
    arrow((.65, .42), (.65, edges['D'][3]))

    print(f'F1_TEXT_BBOX_OVERLAP={"PASS" if not overlapping_text else "FAIL"}; pairs={overlapping_text}', flush=True)
    print(f'F1_BOX_BBOX_OVERLAP={"PASS" if not overlapping_boxes else "FAIL"}; pairs={overlapping_boxes}', flush=True)
    require(not overlapping_text and not overlapping_boxes,
            f'F1_COLLISION_DETECTED: text={overlapping_text}; boxes={overlapping_boxes}')

    dimensions = save_figure(fig, outdir, 'F1', cfg, extra_artists=[a.get_bbox_patch() for a in artists.values()])
    return dimensions, {'text_bbox_overlap': 'PASS', 'box_bbox_overlap': 'PASS'}


def _number(value):
    if pd.isna(value) or str(value).strip() == '':
        return None
    val = float(value)
    require(np.isfinite(val), f'Invalid F2 numeric value: {value}')
    return val


def _errorbar(ax, yy: float, val: float, lo, hi, color: str):
    lov, hiv = _number(lo), _number(hi)
    if lov is None and hiv is None:
        return
    require(lov is not None and hiv is not None and lov <= hiv,
            'Invalid frozen CI in F2')
    ax.plot([lov, hiv], [yy, yy], color=color, lw=1.3, solid_capstyle='butt', zorder=2)
    ax.plot([lov, lov], [yy-.065, yy+.065], color=color, lw=.85, zorder=2)
    ax.plot([hiv, hiv], [yy-.065, yy+.065], color=color, lw=.85, zorder=2)


def draw_f2(cfg: dict, outdir: Path, source: pd.DataFrame):
    # Ten frozen rows: six tested rows plus four descriptive rows.
    expected_labels = (
        'L1/M1', 'L1/M2', 'L2/M1', 'L2/M2', 'L3/M1', 'L3/M2',
        'L1 record/M1', 'L1 record/M2',
        'L2 augmented/M1', 'L2 augmented/M2',
    )
    require(len(source) == 10, 'F2 original data must contain exactly 10 rows')
    require(source['label'].is_unique and Counter(source['label']) == Counter(expected_labels),
            'F2 label set differs from frozen expected rows')
    by_label = source.set_index('label', drop=False)
    fig, ax = plt.subplots(figsize=(float(cfg['figure']['F2_width_in']),
                                    float(cfg['figure']['F2_height_in'])))
    fig.subplots_adjust(left=.36, right=.71, top=.97, bottom=.23)
    ax.axvline(.5, color='#999999', lw=.8, ls=':', zorder=0)
    main_order = expected_labels[:6]
    yticks = np.arange(len(main_order))[::-1]
    names = []
    used_labels = []
    for i, label in enumerate(main_order):
        r = by_label.loc[label]
        yy = float(yticks[i])
        model = label[-2:]
        protocol = label[:2]
        model_name = MODELS[model]
        protocol_name = PROTOCOLS[label] if protocol == 'L3' else PROTOCOLS[protocol]
        names.append(f'{protocol_name}\n{model_name}')

        s = _number(r['S']); l = _number(r['L'])
        require(s is not None and l is not None, f'Missing F2 values {label}')
        _errorbar(ax, yy, s, r['S_low'], r['S_high'], STYLE['strict'])
        _errorbar(ax, yy, l, r['L_low'], r['L_high'], STYLE['leaky'])
        ax.annotate('', xy=(l,yy), xytext=(s,yy),
                    arrowprops={'arrowstyle': '->', 'color': '#888888', 'lw': .65}, zorder=1)
        ax.plot(s, yy, marker='s', ms=4.8, color=STYLE['strict'], ls='none', clip_on=False, zorder=5)
        ax.plot(l, yy, marker='o', ms=4.8, color=STYLE['leaky'], ls='none', clip_on=False, zorder=5)
        require(_number(r['delta']) is not None and _number(r['p_holm']) is not None,
                f'F2 inferential row missing delta/Holm p: {label}')
        ax.text(1.042, yy, f"Δ = {float(r['delta']):.3f}; Holm p = {float(r['p_holm']):.4f}",
                fontsize=7.5, ha='left', va='center', clip_on=False)
        used_labels.append(label)

        # Descriptive point on same protocol/model row, vertical offset +0.18.
        descriptive_name = None
        status = None
        if protocol == 'L1':
            descriptive_name = f'L1 record/{model}'
            status = 'record'
        elif protocol == 'L2':
            descriptive_name = f'L2 augmented/{model}'
            status = 'synthetic'
        if descriptive_name is not None:
            rd = by_label.loc[descriptive_name]
            y_desc = yy + 0.18
            sd = _number(rd['S']); ld = _number(rd['L'])
            require(sd is not None and ld is not None, f'Descriptive F2 missing values: {descriptive_name}')
            _errorbar(ax, y_desc, sd, rd['S_low'], rd['S_high'], STYLE['strict'])
            _errorbar(ax, y_desc, ld, rd['L_low'], rd['L_high'], STYLE[status])
            ax.plot(sd, y_desc, marker='s', ms=4.3, color=STYLE['strict'],
                    ls='none', clip_on=False, zorder=5)
            ax.plot(ld, y_desc, marker='^', ms=5.5, color=STYLE[status],
                    ls='none', clip_on=False, zorder=6)
            ax.text(1.042, y_desc+0.13, 'descriptive', fontsize=7.1,
                    ha='left', va='center', color=STYLE[status], clip_on=False)
            used_labels.append(descriptive_name)

    require(Counter(used_labels) == Counter(expected_labels),
            f'F2_PLOT_ROW_COVERAGE_FAIL: {used_labels}')
    # Group separators AFTER row 2 and AFTER row 4, not after descriptive overlays.
    for yy in (3.5, 1.5):
        ax.axhline(yy, c='#D9D9D9', lw=.65, zorder=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(names, ha='right')
    ax.tick_params(axis='y', pad=6, length=0)
    ax.set_xlim(.4, 1.02)
    ax.set_ylim(-.55, 5.55)
    ax.set_xlabel('AUROC (95% CI)')
    handles = [
        Line2D([0],[0],marker='s',ls='none',color=STYLE['strict'],label='Strict (record-level)'),
        Line2D([0],[0],marker='o',ls='none',color=STYLE['leaky'],label='Leaky (tested segment/row level)'),
        Line2D([0],[0],marker='^',ls='none',color=STYLE['record'],label='Record-level (descriptive)'),
        Line2D([0],[0],marker='^',ls='none',color=STYLE['synthetic'],
               label='Test including synthetic/duplicated samples (descriptive)'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.50,.02),
               frameon=False, ncol=2, fontsize=7.5, columnspacing=1.0, handletextpad=.4)
    size = save_figure(fig, outdir, 'F2', cfg)
    return size, {'frozen_rows_plotted': len(used_labels), 'x_limits': [0.4,1.02],
                  'separators': [3.5, 1.5], 'leaky_color': STYLE['leaky']}


def output_manifest(outdir: Path, root: Path):
    manifest = outdir / 'OUTPUT_SHA256.txt'
    require(not manifest.exists(), 'OUTPUT_SHA256 already exists')
    paths = sorted(p for p in outdir.iterdir() if p.is_file())
    manifest.write_text(''.join(f'{sha256(p)}  {rel(p,root)}\n' for p in paths), encoding='utf-8')
    return sha256(manifest)[:8], len(paths)


def execute(config: dict, stage: str):
    require(config.get('task') == 'P21-T06d' and config.get('prereg_sha256') == PREREG,
            'T06d configuration/PREREG mismatch')
    root = Path(config['work_root'])
    require(root.is_dir(), f'MISSING_ROOT: {root}')
    destination = root / config['output_dir']
    require(not destination.exists(), f'STOP_OUTPUT_EXISTS: {destination}')
    frozen_files, frozen_sha = frozen_plot_data_preflight(root, config)
    copied_files = approved_copy_preflight(root, config)
    print('T06D_INPUT_PREFLIGHT_PASS frozen_plots=5 copied_approved_figures=6', flush=True)
    if stage == 'inspect':
        for key, val in frozen_sha.items():
            print(f'{key}: {val} PASS')
        print('T06D_INSPECT_PASS; no files written')
        return 0

    # Validate both re-rendered figures' frozen rows before creating any output.
    source_f1 = pd.read_csv(frozen_files['F1_plot_data.csv'])
    source_f2 = pd.read_csv(frozen_files['F2_plot_data.csv'])
    validate_f1_source(source_f1)
    expected_f2 = ('L1/M1', 'L1/M2', 'L2/M1', 'L2/M2', 'L3/M1', 'L3/M2',
                   'L1 record/M1', 'L1 record/M2', 'L2 augmented/M1', 'L2 augmented/M2')
    require(len(source_f2) == 10 and
            Counter(source_f2['label']) == Counter(expected_f2),
            'F2 original plot_data row set or count changed')
    set_style(config)
    destination.mkdir(parents=True, exist_ok=False)
    f1, bbox_report = draw_f1(config, destination, source_f1)
    f2, f2_report = draw_f2(config, destination, source_f2)
    sizes = {'F1': f1, 'F2': f2}
    for name, src in copied_files.items():
        dst = destination / name
        shutil.copy2(src, dst)
        require(sha256(src) == sha256(dst), f'BYTEWISE_COPY_MISMATCH: {name}')
    for n in ('F3', 'F4', 'F5'):
        with Image.open(destination / f'{n}.png') as im:
            sizes[n] = im.size

    # All five figure CSVs are byte-identical to frozen T06b inputs.
    for name, src in frozen_files.items():
        dst = destination / name
        shutil.copy2(src, dst)
        require(sha256(src) == sha256(dst), f'PLOT_DATA_COPY_MISMATCH: {name}')

    lines = [
        '=== REPORT P21-T06d ===',
        'status=PASS; CPU-only; presentation-only; no-new-numbers; no-new-rows',
        f'PREREG_v1.3_SHA256={PREREG}',
        f'plot_data_frozen_SHA=PASS; files={len(frozen_files)}; manifest={config["frozen_manifest"]}',
        f'F1_text_bbox_overlap={bbox_report["text_bbox_overlap"]}; F1_box_bbox_overlap={bbox_report["box_bbox_overlap"]}',
        f'F2_source_rows_plotted={f2_report["frozen_rows_plotted"]}/10; xlim=0.4..1.02',
        'F3_F5_bytewise_copy=PASS; approved_T06c_SHA=PASS',
    ]
    for n in ('F1','F2','F3','F4','F5'):
        w,h = sizes[n]
        lines.append(f'{n}: {w}x{h}px; png_sha8={sha256(destination/(n+".png"))[:8]}; svg_sha8={sha256(destination/(n+".svg"))[:8]}')
    lines.append('=== END ===')
    require(len(lines) <= 15, 'REPORT_EXCEEDS_15_LINES')
    (destination / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest_sha, manifest_count = output_manifest(destination, root)
    print('\n'.join(lines))
    print(f'T06D_OUTPUT_MANIFEST_SHA8={manifest_sha}; files={manifest_count}', flush=True)
    return 0


def main(argv=None):
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', required=True)
    argparser.add_argument('--stage', choices=('inspect', 'run'), default='run')
    args = argparser.parse_args(argv)
    try:
        config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
        started = time.perf_counter()
        result = execute(config, args.stage)
        if args.stage == 'run':
            print(f'runtime_s={time.perf_counter()-started:.2f}')
        print('P21_T06D_EXIT_CODE=0')
        return result
    except (ValueError, OSError, KeyError, TypeError, IndexError, AssertionError) as exc:
        print(f'P21_T06D_STOP: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('P21_T06D_EXIT_CODE=2', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

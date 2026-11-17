"""P21-T06d F1fix: layout-only regeneration of F1 from SHA-locked T06b CSV.

Creates only F1.png/F1.svg in a new directory. Never writes T06b, T06c, T06d.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import yaml

PREREG = '40573556ecf4664c430430eb99eaaa20d1d0958aa32647cd30b7144434d048eb'
BOX_TEXT = (
    ('A', .40, .92, 'CTU-UHB\nrecordings\nn = 552'),
    ('B', .40, .62, 'Analysis cohort\nn = 518 (pH < 7.05: 30)'),
    ('E', .78, .77, 'Excluded: > 50% FHR signal loss in last 30 min of stage 1\nn = 34 (acidaemic 10; non-acidaemic 24)'),
    ('S', .78, .92, 'Sensitivity analysis:\nno exclusion, n = 552\n(pH < 7.05: 40)'),
    ('C', .22, .30, 'Strict evaluation\nrepeated nested CV\n(10 × 5 × 5)'),
    ('D', .65, .30, 'Leakage audit\n10-min segments\n5,224 retained; 499 rejected'),
)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha256(p: Path):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def parse_locks(path):
    require(path.is_file(), f'MISSING_FROZEN_MANIFEST: {path}')
    entries = {}
    for ix, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = re.fullmatch(r'([0-9a-fA-F]{8}|[0-9a-fA-F]{64})\s+\*?(.+?)\s*', line)
        require(m is not None, f'INVALID_FROZEN_SHA_LINE_{ix}')
        digest = m.group(1).lower()
        name = m.group(2).strip().replace('\\', '/').removeprefix('./')
        require(name not in entries or entries[name] == digest, f'CONFLICTING_FROZEN_SHA: {name}')
        entries[name] = digest
    return entries


def wrapped(text, width=26):
    lines = []
    for para in text.split('\n'):
        lines.extend(textwrap.wrap(para, width=width, break_long_words=False, break_on_hyphens=False) or [''])
    require(all(len(x) <= width for x in lines), 'F1_LINES_TOO_LONG')
    return '\n'.join(lines)


def validate_frozen_f1(src):
    require(len(src) == 5 and {'stage','label'}.issubset(src.columns), 'F1_COHORT_ROWS_INVALID')
    require(src['stage'].astype(int).tolist() == list(range(5)), 'F1_COHORT_STAGES_INVALID')
    nums = [('552',), ('34','10','24'), ('518','30'), ('10','5'), ('5224','499')]
    for i, items in enumerate(nums):
        entry = str(src.iloc[i]['label']).replace(',', '')
        for n in items:
            require(re.search(r'(?<!\d)' + n + r'(?!\d)', entry) is not None,
                    f'F1_FROZEN_DATA_MISMATCH: stage {i}, n={n}')


def input_preflight(cfg):
    root = Path(cfg['work_root'])
    require(root.is_dir(), f'MISSING_ROOT: {root}')
    expected_out = root/'P21_CTG_Acidemia/outputs/t06d_figures_F1fix'
    require(root/cfg['output_dir'] == expected_out, 'UNAUTHORIZED_OUTPUT_PATH')
    require(not expected_out.exists(), f'STOP_OUTPUT_EXISTS: {expected_out}')
    manifest = parse_locks(root/cfg['frozen_manifest'])
    path = root/cfg['frozen_F1_plot_data']
    require(path.is_file(), f'MISSING_FROZEN_F1_PLOT_DATA: {path}')
    key = path.resolve().relative_to(root.resolve()).as_posix()
    digest = manifest.get(key) or manifest.get(key.removeprefix('P21_CTG_Acidemia/'))
    require(digest is not None, f'MISSING_FROZEN_SHA: {key}')
    actual = sha256(path)
    require(actual.startswith(digest), f'FROZEN_SHA_MISMATCH {key}: locked={digest} actual={actual[:8]}')
    for name, expect in cfg['prior_t06d_F1_sha8'].items():
        require(name in ('F1.png','F1.svg'), 'INVALID_T06D_BASELINE_SHA_ENTRY')
        prior = root/cfg['prior_t06d_F1_dir']/name
        require(prior.is_file(), f'MISSING_T06D_BASELINE: {prior}')
        got = sha256(prior)
        require(got.startswith(str(expect).lower()), f'T06D_BASELINE_SHA_MISMATCH {name}: {got[:8]}')
    source = pd.read_csv(path)
    validate_frozen_f1(source)
    return expected_out, actual[:8]


def _intersection(a, b, tol=0.5):
    """Overlap with positive horizontal/vertical area > tolerance in pixels."""
    return (min(a.x1,b.x1)-max(a.x0,b.x0) > tol and
            min(a.y1,b.y1)-max(a.y0,b.y0) > tol)


def create_f1(cfg):
    fcfg = cfg['figure']
    require(fcfg['font'] == 'Arial' and fcfg['linespacing'] == 1.35 and
            fcfg['box_pad'] == 0.6, 'F1_STYLE_CONTRACT_MISMATCH')
    require(fcfg['y_lower_box'] == 0.30 and fcfg['y_branch'] == 0.45 and
            list(fcfg['ylim']) == [0.18, 1.0], 'F1_GEOMETRY_CONTRACT_MISMATCH')
    plt.rcParams.update({'font.family': fcfg['font'], 'font.size': 8,
                         'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(float(fcfg['width_in']), float(fcfg['height_in'])))
    fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.985)
    ax.set_xlim(0, 1)
    ax.set_ylim(*fcfg['ylim'])
    ax.axis('off')

    text_artists = {}
    for label, x, y, content in BOX_TEXT:
        artist = ax.text(x, y, wrapped(content), ha='center', va='center', fontsize=8,
                         linespacing=float(fcfg['linespacing']), zorder=4,
                         bbox=dict(boxstyle='square,pad=0.6', fc='#F2F2F2',
                                   ec='#4D4D4D', lw=.9))
        if label == 'S':
            artist.get_bbox_patch().set_linestyle('--')
        text_artists[label] = artist

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = {k:a.get_window_extent(renderer=renderer) for k,a in text_artists.items()}
    boxes = {k:a.get_bbox_patch().get_window_extent(renderer=renderer) for k,a in text_artists.items()}
    keys = list(text_artists)
    overlap_text, overlap_boxes = [], []
    for i, k in enumerate(keys):
        for q in keys[i+1:]:
            if _intersection(texts[k], texts[q]): overlap_text.append((k,q))
            if _intersection(boxes[k], boxes[q]): overlap_boxes.append((k,q))
        if not boxes[k].contains(*texts[k].corners()[0]) or not boxes[k].contains(*texts[k].corners()[2]):
            raise ValueError(f'F1_TEXT_OUTSIDE_BOX: {k}')

    # Verify characters are also separated vertically within each multiline text block.
    # This check is renderer-specific; leading 1.35 means positive extra interline space.
    require(all(abs(getattr(a, "_linespacing", None)-1.35) < 1e-10 for a in text_artists.values()),
            'F1_LINESPACING_NOT_APPLIED')
    print(f'F1_TEXT_BBOX_OVERLAP={"PASS" if not overlap_text else "FAIL"}; pairs={overlap_text}', flush=True)
    print(f'F1_BOX_BBOX_OVERLAP={"PASS" if not overlap_boxes else "FAIL"}; pairs={overlap_boxes}', flush=True)
    require(not overlap_text and not overlap_boxes,
            f'F1_COLLISION: text={overlap_text}, boxes={overlap_boxes}')

    inv = ax.transData.inverted()
    edges = {}
    for label,b in boxes.items():
        x0,y0 = inv.transform((b.x0,b.y0))
        x1,y1 = inv.transform((b.x1,b.y1))
        edges[label] = (x0,y0,x1,y1)
    def arrow(a,b,dashed=False):
        ax.annotate('',xy=b,xytext=a,zorder=2,
                    arrowprops=dict(arrowstyle='-|>',lw=.9,
                                    linestyle='--' if dashed else '-',
                                    color='#4D4D4D',shrinkA=0,shrinkB=0))
    def seg(a,b):
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#4D4D4D',lw=.9,zorder=2)
    arrow((.40,edges['A'][1]),(.40,edges['B'][3]))
    arrow((.40,.77),(edges['E'][0],.77))
    arrow((edges['A'][2],.92),(edges['S'][0],.92),dashed=True)
    seg((.40,edges['B'][1]),(.40,.45))
    seg((.22,.45),(.65,.45))
    arrow((.22,.45),(.22,edges['C'][3]))
    arrow((.65,.45),(.65,edges['D'][3]))

    return fig, [a.get_bbox_patch() for a in text_artists.values()]


def run(cfg, stage):
    require(cfg['task'] == 'P21-T06d-F1fix' and cfg['prereg_sha256'] == PREREG,
            'TASK_OR_PREREG_MISMATCH')
    target, inputsha = input_preflight(cfg)
    print(f'F1FIX_INPUT_SHA_PASS F1_plot_data.csv={inputsha}; T06D_BASELINE_SHA=PASS',flush=True)
    if stage == 'inspect':
        print('F1FIX_INSPECT_PASS; no output created')
        return 0
    fig, extra = create_f1(cfg)  # verify bbox before touching filesystem
    target.mkdir(parents=True, exist_ok=False)
    try:
        options={'bbox_inches':'tight','pad_inches':.05,'bbox_extra_artists':extra}
        fig.savefig(target/'F1.png',dpi=600,**options)
        fig.savefig(target/'F1.svg',**options)
    finally:
        plt.close(fig)
    files=sorted(target.iterdir())
    require([p.name for p in files] == ['F1.png','F1.svg'], 'OUTPUT_MUST_CONTAIN_ONLY_F1_PNG_SVG')
    with Image.open(target/'F1.png') as im:
        w,h=im.size
        require(im.info.get('dpi') and abs(im.info['dpi'][0]-600)<1,'F1_PNG_DPI_INVALID')
    require(w>=2000, f'F1_WIDTH_TOO_SMALL: {w}')
    print('=== REPORT P21-T06d-F1fix ===')
    print('status=PASS; layout-only; sources-unchanged; outputs=F1.png,F1.svg')
    print(f'F1_plot_data_frozen_SHA=PASS; sha8={inputsha}')
    print('F1_text_bbox_overlap=PASS; F1_box_bbox_overlap=PASS')
    print('linespacing=1.35; box_pad=0.6; y_lower=0.30; branch_y=0.45; ylim=0.18..1.0')
    print(f'F1.png: {w}x{h}px; sha8={sha256(target/"F1.png")[:8]}')
    print(f'F1.svg: sha8={sha256(target/"F1.svg")[:8]}')
    print('=== END ===')
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=('inspect','run'),default='run')
    args=parser.parse_args(argv)
    started=time.perf_counter()
    try:
        cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
        code=run(cfg,args.stage)
        print(f'runtime_s={time.perf_counter()-started:.2f}')
        print('P21_T06D_F1FIX_EXIT_CODE=0')
        return code
    except (ValueError, OSError, KeyError, TypeError, IndexError, AssertionError) as e:
        print(f'P21_T06D_F1FIX_STOP: {type(e).__name__}: {e}', file=sys.stderr)
        print('P21_T06D_F1FIX_EXIT_CODE=2', file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())

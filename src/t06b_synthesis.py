"""P21-T06b: immutable-source, no-training scientific synthesis.

Only T1 and figure plotting coordinates/Youden are newly computed.
T2-T5 numerical inputs are copied verbatim from locked outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from decimal import Decimal

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve

PREREG = '40573556ecf4664c430430eb99eaaa20d1d0958aa32647cd30b7144434d048eb'
CONFIGS = ('A_primary', 'B_noexcl', 'C_w60', 'D_y715', 'E_y720')
COLORS = {'M0': '#000000', 'M1': '#0072B2', 'M2': '#E69F00',
          'M3': '#009E73', 'S': '#0072B2', 'L': '#D55E00', 'aug': '#CC79A7'}
FIG_SIZES = {'F1': (3.5, 5.0), 'F2': (7.0, 5.7), 'F3': (7.0, 3.2),
             'F4': (3.5, 2.8), 'F5': (3.5, 2.8)}


def require(test, reason):
    if not test:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rel(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def parse_lock_lines(text, base):
    """Parse sha8/sha256 + path; ignore unrelated report fields, never infer a hash."""
    items = {}
    for line in text.splitlines():
        line = line.strip().lstrip('\ufeff')
        if not line or line.startswith('#'):
            continue
        # sha256sum and T04_FROZEN_SHA formats, including optional binary marker *.
        m = re.fullmatch(r'([0-9a-fA-F]{8}|[0-9a-fA-F]{64})\s+\*?(.+?)\s*', line)
        if not m:
            # Also accept path,hash csv form without additional fields.
            q = next(csv.reader([line])) if ',' in line else []
            if len(q) == 2 and re.fullmatch(r'[0-9a-fA-F]{8}|[0-9a-fA-F]{64}', q[1].strip()):
                name, digest = q[0].strip(), q[1].strip().lower()
            else:
                continue
        else:
            digest, name = m.group(1).lower(), m.group(2).strip()
        name = name.replace('\\', '/').removeprefix('./')
        # Lock paths can be project-relative or root-relative.
        if name.startswith('P21_CTG_Acidemia/'):
            name = name[len('P21_CTG_Acidemia/'):]
        if name.lower().startswith('d:/tuan/ai_training/'):
            name = name[len('D:/Tuan/AI_training/'):]
        # Preserve both project-relative and workroot-relative aliases.
        keys = {name}
        if name.startswith('data_proc/'):
            keys.add(name)
        else:
            keys.add('P21_CTG_Acidemia/' + name)
        for key in keys:
            existing = items.get(key)
            require(existing is None or existing.startswith(digest) or digest.startswith(existing),
                    f'Conflicting frozen SHA for {key}')
            items[key] = max([x for x in (existing, digest) if x], key=len)
    return items


def verify_locked_inputs(root, source_paths):
    """Fail closed if any read source has no unambiguous frozen SHA record."""
    report_dir = root / 'P21_CTG_Acidemia' / 'reports'
    locks = sorted(report_dir.glob('*_FROZEN_SHA.txt'))
    require(bool(locks), f'No *_FROZEN_SHA.txt in {report_dir}')
    declarations = {}
    for path in locks:
        for name, digest in parse_lock_lines(path.read_text(encoding='utf-8-sig'), root).items():
            if name in declarations:
                require(declarations[name].startswith(digest) or digest.startswith(declarations[name]),
                        f'Conflicting lock files for {name}')
                digest = max(declarations[name], digest, key=len)
            declarations[name] = digest
    report = {}
    for path in sorted({Path(p).resolve() for p in source_paths}):
        require(path.is_file(), f'Missing input: {path}')
        name = rel(path, root)
        key = name.removeprefix('P21_CTG_Acidemia/')
        expected = declarations.get(name, declarations.get(key))
        require(expected is not None, f'MISSING_FROZEN_SHA for {name}; request Director lock; DO NOT self-freeze')
        current = sha(path)
        require(current.startswith(expected),
                f'FROZEN_SHA_MISMATCH {name}: locked={expected} actual={current[:8]}')
        report[name] = current[:8]
    return report, [rel(p, root) for p in locks]


def input_paths(root):
    base = root / 'P21_CTG_Acidemia'
    derived = root / 'data_proc/ctu-uhb-ctgdb/1.0.0'
    paths = [derived / name for name in ('labels.parquet', 'clinical.parquet',
                                          'signal_qc.parquet', 'features_w30.parquet')]
    for config in CONFIGS:
        paths.extend((base / f'outputs/t03_baselines/{config}/metrics.json',
                      base / f'outputs/t04_dl/{config}/metrics.json'))
    paths.extend((base / 'outputs/t03_baselines/A_primary/oof.parquet',
                  base / 'outputs/t04_dl/A_primary/oof.parquet',
                  base / 'outputs/t05_leakage/h2_tests.csv',
                  base / 'outputs/t05_leakage/segments_meta.parquet',
                  base / 'outputs/t06a_stats/h1_tests.csv',
                  base / 'outputs/t06a_stats/sensitivity.csv',
                  base / 'outputs/t06a_stats/exploratory.csv'))
    return paths


def scalar(v):
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        require(np.isfinite(v), 'Missing/nonfinite result')
        return repr(float(v))
    return str(v)


def fmt(v, digits=3):
    try:
        return f'{float(v):.{digits}f}'
    except (ValueError, TypeError):
        return str(v)


def numeric(v):
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def suppress_count(n):
    return '<10' if 1 <= int(n) <= 9 else str(int(n))


def stats_triplet(x):
    vals = pd.to_numeric(x, errors='coerce').dropna().to_numpy(dtype=float)
    if not len(vals):
        return (None, None, None)
    q1, md, q3 = np.percentile(vals, [25, 50, 75])
    return (float(md), float(q1), float(q3))


def format_triplet(triplet):
    md, q1, q3 = triplet
    return 'not reported' if md is None else f'{md:.4f} [{q1:.4f}, {q3:.4f}]'


def trace_add(trace, table, row, column, value, source_file, source_field, root, derived=False):
    if not numeric(value):
        return
    trace.append({'table': table, 'row': str(row), 'column': column,
                  'value': scalar(value), 'source_file': rel(source_file, root),
                  'source_sha8': None if derived else sha(source_file)[:8],
                  'source_field': source_field, '_derived': bool(derived)})


def make_t1(root, trace):
    d = root / 'data_proc/ctu-uhb-ctgdb/1.0.0'
    labels = pd.read_parquet(d / 'labels.parquet')
    clinical = pd.read_parquet(d / 'clinical.parquet')
    qc = pd.read_parquet(d / 'signal_qc.parquet')
    require(len(labels) == 552 and labels.record_id.is_unique, 'T1 labels n or key invalid')
    require(clinical.record_name.is_unique and qc.record_name.is_unique, 'T1 clinical/QC key invalid')
    require(set(labels.record_id)==set(clinical.record_name)==set(qc.record_name),
            'T1 record identities differ between T01/T02 tables')
    df = labels.merge(clinical, left_on='record_id', right_on='record_name', validate='one_to_one')
    require(len(df) == 552 and df.record_id.is_unique, 'T1 clinical join incomplete')
    require('pH' in df and 'BDecf' in df, 'T1 clinical columns pH/BDecf missing')
    require(int(df.excluded.sum()) == 34 and int((~df.excluded).sum()) == 518,
            'T1 excluded/eligible count mismatch')
    groups = [('All_552', df), ('Excluded_34', df[df.excluded]),
              ('Eligible_518', df[~df.excluded]),
              ('Eligible_y705_0', df[(~df.excluded) & (df.y705 == 0)]),
              ('Eligible_y705_1', df[(~df.excluded) & (df.y705 == 1)])]
    rows = []
    for group, sub in groups:
        n = len(sub)
        no_stage2 = int(sub.no_stage2.sum())
        pos = int(sub.y705.sum())
        pht = stats_triplet(sub['pH'])
        bdt = stats_triplet(sub['BDecf'])
        mst = stats_triplet(sub.fhr_nan_pct_w30)
        row = {'cohort': group, 'n': suppress_count(n), 'positive_y705_n': suppress_count(pos),
               'pH_median_IQR': format_triplet(pht),
               'BDecf_median_IQR': format_triplet(bdt),
               'no_stage2_n': suppress_count(no_stage2),
               'no_stage2_percent': f'{no_stage2/n*100:.4f}' if n and (no_stage2 == 0 or no_stage2 >= 10) else '<10 suppressed',
               'fhr_nan_pct_w30_median_IQR': format_triplet(mst)}
        for name,values in (('pH',pht),('BDecf',bdt),('fhr_nan_pct_w30',mst)):
            for suffix,value in zip(('median','q1','q3'),values):
                row[name+'_'+suffix] = '' if value is None else scalar(value)
        rows.append(row)
    return pd.DataFrame(rows)


def extract_metric_node(name, model, t03, t04):
    if model in ('M0', 'M1'):
        m = t03[model]
        return m['mean_probability_metrics'], m, 'mean_probability_metrics'
    m = t04['models']['M2UC' if model == 'M2_UC' else model]
    return m['mean_probability_over_repeats_and_seeds'], m['mean_probability_over_repeats_and_seeds'], 'models.' + ('M2UC' if model == 'M2_UC' else model) + '.mean_probability_over_repeats_and_seeds'


def make_t2(root, trace):
    base = root / 'P21_CTG_Acidemia' / 'outputs'
    rows = []
    fields = [('auroc', 'AUROC'), ('auroc_ci95', 'AUROC_CI95'),
              ('auprc_average_precision', 'AUPRC'), ('auprc_ci95', 'AUPRC_CI95'),
              ('brier', 'Brier'), ('calibration_intercept_offset_logit', 'Cal_intercept'),
              ('calibration_slope', 'Cal_slope'),
              ('sensitivity_at_specificity_0_85', 'Se_at_Sp85')]
    for name in CONFIGS:
        p3 = base / 't03_baselines' / name / 'metrics.json'
        p4 = base / 't04_dl' / name / 'metrics.json'
        t03 = json.loads(p3.read_text(encoding='utf-8'))
        t04 = json.loads(p4.read_text(encoding='utf-8'))
        for model in ('M0', 'M1', 'M2', 'M3') + (('M2_UC',) if name == 'A_primary' else ()):
            node, ci_node, prefix = extract_metric_node(name, model, t03, t04)
            source = p3 if model in ('M0', 'M1') else p4
            keybase = f'{model}.{prefix}' if model in ('M0', 'M1') else prefix
            row_id = name + '/' + model
            row = {'config': name, 'model': model}
            for key, col in fields:
                if key.endswith('ci95'):
                    pair = ci_node[key]
                    require(len(pair) == 2, 'T2 invalid CI pair')
                    row[col + '_low'], row[col + '_high'] = map(scalar, pair)
                    for i, end in enumerate(('low', 'high')):
                        trace_add(trace, 'T2', row_id, col + '_' + end, pair[i], source,
                                  f'{keybase if model not in ("M0","M1") else model}.{key}[{i}]', root)
                else:
                    value = node[key]
                    require(value is not None and numeric(value), f'T2 null metric {row_id}.{key}')
                    row[col] = scalar(value)
                    trace_add(trace, 'T2', row_id, col, value, source,
                              f'{keybase}.{key}', root)
            rows.append(row)
    require(len(rows) == 21, 'T2 expected 21 config/model rows')
    return pd.DataFrame(rows)


def copy_csv_with_trace(root, filename, table, trace, row_keys):
    p = root / 'P21_CTG_Acidemia' / 'outputs' / filename
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    require(len(df) > 0 and df.columns.is_unique, f'{table} empty/duplicate columns')
    for i, record in df.iterrows():
        rid = '/'.join(str(record[k]) for k in row_keys)
        for col, value in record.items():
            trace_add(trace, table, rid, col, value, p, col, root)
    return df


def make_t3(root, trace):
    a = copy_csv_with_trace(root, 't06a_stats/h1_tests.csv', 'T3', trace, ['config', 'comparison'])
    a.insert(0, 'section', 'confirmatory_H1')
    b = copy_csv_with_trace(root, 't06a_stats/sensitivity.csv', 'T3', trace, ['config', 'comparison'])
    b.insert(0, 'section', 'sensitivity_no_Holm')
    result = pd.concat((a, b), ignore_index=True).fillna('')
    require(len(result) == 10, 'T3 expected 10 comparisons')
    return result


def make_t4(root, trace):
    p = root / 'P21_CTG_Acidemia' / 'outputs/t05_leakage/h2_tests.csv'
    raw = pd.read_csv(p, dtype=str, keep_default_na=False)
    require(len(raw) == 8, 'T4 expected eight frozen comparisons')
    rows = []
    for _, source_row in raw.iterrows():
        proto, model = source_row['protocol'], source_row['model']
        rid = f'{proto}/{model}'
        details = json.loads(source_row['details'])
        row = {key: source_row[key] for key in raw.columns if key != 'details'}
        for col in ('delta', 'ci95_low', 'ci95_high', 'p_raw', 'p_holm', 'bootstrap_valid'):
            trace_add(trace, 'T4', rid, col, row[col], p, col, root)
        if proto in ('L1', 'L2', 'L3'):
            unit = 'segment' if proto == 'L1' else 'record'
            fields = {'AUROC_S': f'auroc_s_{unit}', 'AUROC_L': f'auroc_l_{unit}'}
            for dest, src in fields.items():
                row[dest] = scalar(details[src])
                trace_add(trace, 'T4', rid, dest, details[src], p, f'details.{src}', root)
            for end in ('s', 'l'):
                pair = details[f'ci95_{end}_{unit}']
                for i, side in enumerate(('low', 'high')):
                    dest = f'AUROC_{end.upper()}_CI95_{side}'
                    row[dest] = scalar(pair[i])
                    trace_add(trace, 'T4', rid, dest, pair[i], p, f'details.ci95_{end}_{unit}[{i}]', root)
            if proto == 'L1':
                for k in ('auroc_s_record', 'auroc_l_record'):
                    row[k] = scalar(details[k])
                    trace_add(trace, 'T4', rid, k, details[k], p, f'details.{k}', root)
                for end in ('s', 'l'):
                    for i, side in enumerate(('low', 'high')):
                        key = f'ci95_{end}_record_{side}'
                        row[key] = scalar(details[f'ci95_{end}_record'][i])
                        trace_add(trace, 'T4', rid, key, row[key], p, f'details.ci95_{end}_record[{i}]', root)
            if proto == 'L2':
                for dest, field in [('auroc_l_all_test', 'auroc_l_all_test'),
                                    ('delta_l_all_test_descriptive_only', 'delta_l_all_test_descriptive_only')]:
                    row[dest] = scalar(details[field])
                    trace_add(trace, 'T4', rid, dest, details[field], p, f'details.{field}', root)
                row['all_test_note'] = 'descriptive; includes augmented/duplicated test rows'
        else:
            for key in ('mean_fold_Se_S', 'mean_fold_Sp_S', 'mean_fold_Se_L', 'mean_fold_Sp_L'):
                row[key] = scalar(details[key])
                trace_add(trace, 'T4', rid, key, details[key], p, f'details.{key}', root)
            row['Youden_S'] = scalar(float(details['mean_fold_Se_S']) + float(details['mean_fold_Sp_S']) - 1)
            row['Youden_L'] = scalar(float(details['mean_fold_Se_L']) + float(details['mean_fold_Sp_L']) - 1)
            row['Youden_note'] = 'descriptive, post hoc; not tested'
        rows.append(row)
    result = pd.DataFrame(rows).fillna('')
    return result


def make_t5(root, trace):
    df = copy_csv_with_trace(root, 't06a_stats/exploratory.csv', 'T5', trace,
                             ['analysis', 'config', 'group'])
    require(len(df) == 17, 'T5 expected 17 exploratory rows')
    df.insert(0, 'analysis_label', 'exploratory, post hoc')
    return df


def latex_escape(value):
    trans = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
             '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}'}
    return ''.join(trans.get(c, c) for c in str(value))


def emit_table(folder, name, frame):
    path = folder / (name + '.csv')
    frame.to_csv(path, index=False, encoding='utf-8', float_format='%.17g')
    lines = [r'\begin{tabular}{' + 'l' * len(frame.columns) + '}', r'\toprule',
             ' & '.join(map(latex_escape, frame.columns)) + r' \\', r'\midrule']
    for _, row in frame.iterrows():
        lines.append(' & '.join(latex_escape(v) for v in row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (folder / (name + '.tex')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def load_means(root):
    base = root / 'P21_CTG_Acidemia/outputs'
    t3 = pd.read_parquet(base / 't03_baselines/A_primary/oof.parquet')
    t4 = pd.read_parquet(base / 't04_dl/A_primary/oof.parquet')
    require(len(t3) == 5180 and len(t4) == 5180, 'F3/F4 frozen OOF count invalid')
    keys = ['record_id', 'repeat', 'outer_fold', 'y']
    require(not t3[keys].duplicated().any() and not t4[keys].duplicated().any(), 'Frozen OOF duplicates')
    merged = t3[keys + ['p_M0', 'p_M1']].merge(
        t4[keys + [f'p_{m}_s{s}' for m in ('M2','M3') for s in range(3)]],
        on=keys, validate='one_to_one')
    require(len(merged) == 5180, 'Frozen OOF alignment mismatch')
    for model in ('M2', 'M3'):
        merged['p_' + model] = merged[[f'p_{model}_s{s}' for s in range(3)]].mean(axis=1)
    means = merged.groupby('record_id', sort=True).agg(
        y=('y', 'first'), n=('y', 'size'), **{m: (f'p_{m}', 'mean') for m in ('M0','M1','M2','M3')})
    require(len(means) == 518 and means.n.eq(10).all() and means.y.sum() == 30,
            'F3/F4 frozen record and labels mismatch')
    require(np.isfinite(means[['M0','M1','M2','M3']].to_numpy()).all(), 'Nonfinite frozen scores')
    return means


def figure_style():
    plt.rcParams.update({'font.family':'Arial', 'font.size':8, 'axes.labelsize':8,
                         'xtick.labelsize':8, 'ytick.labelsize':8,
                         'legend.fontsize':7, 'axes.titlesize':8,
                         'svg.fonttype':'none', 'savefig.facecolor':'white',
                         'axes.spines.top':False, 'axes.spines.right':False})


def save_figure(fig, folder, name, width):
    fig.set_size_inches(*FIG_SIZES[name])
    fig.savefig(folder / f'{name}.png', dpi=600, bbox_inches=None)
    fig.savefig(folder / f'{name}.svg', bbox_inches=None)
    plt.close(fig)
    from PIL import Image
    with Image.open(folder / f'{name}.png') as im:
        w, h = im.size
    require(w >= 2000 and abs(w - width * 600) <= 3,
            f'{name} image width invalid: {w}')
    require((folder / f'{name}.svg').stat().st_size > 100, f'{name} empty SVG')
    return w, h


def f1_flow(folder, n_seg, dropped):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    blocks = [
        (0.08, .81, .84, .13, 'CTU-UHB recordings\n552'),
        (0.08, .64, .84, .13, 'Excluded by T02 QC\n34 (10 positive; 24 negative)'),
        (0.08, .47, .84, .13, 'Eligible cohort\n518 (30 positive)'),
        (0.05, .19, .43, .16, 'Strict evaluation\n10 × 5 × 5 nested CV'),
        (0.52, .19, .43, .16,
         f'T05 leakage audit\n{n_seg:,} retained\n{dropped:,} rejected segments'),
    ]
    for x, y, w, h, label in blocks:
        rect = Rectangle((x, y), w, h, ec=COLORS['S'], fc='#EEF5FA', lw=1)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=7)
    for hi, lo in zip(blocks[:2], blocks[1:3]):
        ax.annotate('', xy=(.5,lo[1]+lo[3]), xytext=(.5,hi[1]),
                    arrowprops={'arrowstyle':'->','lw':1,'color':'#555555'})
    ax.plot([.5,.5],[.47,.40], color='#555555',lw=1)
    ax.plot([.265,.735],[.40,.40], color='#555555',lw=1)
    for x in (.265,.735):
        ax.annotate('', xy=(x,.35),xytext=(x,.40),
                    arrowprops={'arrowstyle':'->','lw':1,'color':'#555555'})
    pd.DataFrame([{'stage':i, 'label':label, 'x':x,'y':y,'width':w,'height':h}
                  for i,(x,y,w,h,label) in enumerate(blocks)]).to_csv(
                      folder/'F1_plot_data.csv', index=False)
    return save_figure(fig, folder, 'F1', 3.5)


def f2_forest(folder, t4):
    fig, ax = plt.subplots()
    rows = []
    for _, t in t4.iterrows():
        if t.protocol == 'L4':
            continue
        details = [(t.protocol + '/' + t.model, t.AUROC_S, t.AUROC_L,
                    t.AUROC_S_CI95_low, t.AUROC_S_CI95_high,
                    t.AUROC_L_CI95_low, t.AUROC_L_CI95_high,
                    t.delta, t.p_holm, False)]
        if t.protocol == 'L1':
            details.append(('L1 record/' + t.model, t.auroc_s_record, t.auroc_l_record,
                            t.ci95_s_record_low, t.ci95_s_record_high,
                            t.ci95_l_record_low, t.ci95_l_record_high, '', '', True))
        if t.protocol == 'L2':
            details.append(('L2 augmented/' + t.model, t.AUROC_S, t.auroc_l_all_test,
                            '', '', '', '', t.delta_l_all_test_descriptive_only, '', True))
        for label,s,l,slo,shi,llo,lhi,d,p,desc in details:
            rows.append({'label':label, 'S':s, 'L':l, 'S_low':slo,'S_high':shi,
                         'L_low':llo,'L_high':lhi,'delta':d,'p_holm':p,
                         'descriptive':desc})
    df = pd.DataFrame(rows)
    require(len(df) == 10, 'F2 plot must have 10 lines')
    df.to_csv(folder/'F2_plot_data.csv',index=False)
    y = np.arange(len(df))[::-1]
    ax.axvline(.5, color='#888888', ls=':', lw=.8)
    for idx, r in df.iterrows():
        yy = y[idx]
        s, l = float(r.S), float(r.L)
        if numeric(r.S_low):
            ax.plot([float(r.S_low),float(r.S_high)],[yy,yy], c=COLORS['S'],lw=1.25)
        if numeric(r.L_low):
            ax.plot([float(r.L_low),float(r.L_high)],[yy,yy], c=COLORS['L'],lw=1.25)
        ax.annotate('', xy=(l,yy),xytext=(s,yy),
                    arrowprops={'arrowstyle':'->','lw':.7,'color':'#666666'})
        marker = '^' if bool(r.descriptive) else 'o'
        ax.plot(s, yy, marker='s', ms=4, c=COLORS['S'])
        ax.plot(l, yy, marker=marker, ms=4, c=COLORS['aug'] if 'augmented' in r.label else COLORS['L'])
        annot = (f'Δ={fmt(r.delta,3)}; p={fmt(r.p_holm,4)}'
                 if numeric(r.p_holm) else (f'Δ={fmt(r.delta,3)}; descriptive' if 'augmented' in r.label else 'descriptive; not tested'))
        ax.text(1.02,yy,annot,ha='left',va='center',fontsize=7)
    ax.set_yticks(y,df.label)
    ax.set_xlim(.38,1.00)
    ax.set_ylim(-.8,len(df)-.2)
    ax.set_xlabel('AUROC (95% CI)')
    ax.text(.60, len(df)-.05, '■ S', c=COLORS['S'],fontsize=7)
    ax.text(.77, len(df)-.05, '● L', c=COLORS['L'],fontsize=7)
    # Reserve right side for annotation outside of axes.
    fig.subplots_adjust(left=.23,right=.69,bottom=.11,top=.96)
    return save_figure(fig,folder,'F2',7.0)


def f3_roc_calibration(folder, means, t2):
    fig,(ax1,ax2)=plt.subplots(1,2)
    y = means.y.to_numpy(dtype=int)
    plot = []
    for model in ('M0','M1','M2','M3'):
        p = means[model].to_numpy(dtype=float)
        fpr,tpr,_ = roc_curve(y,p,drop_intermediate=False)
        ref = t2.loc[(t2.config=='A_primary') & (t2.model==model)].iloc[0]
        auc = float(ref.AUROC)
        ax1.plot(fpr,tpr,color=COLORS[model],lw=1.4,label=f'{model} (AUROC {auc:.3f})')
        for a,b in zip(fpr,tpr):
            plot.append({'panel':'ROC','model':model,'x':a,'y':b,'n':'','positive_n':'',
                         'source_metric':'frozen OOF ROC coordinates'})
        # Five equal-count quantile groups; no recalibration or refitting.
        groups = pd.qcut(pd.Series(p).rank(method='first'),q=5,labels=False)
        for g in range(5):
            selector = groups.to_numpy()==g
            xp=float(p[selector].mean()); yp=float(y[selector].mean())
            plot.append({'panel':'calibration','model':model,'x':xp,'y':yp,
                         'n':int(selector.sum()),'positive_n':int(y[selector].sum()),
                         'source_metric':'5 quantile bins; original probabilities'})
        subset=[r for r in plot if r['panel']=='calibration' and r['model']==model]
        ax2.plot([r['x'] for r in subset],[r['y'] for r in subset],marker='o',ms=3,
                 color=COLORS[model],lw=1.2,label=model)
    ax1.plot([0,1],[0,1],ls=':',c='#777777',lw=.8)
    ax2.plot([0,1],[0,1],ls=':',c='#777777',lw=.8)
    ax1.set(xlim=(0,1),ylim=(0,1),xlabel='1 − specificity',ylabel='Sensitivity')
    ax2.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed event frequency')
    ax1.legend(loc='lower right',frameon=False)
    ax2.legend(loc='upper left',frameon=False)
    ax1.text(-.15,1.05,'(a)',transform=ax1.transAxes,fontweight='bold')
    ax2.text(-.15,1.05,'(b)',transform=ax2.transAxes,fontweight='bold')
    fig.subplots_adjust(left=.09,right=.98,wspace=.3,bottom=.18,top=.91)
    pd.DataFrame(plot).to_csv(folder/'F3_plot_data.csv',index=False)
    return save_figure(fig,folder,'F3',7.0)


def f4_dca(folder, means):
    y=means.y.to_numpy(dtype=int)
    n=len(y)
    plot=[]
    fig,ax=plt.subplots()
    for th in np.arange(5,31)/100:
        weight=th/(1-th)
        for model in ('M1','M3'):
            p=means[model].to_numpy(dtype=float)
            pred=p>=th
            nb=(int((pred & (y==1)).sum())-int((pred & (y==0)).sum())*weight)/n
            plot.append({'threshold':float(th),'strategy':model,'net_benefit':float(nb)})
        plot.append({'threshold':float(th),'strategy':'treat_all',
                     'net_benefit':float(y.mean()-(1-y.mean())*weight)})
        plot.append({'threshold':float(th),'strategy':'treat_none','net_benefit':0.0})
    df=pd.DataFrame(plot)
    df.to_csv(folder/'F4_plot_data.csv',index=False)
    for model in ('M1','M3','treat_all','treat_none'):
        sub=df.loc[df.strategy==model]
        color=COLORS.get(model,'#777777' if model=='treat_all' else '#000000')
        ax.plot(sub.threshold*100,sub.net_benefit,c=color,lw=1.4,
                ls='--' if model=='treat_all' else (':' if model=='treat_none' else '-'),label=model)
    ax.set(xlabel='Threshold probability (%)',ylabel='Net benefit',xlim=(5,30))
    ax.legend(frameon=False,loc='best')
    fig.subplots_adjust(left=.18,right=.98,bottom=.2,top=.97)
    return save_figure(fig,folder,'F4',3.5)


def f5_e2(folder,t5):
    df=t5.loc[(t5.analysis=='E2') & (~t5.group.str.contains('M3-M1',regex=False))].copy()
    require(len(df)==4,'F5 expected four E2 model/group entries')
    fig,ax=plt.subplots()
    plot=[]
    for k,group in enumerate(('le_median','gt_median')):
        for j,model in enumerate(('M1','M3')):
            r=df.loc[df.group.eq(group+'_'+model)]
            require(len(r)==1,f'F5 missing {group}/{model}')
            r=r.iloc[0]
            x=k+(j-.5)*.27
            lo,hi,est=float(r.ci95_low),float(r.ci95_high),float(r.estimate)
            ax.errorbar([x],[est],yerr=[[est-lo],[hi-est]],fmt='o',ms=4,
                        capsize=3,color=COLORS[model],lw=1.1)
            plot.append({'group':group,'model':model,'x':x,'auroc':est,
                         'ci95_low':lo,'ci95_high':hi,'n':r.n,'positives':r.positives})
    ax.set_xticks([0,1],['≤ median\nmissingness','> median\nmissingness'])
    ax.set_xlim(-.38,1.38)
    ax.set_ylim(0,1)
    ax.set_ylabel('AUROC (95% CI)')
    ax.axhline(.5,linestyle=':',color='#888888',lw=.7)
    for j,g in enumerate(('le_median','gt_median')):
        r=df.loc[df.group.eq(g+'_M1')].iloc[0]
        ax.text(j,.05,f'n={r.n}; positive={r.positives}',ha='center',fontsize=7)
    ax.plot([],[],color=COLORS['M1'],marker='o',ls='',label='M1')
    ax.plot([],[],color=COLORS['M3'],marker='o',ls='',label='M3')
    ax.legend(frameon=False,loc='upper right')
    ax.tick_params(axis='x',labelsize=7)
    fig.subplots_adjust(left=.19,right=.97,bottom=.25,top=.96)
    pd.DataFrame(plot).to_csv(folder/'F5_plot_data.csv',index=False)
    return save_figure(fig,folder,'F5',3.5)


def finish_derived_trace(trace, root):
    for r in trace:
        if r.pop('_derived'):
            source=root/r['source_file']
            require(source.is_file(),'Derived trace source not written: '+str(source))
            r['source_sha8']=sha(source)[:8]
        require(r['source_file'] and r['source_sha8'] and r['source_field'],
                'NUMBERS_TRACE missing source')


def trace_derived_frame(trace, root, table, frame, file, row_field):
    for idx, record in frame.iterrows():
        row = str(record[row_field]) if row_field in frame.columns else str(idx)
        for col,val in record.items():
            trace_add(trace,table,row,col,val,file,col,root,derived=True)


def audit_t2_t5_exact(trace, root):
    """Resolve each traced numeric directly from its locked JSON/CSV field."""
    cache = {}
    count = 0
    for rec in trace:
        if rec['table'] not in ('T2', 'T3', 'T4', 'T5'):
            continue
        path = root / rec['source_file']
        if path not in cache:
            cache[path] = (json.loads(path.read_text(encoding='utf-8'))
                           if path.suffix == '.json' else
                           pd.read_csv(path, dtype=str, keep_default_na=False))
        source = cache[path]
        row = rec['row']
        if isinstance(source, pd.DataFrame):
            if rec['table'] == 'T3':
                config, comparison = row.split('/', 1)
                sub = source.loc[source.config.eq(config) & source.comparison.eq(comparison)]
            elif rec['table'] == 'T4':
                protocol, model = row.split('/', 1)
                sub = source.loc[source.protocol.eq(protocol) & source.model.eq(model)]
            else:
                analysis, config, group = row.split('/', 2)
                sub = source.loc[source.analysis.eq(analysis) & source.config.eq(config)
                                 & source.group.eq(group)]
            require(len(sub) == 1, f'Non-unique source row {path}: {row}')
            obj = sub.iloc[0].to_dict()
        else:
            obj = source
        field = rec['source_field']
        for part in re.findall(r'[^.\[\]]+|\[\d+\]', field):
            if part.startswith('['):
                obj = obj[int(part[1:-1])]
            elif isinstance(obj, dict):
                require(part in obj, f'Missing source key {part} in {path}')
                obj = obj[part]
            else:
                raise ValueError(f'Unexpected source structure {field}')
            # JSON object nested inside one CSV cell.
            if isinstance(obj, str) and obj.startswith('{') and part == 'details':
                obj = json.loads(obj)
        require(numeric(obj), f'Non-numeric frozen source {path}: {field}')
        require(Decimal(str(obj)) == Decimal(rec['value']),
                f'EXACT_MATCH_FAIL {path} {row} {field}: {obj} != {rec["value"]}')
        count += 1
    return count


def run(config_path, stage='run'):
    started=time.perf_counter()
    cfg=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    require(cfg['task']=='P21-T06b' and cfg['run_name']=='t06b_synthesis'
            and cfg['prereg_sha256']==PREREG, 'T06b config/prereg mismatch')
    root=Path(cfg['work_root']).resolve()
    base=root/'P21_CTG_Acidemia'
    out=base/'outputs/t06b_synthesis'
    require(not out.exists(),f'STOP: output directory exists: {out}')
    sources=input_paths(root)
    freeze,locks=verify_locked_inputs(root,sources)
    print(f'FROZEN_INPUT_SHA_PASS {len(freeze)}/{len(sources)}; manifests={len(locks)}',flush=True)
    if stage=='inspect':
        print('T06B_INSPECT_PASS; no output created')
        return 0
    require(stage=='run','Invalid stage')
    # Validate and derive all tabular content before creating the output directory.
    trace=[]
    t1=make_t1(root,trace)
    t2=make_t2(root,trace)
    t3=make_t3(root,trace)
    t4=make_t4(root,trace)
    t5=make_t5(root,trace)
    means=load_means(root)
    segments=pd.read_parquet(base/'outputs/t05_leakage/segments_meta.parquet')
    nseg=int(segments.n_segments.sum())
    ndrop=int(segments.n_dropped_missing.sum())
    require(nseg==5224 and ndrop==499,'F1 segment counts inconsistent with T05 report')
    # T2-T5 source verification: numeric values retained with zero tolerance.
    checks=0
    for item in trace:
        require(item['source_sha8']==freeze[item['source_file']], 'T2-T5 source SHA tracking error')
    checks=audit_t2_t5_exact(trace,root)
    require(checks>0, 'Exact-match audit found no values')
    tracefile=base/'reports/NUMBERS_TRACE.csv'
    require(not tracefile.exists(), f'STOP: NUMBERS_TRACE.csv already exists: {tracefile}')
    figure_style()
    out.mkdir(parents=False,exist_ok=False)
    try:
        td=out/'tables'; fd=out/'figures'; td.mkdir();fd.mkdir()
        frames={'T1_cohort':t1,'T2_H1_models':t2,'T3_H1_comparisons':t3,
                'T4_H2_leakage':t4,'T5_exploratory':t5}
        for name,frame in frames.items():
            emit_table(td,name,frame)
        trace_derived_frame(trace,root,'T1',t1,td/'T1_cohort.csv','cohort')
        trace_derived_frame(trace,root,'T4_Youden',t4[['protocol','model','Youden_S','Youden_L']]
                            .loc[t4.protocol.eq('L4')],td/'T4_H2_leakage.csv','protocol')
        dims={}
        dims['F1']=f1_flow(fd,nseg,ndrop)
        dims['F2']=f2_forest(fd,t4)
        dims['F3']=f3_roc_calibration(fd,means,t2)
        dims['F4']=f4_dca(fd,means)
        dims['F5']=f5_e2(fd,t5)
        for fig in dims:
            path=fd/(fig+'_plot_data.csv')
            plot=pd.read_csv(path,dtype=str,keep_default_na=False)
            trace_derived_frame(trace,root,fig,plot,path,'stage' if fig=='F1' else 'row')
        finish_derived_trace(trace,root)
        # Trace outside synthesis output by instruction. Atomic create, never overwrite.
        import tempfile
        with tempfile.NamedTemporaryFile('w',dir=tracefile.parent,encoding='utf-8',
                                         newline='',delete=False,prefix='.trace-t06b-',suffix='.tmp') as f:
            tmp=Path(f.name)
            writer=csv.DictWriter(f,fieldnames=['table','row','column','value','source_file','source_sha8','source_field'])
            writer.writeheader();writer.writerows(trace)
        os.link(tmp,tracefile)  # atomic create: cannot replace an existing trace
        tmp.unlink()
        produced=[p for p in sorted(out.rglob('*')) if p.is_file()]
        result={'task':'P21-T06b','status':'PASS','prereg_sha256':PREREG,
                'locked_inputs':freeze,'manifests':locks,'source_numeric_values_checked':checks,
                'numbers_trace_rows':len(trace),'figures_pixels':dims,
                'output_sha8':{rel(p,root):sha(p)[:8] for p in produced},
                'script_sha8':sha(__file__)[:8],'runtime_s':time.perf_counter()-started}
        (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        report=['=== REPORT P21-T06b ===','status=PASS; PC-A CPU; no-training; no-new-tests',
                'PREREG_v1.3_SHA256='+PREREG,
                f'FROZEN_SHA=PASS; locked_files={len(freeze)}; manifests={len(locks)}',
                'tables: '+', '.join(frames),
                'figures: '+', '.join(f'{k}={v[0]}x{v[1]}px PNG600+SVG' for k,v in dims.items()),
                f'NUMBERS_TRACE_rows={len(trace)}; missing_sources=0',
                f'T2_T5_numeric_exact_match={checks}/{checks}=100%; tolerance=0',
                f'runtime_s={time.perf_counter()-started:.2f}; script_sha8={sha(__file__)[:8]}',
                'output_sha8: '+', '.join(f'{p.name}={sha(p)[:8]}' for p in sorted(td.glob('*.csv'))),
                'figure_csv_sha8: '+', '.join(f'{p.name}={sha(p)[:8]}' for p in sorted(fd.glob('*.csv'))),
                f'NUMBERS_TRACE_sha8={sha(tracefile)[:8]}', '=== END ===']
        require(len(report)<=30, 'Report too long')
        (out/'report.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
        # Final provenance manifest includes output PNG, SVG, CSV, TeX, JSON, report
        # and the independent trace. It does not include itself.
        all_files = sorted(p for p in out.rglob('*') if p.is_file()) + [tracefile]
        lines = [f'{sha(p)}  {rel(p,root)}' for p in all_files]
        (out/'OUTPUT_SHA256.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('\n'.join(report))
        print(f'OUTPUT_SHA256_MANIFEST={rel(out/"OUTPUT_SHA256.txt",root)}; '
              f'file_count={len(all_files)}; sha8={sha(out/"OUTPUT_SHA256.txt")[:8]}')
        return 0
    except Exception:
        # Keep any staging artifacts for diagnosis; never overwrite or re-run in place.
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=('inspect','run'),default='run')
    args=parser.parse_args()
    try:
        return run(args.config,args.stage)
    except (ValueError,FileNotFoundError,KeyError,AssertionError,TypeError) as exc:
        print(f'P21_T06B_STOP: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())

from __future__ import annotations
import hashlib
from pathlib import Path

import pandas as pd
import yaml

from P21_CTG_Acidemia.src.t06c_figures import parse_manifest, verify_inputs, main


def sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def write_csvs(root: Path):
    figdir = root / 'P21_CTG_Acidemia/outputs/t06b_synthesis/figures'
    figdir.mkdir(parents=True)
    pd.DataFrame([{'stage': 0, 'label': 'dummy', 'x': 0.0, 'y': 0.0, 'width': 1.0, 'height': 1.0}]).to_csv(figdir / 'F1_plot_data.csv', index=False)
    pd.DataFrame([
        {'label': 'L1/M1', 'S': 0.63, 'L': 0.72, 'S_low': 0.55, 'S_high': 0.70, 'L_low': 0.66, 'L_high': 0.78, 'delta': 0.09, 'p_holm': 0.0008, 'descriptive': False},
        {'label': 'L1 record/M1', 'S': 0.67, 'L': 0.80, 'S_low': 0.57, 'S_high': 0.77, 'L_low': 0.71, 'L_high': 0.87, 'delta': '', 'p_holm': '', 'descriptive': True},
        {'label': 'L1/M2', 'S': 0.64, 'L': 0.84, 'S_low': 0.55, 'S_high': 0.72, 'L_low': 0.79, 'L_high': 0.87, 'delta': 0.20, 'p_holm': 0.0008, 'descriptive': False},
        {'label': 'L1 record/M2', 'S': 0.66, 'L': 0.92, 'S_low': 0.56, 'S_high': 0.76, 'L_low': 0.88, 'L_high': 0.96, 'delta': '', 'p_holm': '', 'descriptive': True},
        {'label': 'L2/M1', 'S': 0.60, 'L': 0.84, 'S_low': 0.49, 'S_high': 0.71, 'L_low': 0.77, 'L_high': 0.91, 'delta': 0.24, 'p_holm': 0.0008, 'descriptive': False},
        {'label': 'L2 augmented/M1', 'S': 0.60, 'L': 0.99, 'S_low': '', 'S_high': '', 'L_low': '', 'L_high': '', 'delta': 0.386, 'p_holm': '', 'descriptive': True},
        {'label': 'L2/M2', 'S': 0.70, 'L': 1.00, 'S_low': 0.62, 'S_high': 0.77, 'L_low': 0.999, 'L_high': 1.0, 'delta': 0.30, 'p_holm': 0.0008, 'descriptive': False},
        {'label': 'L2 augmented/M2', 'S': 0.70, 'L': 0.9995, 'S_low': '', 'S_high': '', 'L_low': '', 'L_high': '', 'delta': 0.302, 'p_holm': '', 'descriptive': True},
        {'label': 'L3/M1', 'S': 0.60, 'L': 0.60, 'S_low': 0.49, 'S_high': 0.71, 'L_low': 0.49, 'L_high': 0.70, 'delta': -0.002, 'p_holm': 0.9666, 'descriptive': False},
        {'label': 'L3/M2', 'S': 0.70, 'L': 0.78, 'S_low': 0.62, 'S_high': 0.77, 'L_low': 0.70, 'L_high': 0.85, 'delta': 0.079, 'p_holm': 0.0108, 'descriptive': False},
    ]).to_csv(figdir / 'F2_plot_data.csv', index=False)
    rows = []
    for model in ['M0', 'M1', 'M2', 'M3']:
        for x, y in [(0, 0), (0.2, 0.4), (1, 1)]:
            rows.append({'panel': 'ROC', 'model': model, 'x': x, 'y': y, 'n': '', 'positive_n': '', 'source_metric': 'src'})
        for x, y in [(0.1, 0.05), (0.2, 0.1), (0.4, 0.2), (0.6, 0.4), (0.8, 0.7)]:
            rows.append({'panel': 'calibration', 'model': model, 'x': x, 'y': y, 'n': 10, 'positive_n': 2, 'source_metric': 'src'})
    pd.DataFrame(rows).to_csv(figdir / 'F3_plot_data.csv', index=False)
    rows = []
    for t in [0.05, 0.1, 0.2, 0.3]:
        rows += [
            {'threshold': t, 'strategy': 'M1', 'net_benefit': 0.2 - t / 10},
            {'threshold': t, 'strategy': 'M3', 'net_benefit': 0.22 - t / 12},
            {'threshold': t, 'strategy': 'treat_all', 'net_benefit': 0.1 - t / 8},
            {'threshold': t, 'strategy': 'treat_none', 'net_benefit': 0.0},
        ]
    pd.DataFrame(rows).to_csv(figdir / 'F4_plot_data.csv', index=False)
    pd.DataFrame([
        {'group': 'le_median', 'model': 'M1', 'x': -0.1, 'auroc': 0.47, 'ci95_low': 0.26, 'ci95_high': 0.70, 'n': 259, 'positives': '<10'},
        {'group': 'le_median', 'model': 'M3', 'x': 0.1, 'auroc': 0.75, 'ci95_low': 0.61, 'ci95_high': 0.87, 'n': 259, 'positives': '<10'},
        {'group': 'gt_median', 'model': 'M1', 'x': 0.9, 'auroc': 0.66, 'ci95_low': 0.52, 'ci95_high': 0.79, 'n': 259, 'positives': '21'},
        {'group': 'gt_median', 'model': 'M3', 'x': 1.1, 'auroc': 0.70, 'ci95_low': 0.59, 'ci95_high': 0.80, 'n': 259, 'positives': '21'},
    ]).to_csv(figdir / 'F5_plot_data.csv', index=False)
    return figdir


def write_manifest(root: Path, figdir: Path):
    reportdir = root / 'P21_CTG_Acidemia/reports'
    reportdir.mkdir(parents=True, exist_ok=True)
    lines = ['# frozen plot data']
    for p in sorted(figdir.glob('F*_plot_data.csv')):
        lines.append(f'{sha8(p)} {p.relative_to(root).as_posix()}')
    man = reportdir / 'T06b_FROZEN_SHA.txt'
    man.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return man


def write_config(root: Path):
    cfg = {
        'work_root': str(root).replace('\\', '/'),
        'prereg_sha256': '40573556ecf4664c430430eb99eaaa20d1d0958aa32647cd30b7144434d048eb',
        'input_manifest': 'P21_CTG_Acidemia/reports/T06b_FROZEN_SHA.txt',
        'input_plot_folder': 'P21_CTG_Acidemia/outputs/t06b_synthesis/figures',
        'output_folder': 'P21_CTG_Acidemia/outputs/t06c_figures',
        'font_family': 'Arial',
        'palette': {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'vermillion': '#D55E00', 'purple': '#CC79A7', 'black': '#000000', 'grey': '#999999'},
        'figure_specs': {'F1': {'width_in': 3.5, 'height_in': 5.0}, 'F2': {'width_in': 7.0, 'height_in': 5.4}, 'F3': {'width_in': 7.0, 'height_in': 3.2}, 'F4': {'width_in': 3.5, 'height_in': 2.8}, 'F5': {'width_in': 3.5, 'height_in': 2.8}},
    }
    p = root / 'P21_CTG_Acidemia/configs/t06c.yaml'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return p


def test_parse_manifest(tmp_path):
    p = tmp_path / 'm.txt'
    p.write_text('#note\n12345678 some/file.csv\n', encoding='utf-8')
    got = parse_manifest(p)
    assert got['some/file.csv'] == '12345678'


def test_verify_inputs(tmp_path):
    figdir = write_csvs(tmp_path)
    write_manifest(tmp_path, figdir)
    files, report = verify_inputs(tmp_path, 'P21_CTG_Acidemia/reports/T06b_FROZEN_SHA.txt', 'P21_CTG_Acidemia/outputs/t06b_synthesis/figures')
    assert len(files) == 5
    assert len(report) == 5


def test_run_and_preserve_plot_data(tmp_path):
    figdir = write_csvs(tmp_path)
    write_manifest(tmp_path, figdir)
    cfg = write_config(tmp_path)
    assert main(['--stage', 'inspect', '--config', str(cfg)]) == 0
    assert main(['--stage', 'run', '--config', str(cfg)]) == 0
    out = tmp_path / 'P21_CTG_Acidemia/outputs/t06c_figures'
    for name in ['F1_plot_data.csv', 'F2_plot_data.csv', 'F3_plot_data.csv', 'F4_plot_data.csv', 'F5_plot_data.csv']:
        assert sha8(figdir / name) == sha8(out / name)
    assert main(['--stage', 'run', '--config', str(cfg)]) == 2


def test_output_sizes(tmp_path):
    figdir = write_csvs(tmp_path)
    write_manifest(tmp_path, figdir)
    cfg = write_config(tmp_path)
    assert main(['--stage', 'run', '--config', str(cfg)]) == 0
    out = tmp_path / 'P21_CTG_Acidemia/outputs/t06c_figures'
    from PIL import Image
    for name in ['F1', 'F2', 'F3', 'F4', 'F5']:
        with Image.open(out / f'{name}.png') as im:
            assert im.size[0] >= 2000

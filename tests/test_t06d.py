from __future__ import annotations
import hashlib
from pathlib import Path

import pandas as pd
import yaml

from P21_CTG_Acidemia.src.t06d_figures import parse_sha_manifest, frozen_plot_data_preflight, draw_f1, draw_f2, main, sha256, wrapped


def sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def write_csvs(root: Path):
    figdir = root / 'P21_CTG_Acidemia/outputs/t06b_synthesis/figures'
    figdir.mkdir(parents=True)
    pd.DataFrame([
        {'stage': 0, 'label': 'CTU-UHB recordings\n552', 'x': .08, 'y': .81, 'width': .84, 'height': .13},
        {'stage': 1, 'label': 'Excluded by T02 QC\n34 (10 positive; 24 negative)', 'x': .08, 'y': .64, 'width': .84, 'height': .13},
        {'stage': 2, 'label': 'Eligible cohort\n518 (30 positive)', 'x': .08, 'y': .47, 'width': .84, 'height': .13},
        {'stage': 3, 'label': 'Strict evaluation\n10 × 5 × 5 nested CV', 'x': .05, 'y': .19, 'width': .43, 'height': .16},
        {'stage': 4, 'label': 'T05 leakage audit\n5,224 retained\n499 rejected segments', 'x': .52, 'y': .19, 'width': .43, 'height': .16},
    ]).to_csv(figdir / 'F1_plot_data.csv', index=False)
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




def setup_case(tmp_path: Path):
    from PIL import Image
    figdir = write_csvs(tmp_path)
    write_manifest(tmp_path, figdir)
    old = tmp_path / 'P21_CTG_Acidemia/outputs/t06c_figures'
    old.mkdir(parents=True)
    pins = {}
    for i in (3, 4, 5):
        png = old / f'F{i}.png'
        svg = old / f'F{i}.svg'
        Image.new('RGB', (2100, 1680), 'white').save(png)
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><text x="1" y="5">AUROC</text></svg>')
        pins[png.name] = sha8(png)
        pins[svg.name] = sha8(svg)
    cfg = {
        'task': 'P21-T06d',
        'work_root': str(tmp_path),
        'prereg_sha256': '40573556ecf4664c430430eb99eaaa20d1d0958aa32647cd30b7144434d048eb',
        'frozen_manifest': 'P21_CTG_Acidemia/reports/T06b_FROZEN_SHA.txt',
        'frozen_plot_dir': 'P21_CTG_Acidemia/outputs/t06b_synthesis/figures',
        'approved_t06c_dir': 'P21_CTG_Acidemia/outputs/t06c_figures',
        'output_dir': 'P21_CTG_Acidemia/outputs/t06d_figures',
        'approved_t06c_sha8': pins,
        'figure': {'font': 'DejaVu Sans', 'dpi': 600,
                   'F1_width_in': 3.5, 'F1_height_in': 4.2,
                   'F2_width_in': 7.0, 'F2_height_in': 5.4,
                   'bbox_inches': 'tight', 'pad_inches': 0.05},
    }
    yamlpath = tmp_path / 't06d.yaml'
    yamlpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg, yamlpath, figdir, old


def test_manifest_parse(tmp_path):
    p = tmp_path / 'sha.txt'
    p.write_text('# comment\n12345678 abc.csv\n', encoding='utf-8')
    assert parse_sha_manifest(p) == {'abc.csv': '12345678'}


def test_textwrap_limit():
    txt = wrapped('Excluded: > 50% FHR signal loss in last 30 min of stage 1\nn = 34 (acidaemic 10; non-acidaemic 24)')
    assert max(map(len, txt.splitlines())) <= 26


def test_inputs_sha(tmp_path):
    cfg, p, figdir, old = setup_case(tmp_path)
    files, hashes = frozen_plot_data_preflight(tmp_path, cfg)
    assert len(files) == 5 and len(hashes) == 5


def test_end_to_end(tmp_path):
    cfg, p, figdir, old = setup_case(tmp_path)
    assert main(['--config', str(p), '--stage', 'inspect']) == 0
    assert main(['--config', str(p), '--stage', 'run']) == 0
    out = tmp_path / cfg['output_dir']
    assert out.exists()
    for i in range(1, 6):
        assert (out / f'F{i}.png').is_file()
        assert (out / f'F{i}.svg').is_file()
        assert sha256(figdir / f'F{i}_plot_data.csv') == sha256(out / f'F{i}_plot_data.csv')
    for i in (3, 4, 5):
        for ext in ('png', 'svg'):
            assert sha256(old / f'F{i}.{ext}') == sha256(out / f'F{i}.{ext}')
    report = (out / 'report.txt').read_text()
    assert 'F1_text_bbox_overlap=PASS' in report
    assert 'F2_source_rows_plotted=10/10' in report
    assert len(report.splitlines()) <= 15
    assert main(['--config', str(p), '--stage', 'run']) == 2


def test_fail_closed_sha_mismatch(tmp_path):
    cfg, p, figdir, old = setup_case(tmp_path)
    (figdir / 'F4_plot_data.csv').write_text('modified\n')
    assert main(['--config', str(p), '--stage', 'inspect']) == 2
    assert not (tmp_path / cfg['output_dir']).exists()


def test_fail_closed_unapproved_copy(tmp_path):
    cfg, p, figdir, old = setup_case(tmp_path)
    (old / 'F3.svg').write_text('modified')
    assert main(['--config', str(p), '--stage', 'run']) == 2
    assert not (tmp_path / cfg['output_dir']).exists()


def test_f1_approved_count_must_match_source(tmp_path):
    cfg, p, figdir, old = setup_case(tmp_path)
    # Simulate a syntactically valid SHA lock for a scientifically inconsistent file.
    f1 = figdir / 'F1_plot_data.csv'
    data = pd.read_csv(f1)
    data.loc[0, 'label'] = 'CTU-UHB recordings\n551'
    data.to_csv(f1, index=False)
    write_manifest(tmp_path, figdir)
    assert main(['--config', str(p), '--stage', 'run']) == 2
    # Preflight must not leave behind a partially created output directory.
    assert not (tmp_path / cfg['output_dir']).exists()
    assert (tmp_path / cfg['approved_t06c_dir'] / 'F3.png').is_file()

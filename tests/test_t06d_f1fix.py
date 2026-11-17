from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from P21_CTG_Acidemia.src.t06d_f1fix import BOX_TEXT, create_f1, main, parse_locks, sha256, wrapped


def fixture_root(tmp_path: Path):
    figdir = tmp_path/'P21_CTG_Acidemia/outputs/t06b_synthesis/figures'
    figdir.mkdir(parents=True)
    file = figdir/'F1_plot_data.csv'
    pd.DataFrame([
        {'stage': 0, 'label': 'CTU-UHB recordings\n552'},
        {'stage': 1, 'label': 'Excluded by T02 QC\n34 (10 positive; 24 negative)'},
        {'stage': 2, 'label': 'Eligible cohort\n518 (30 positive)'},
        {'stage': 3, 'label': 'Strict evaluation\n10 × 5 × 5 nested CV'},
        {'stage': 4, 'label': 'T05 leakage audit\n5,224 retained\n499 rejected segments'},
    ]).to_csv(file, index=False)
    reports=tmp_path/'P21_CTG_Acidemia/reports'
    reports.mkdir(parents=True)
    lock=reports/'T06b_FROZEN_SHA.txt'
    lock.write_text('# plot-data frozen input\n'+sha256(file)[:8]+'  '+file.relative_to(tmp_path).as_posix()+'\n',encoding='utf-8')
    old=tmp_path/'P21_CTG_Acidemia/outputs/t06d_figures'
    old.mkdir(parents=True)
    (old/'F1.png').write_bytes(b'old approved F1 PNG immutable bytes')
    (old/'F1.svg').write_bytes(b'old approved F1 SVG immutable bytes')
    pins={p.name:sha256(p)[:8] for p in old.iterdir()}
    cfg={
        'task':'P21-T06d-F1fix',
        'work_root':str(tmp_path),
        'prereg_sha256':'40573556ecf4664c430430eb99eaaa20d1d0958aa32647cd30b7144434d048eb',
        'frozen_manifest':'P21_CTG_Acidemia/reports/T06b_FROZEN_SHA.txt',
        'frozen_F1_plot_data':'P21_CTG_Acidemia/outputs/t06b_synthesis/figures/F1_plot_data.csv',
        'prior_t06d_F1_dir':'P21_CTG_Acidemia/outputs/t06d_figures',
        'prior_t06d_F1_sha8':pins,
        'output_dir':'P21_CTG_Acidemia/outputs/t06d_figures_F1fix',
        'figure':{'font':'Arial','fontsize_pt':8,'dpi':600,'width_in':3.5,'height_in':4.2,
                  'linespacing':1.35,'box_pad':0.6,'y_lower_box':0.30,'y_branch':0.45,
                  'ylim':[0.18,1.0],'bbox_inches':'tight','pad_inches':0.05}
    }
    cfgp=tmp_path/'t06d_f1fix.yaml'
    cfgp.write_text(yaml.safe_dump(cfg, sort_keys=False),encoding='utf-8')
    return cfg,cfgp,file,old


def test_exact_text_and_horizontal_positions():
    assert [r[:2] for r in BOX_TEXT]==[
        ('A', .40),('B', .40),('E', .78),('S', .78),('C', .22),('D', .65)]
    assert [r[2] for r in BOX_TEXT] == [.92,.62,.77,.92,.30,.30]


def test_wrapping():
    assert max(len(x) for x in wrapped(BOX_TEXT[2][3]).splitlines()) <= 26


def test_manifest_format(tmp_path):
    p = tmp_path/'lock.txt'
    p.write_text('# comment\n12345678 foo.csv\n', encoding='utf-8')
    assert parse_locks(p) == {'foo.csv': '12345678'}


def test_inspect_without_writing(tmp_path):
    cfg, cpath, src, old = fixture_root(tmp_path)
    assert main(['--config',str(cpath),'--stage','inspect']) == 0
    assert not (tmp_path/cfg['output_dir']).exists()


def test_run_saves_only_two_files_and_preserves_inputs(tmp_path):
    cfg, cpath, src, old = fixture_root(tmp_path)
    orig = {p:sha256(p) for p in [src, old/'F1.png', old/'F1.svg']}
    assert main(['--config',str(cpath),'--stage','run']) == 0
    dest = tmp_path/cfg['output_dir']
    assert sorted(p.name for p in dest.iterdir()) == ['F1.png','F1.svg']
    with Image.open(dest/'F1.png') as im:
        assert im.size[0] >= 2000
        assert abs(im.info['dpi'][0] - 600) < 1
    assert {p:sha256(p) for p in orig} == orig
    assert main(['--config',str(cpath),'--stage','run']) == 2


def test_frozen_sha_mismatch_fails_closed(tmp_path):
    cfg, cpath, src, old = fixture_root(tmp_path)
    src.write_text('corrupted\n', encoding='utf-8')
    assert main(['--config',str(cpath),'--stage','run']) == 2
    assert not (tmp_path/cfg['output_dir']).exists()


def test_bad_prior_sha_fails_closed(tmp_path):
    cfg, cpath, src, old = fixture_root(tmp_path)
    (old/'F1.png').write_bytes(b'changed')
    assert main(['--config',str(cpath),'--stage','run']) == 2
    assert not (tmp_path/cfg['output_dir']).exists()


def test_layout_contract_fail_closed(tmp_path):
    cfg, cpath, src, old = fixture_root(tmp_path)
    cfg['figure']['linespacing']=0.84
    cpath.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    assert main(['--config',str(cpath),'--stage','run']) == 2
    assert not (tmp_path/cfg['output_dir']).exists()

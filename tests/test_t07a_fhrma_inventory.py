"""Synthetic read-only tests. Do not require the real FHRMA repository."""
from pathlib import Path
import hashlib
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import savemat

from P21_CTG_Acidemia.src.t07a_fhrma_inventory import (
    PREREG_V14_SHA256, collect_files, display_tree, mat_metadata, readme_head,
    report, split_counts, main
)


def make_repo(tmp_path):
    repo = tmp_path / 'fhrma'
    (repo / 'FHRMAdataset' / 'traindata').mkdir(parents=True)
    (repo / 'FHRMAdataset' / 'testdata').mkdir(parents=True)
    (repo / 'FHRMAdataset' / 'analyses').mkdir(parents=True)
    (repo / '.git').mkdir()
    (repo / '.git' / 'config').write_text('dummy')
    for p in [repo/'FHRMAdataset/traindata/train01.fhr',repo/'FHRMAdataset/traindata/train02.fhr',repo/'FHRMAdataset/testdata/test01.fhr']:
        p.write_bytes(b'EXAMPLE_BINARY_DATA_NOT_TO_PARSE')
    savemat(str(repo/'FHRMAdataset/analyses/expertAnalyses.mat'),{
        'baseline': np.zeros((12, 1), dtype=np.float64),
        'mark': np.arange(4, dtype=np.uint16).reshape(2,2)
    })
    (repo/'README.md').write_text('# Example\nTwo lines\n',encoding='utf-8')
    (repo/'fhropen.m').write_text('% no parser; metadata only\n',encoding='utf-8')
    (repo/'evaluateaam.m').write_text('% no parser; metadata only\n',encoding='utf-8')
    return repo


def test_inventory_counts_no_git(tmp_path):
    r=make_repo(tmp_path)
    fs,ds=collect_files(r)
    assert len(fs)==7
    assert not any('.git' in p.parts for p in fs)
    assert split_counts(r,fs)['traindata']['fhr_files']==2
    assert split_counts(r,fs)['testdata']['fhr_files']==1
    assert all(not ('/config' in s) for s in display_tree(r,fs,ds,3))


def test_mat_metadata_names_shapes_dtypes_only(tmp_path):
    r=make_repo(tmp_path)
    entries,err=mat_metadata(r/'FHRMAdataset/analyses/expertAnalyses.mat')
    assert err is None
    assert {v['name'] for v in entries}=={'baseline','mark'}
    assert {v['dtype'] for v in entries}=={'float64','uint16'}
    assert any(v['shape']=='(12, 1)' for v in entries)
    assert all('value' not in str(e).lower() for e in entries)


def test_readme_limit(tmp_path):
    r=make_repo(tmp_path)
    (r/'README.md').write_text('\n'.join(f'line{i}' for i in range(50)),encoding='utf-8')
    path, lines=readme_head(r)
    assert path.name=='README.md' and len(lines)==40


def test_report_does_not_modify_source(tmp_path):
    r=make_repo(tmp_path)
    before={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in collect_files(r)[0]}
    output,failed=report(r)
    after={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in collect_files(r)[0]}
    assert before==after
    assert not failed
    assert 'train_fhr_files=2' in output and 'test_fhr_files=1' in output
    assert 'expertAnalyses.mat' in output
    assert 'SCHEMA' not in output or 'schema NOT verified' in output
    assert PREREG_V14_SHA256 in output
    assert 'NEXT_ACTION=STOP_AND_REQUEST_MATLAB_SOURCE_FILES' in output


def test_cli_missing_repo_fails_closed(tmp_path,capfd):
    assert main(['--repo',str(tmp_path/'NO_REPO')])==2
    assert 'FHRMA_REPOSITORY_NOT_FOUND' in capfd.readouterr().err


def test_hdf5_metadata_only(tmp_path):
    h5py=pytest.importorskip('h5py')
    file=tmp_path/'v73.mat'
    with h5py.File(file,'w') as f:
        ds=f.create_dataset('label',shape=(100,2),dtype='float32')
        ds.attrs['MATLAB_class']=b'single'
        f.create_group('#refs#')
    vars_,err=mat_metadata(file)
    assert err is None
    assert len(vars_)==1
    assert vars_[0]['name']=='label'
    assert vars_[0]['shape']=='(100, 2)'
    assert vars_[0]['dtype']=='float32'

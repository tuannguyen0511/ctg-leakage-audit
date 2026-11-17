"""P21-T07a: read-only inventory of the FHRMA repository.

Intentionally does NOT parse FHR or expert annotation contents, load signal values,
call ctg_features.py, or perform any validation statistics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import re
import subprocess
import sys

PREREG_V14_SHA256 = '85161286492355667fefb0d55d1643cedc5b2736d8234cac1914023c05744c58'
KEYWORDS = ('open', 'read', 'load', 'eval', 'expert', 'analysis')
RECOMMENDED_M = ('fhropen.m', 'fhrmorphoopenandplot.m', 'fhrmorpho.m',
                 'evaluateaam.m', 'fhrplotMA.m', 'startendlist.m',
                 'simpleaddetection.m', 'showFSAnalysis.m')
MAT_CLASS_TO_DTYPE = {
    'double': 'float64', 'single': 'float32', 'logical': 'bool',
    'int8': 'int8', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64',
    'uint8': 'uint8', 'uint16': 'uint16', 'uint32': 'uint32',
    'uint64': 'uint64', 'char': 'MATLAB char', 'cell': 'MATLAB cell',
    'struct': 'MATLAB struct', 'object': 'MATLAB object',
}


def collect_files(repo: Path):
    """All repository files, excluding Git internals and unsafe symlinks."""
    files, dirs = [], []
    for parent, subdirs, filenames in os.walk(repo, topdown=True, followlinks=False):
        parent_path = Path(parent)
        subdirs[:] = sorted(d for d in subdirs if d != '.git' and not (parent_path / d).is_symlink())
        if parent_path != repo:
            dirs.append(parent_path)
        for name in sorted(filenames):
            path = parent_path / name
            if path.is_file() and not path.is_symlink():
                files.append(path)
    return sorted(files, key=lambda p: p.relative_to(repo).as_posix().lower()), sorted(dirs, key=lambda p: p.relative_to(repo).as_posix().lower())


def display_tree(repo: Path, files: list[Path], dirs: list[Path], depth: int):
    """Print file/folder paths only if they are within the chosen max depth."""
    entries = [(p.relative_to(repo), False) for p in files]
    entries += [(p.relative_to(repo), True) for p in dirs]
    entries = sorted((r, is_dir) for r, is_dir in entries if len(r.parts) <= depth)
    lines = [f'{repo.name}/  [tree depth <= {depth}; .git excluded]']
    for name, is_dir in entries:
        lines.append('  ' * len(name.parts) + ('/' if is_dir else '- ') + str(name).replace('\\', '/') + ('/' if is_dir else ''))
    return lines


def mat_metadata(path: Path):
    """Read MATLAB variable HEADERS ONLY. Never load arrays or values."""
    try:
        import scipy.io
    except ImportError:
        return [], 'SCIPY_NOT_INSTALLED'
    try:
        # whosmat reads metadata (variable name, shape, MATLAB class).
        variables = scipy.io.whosmat(str(path))
        return [
            {'name': name, 'shape': str(tuple(shape)),
             'dtype': MAT_CLASS_TO_DTYPE.get(kind, f'MATLAB:{kind}'),
             'matlab_class': kind}
            for name, shape, kind in variables
        ], None
    except Exception as exc:
        # Some .mat files are MATLAB 7.3 HDF5; inspect only HDF5 headers.
        try:
            import h5py
        except ImportError:
            return [], f'SCIPY_METADATA_ERROR={type(exc).__name__}; H5PY_NOT_INSTALLED'
        try:
            if not h5py.is_hdf5(str(path)):
                return [], f'MAT_METADATA_ERROR={type(exc).__name__}: {str(exc)[:150]}'
            rows = []
            with h5py.File(path, 'r') as h:
                for name in sorted(h):
                    # #refs# is a MATLAB internal reference store, not a user variable.
                    if name == '#refs#':
                        continue
                    item = h[name]
                    if isinstance(item, h5py.Dataset):
                        shape = str(tuple(item.shape))
                        dtype = str(item.dtype)
                        klass = item.attrs.get('MATLAB_class', b'')
                        if isinstance(klass, bytes):
                            klass = klass.decode('utf-8', errors='replace')
                        klass = str(klass) or 'not reported'
                    else:
                        shape = 'HDF5 group / variable structure'
                        dtype = 'HDF5 group'
                        klass = 'MATLAB class not inferred'
                    rows.append({'name': name, 'shape': shape,
                                 'dtype': dtype, 'matlab_class': klass})
            return rows, None
        except Exception as sub_exc:
            return [], f'HDF5_METADATA_ERROR={type(sub_exc).__name__}: {str(sub_exc)[:150]}'


def git_meta(repo: Path):
    def git(*args):
        try:
            cp = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=15, check=False)
            return cp.stdout.strip() if cp.returncode == 0 else 'not available'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return 'not available'
    return git('rev-parse', 'HEAD'), git('remote', 'get-url', 'origin')


def split_counts(repo: Path, files: list[Path]):
    """Counts file names, not record contents, labels or patient identities."""
    result = {}
    for split in ('traindata', 'testdata'):
        prefix = ('FHRMAdataset', split)
        paths = [p for p in files if p.relative_to(repo).parts[:2] == prefix]
        fhr = [p for p in paths if p.suffix.lower() == '.fhr']
        result[split] = {'all_files': len(paths), 'fhr_files': len(fhr),
                         'unique_fhr_stems': len({p.stem for p in fhr})}
    return result


def readme_head(repo: Path):
    matches = [p for p in repo.iterdir() if p.is_file() and p.name.lower().startswith('readme')]
    if not matches:
        return None, []
    p = sorted(matches)[0]
    # Read text only; no binary dataset reading.
    try:
        lines = p.read_text(encoding='utf-8-sig', errors='replace').splitlines()
    except OSError as e:
        return p, [f'ERROR_README: {type(e).__name__}: {e}']
    return p, lines[:40]


def report(repo: Path, max_depth: int = 3):
    if not repo.is_dir():
        raise ValueError(f'FHRMA_REPOSITORY_NOT_FOUND: {repo}')
    if not (repo / 'FHRMAdataset').is_dir():
        raise ValueError(f'FHRMA_DATASET_FOLDER_NOT_FOUND: {repo / "FHRMAdataset"}')

    files, dirs = collect_files(repo)
    extensions = Counter(p.suffix.lower() if p.suffix else '[no extension]' for p in files)
    commit, origin = git_meta(repo)
    split = split_counts(repo, files)
    mat_files = [p for p in files if p.suffix.lower() == '.mat']
    m_files = [p for p in files if p.suffix.lower() == '.m']
    keywords = [p for p in m_files if any(w in p.stem.lower() for w in KEYWORDS)]
    recommended = [p for p in m_files if p.name in RECOMMENDED_M]
    readme, read_lines = readme_head(repo)

    out = []
    def emit(*ss):
        out.extend(str(s) for s in ss)

    emit('=== P21-T07a FHRMA READ-ONLY INVENTORY ===',
         f'PREREG_v1.4_SHA256={PREREG_V14_SHA256}',
         f'REPOSITORY={repo}', f'GIT_COMMIT={commit}', f'GIT_ORIGIN={origin}',
         'STAGE=a_only; no_parser; no_signal_values; no_expert_values; no_metrics')
    emit('=== TREE (MAX 3 LEVELS) ===')
    emit(*display_tree(repo, files, dirs, max_depth))
    emit('=== FILE COUNTS BY EXTENSION ===')
    for ext, count in sorted(extensions.items(), key=lambda x: (-x[1], x[0])):
        emit(f'{ext}: {count}')
    emit('=== MATLAB .mat VARIABLE METADATA (NO VALUES) ===')
    mat_failed = []
    for p in mat_files:
        name = p.relative_to(repo).as_posix()
        vars_, err = mat_metadata(p)
        emit(f'MAT_FILE={name}; vars={len(vars_)}; size_bytes={p.stat().st_size}')
        for item in vars_:
            emit(f'  variable={item["name"]}; shape={item["shape"]}; '
                 f'dtype={item["dtype"]}; class={item["matlab_class"]}')
        if err:
            mat_failed.append((name, err))
            emit(f'  METADATA_UNAVAILABLE: {err}')
    emit('=== README FIRST 40 LINES ===')
    emit(f'README_PATH={readme.relative_to(repo).as_posix() if readme else "NOT_FOUND"}')
    for i, line in enumerate(read_lines, 1):
        emit(f'{i:02d}: {line}')
    emit('=== MATLAB FILES WITH OPEN/READ/LOAD/EVAL/EXPERT/ANALYSIS IN NAME ===')
    for p in keywords:
        emit(p.relative_to(repo).as_posix())
    emit('=== ADDITIONAL MATLAB FILES RECOMMENDED FOR FORMAT REVIEW ===')
    for p in recommended:
        emit(p.relative_to(repo).as_posix())
    emit('=== REPORT P21-T07a ===')
    emit('status=' + ('PASS' if not mat_failed else 'PARTIAL_MAT_METADATA'),
         f'PREREG_v1.4_SHA256={PREREG_V14_SHA256}',
         f'git_commit={commit}', f'git_origin={origin}',
         f'total_files={len(files)}; total_dirs={len(dirs)}; mat_files={len(mat_files)}; '
         f'mat_metadata_errors={len(mat_failed)}',
         f'train_fhr_files={split["traindata"]["fhr_files"]}; '
         f'train_unique_file_stems={split["traindata"]["unique_fhr_stems"]}',
         f'test_fhr_files={split["testdata"]["fhr_files"]}; '
         f'test_unique_file_stems={split["testdata"]["unique_fhr_stems"]}',
         'signal_file_format=.fhr (identified by file extension; contents NOT parsed)',
         'expert_annotation_candidate=FHRMAdataset/analyses/expertAnalyses.mat '
         '(variable metadata only; schema NOT verified)',
         'matlab_name_matched_files=' + ', '.join(p.relative_to(repo).as_posix() for p in keywords),
         'matlab_recommended_files=' + ', '.join(p.relative_to(repo).as_posix() for p in recommended),
         'DATA_PARSER=NOT_WRITTEN; FHR_VALUES_NOT_READ; EXPERT_VALUES_NOT_READ',
         'T07b_VALIDATION=NOT_STARTED; CTG_FEATURES_MODIFIED=NO',
         'NEXT_ACTION=STOP_AND_REQUEST_MATLAB_SOURCE_FILES',
         '=== END REPORT P21-T07a ===')
    return '\n'.join(out) + '\n', mat_failed


def main(argv=None):
    ap = argparse.ArgumentParser(description='Read-only FHRMA file/variable inventory. No parser or tests.')
    ap.add_argument('--repo', default='D:/Tuan/AI_training/data_raw/fhrma',
                    help='Path to cloned FHRMA repository')
    ap.add_argument('--max-depth', type=int, choices=(1, 2, 3), default=3,
                    help='Number of directory levels to list (1..3)')
    args = ap.parse_args(argv)
    try:
        text, failures = report(Path(args.repo).resolve(), args.max_depth)
        print(text, end='', flush=True)
        return 0 if not failures else 2
    except (ValueError, PermissionError, OSError) as exc:
        print(f'P21_T07A_STOP: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Portable path/audit wrapper for the byte-identical historical evaluator.

This file is release packaging, not the historical numerical implementation.
No training is started. Raw images and the six historical checkpoints are external.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
DATASETS = ('caltech256', 'mit_indoor67', 'flowers102')
METHODS = ('base', 'dasupcon')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def verify_package():
    """Pin every distributed code/config/manifest, not an obsolete whole project."""
    expected = read(HERE / 'PACKAGE_SHA256.json')
    actual_paths = {p.relative_to(HERE).as_posix() for p in HERE.rglob('*') if p.is_file()
                    and '__pycache__' not in p.parts and p.name != 'PACKAGE_SHA256.json'
                    and p.suffix != '.pyc'}
    if actual_paths != set(expected):
        raise RuntimeError('Controlled package file set changed; restore the release or explicitly rebuild its fingerprint.')
    for rel, checksum in expected.items():
        if sha(HERE / rel) != checksum:
            raise RuntimeError(f'Controlled package hash mismatch: {rel}')
    provenance = read(HERE / 'SOURCE_PROVENANCE.json')
    for rel, record in provenance['byte_identical_files'].items():
        if sha(HERE / rel) != record['sha256']:
            raise RuntimeError(f'Historical numerical source/manifest changed: {rel}')
    return read(HERE / 'registry.json')


def install_missing_exact(source, destination, expected):
    """Never overwrite a different existing manifest or source file."""
    source, destination = Path(source), Path(destination)
    if sha(source) != expected:
        raise RuntimeError(f'Bundled source hash mismatch: {source}')
    if destination.exists():
        if not destination.is_file() or sha(destination) != expected:
            raise RuntimeError(f'Refuse overwriting a different existing file: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def arguments(values):
    previous = sys.argv
    sys.argv = values
    try:
        yield
    finally:
        sys.argv = previous


def runtime_paths(args):
    legacy = args.legacy_root.expanduser().resolve()
    runtime = args.run_root.expanduser().resolve()
    # Keep generated images, model caches and results outside the source release.
    if runtime == HERE or HERE in runtime.parents:
        raise RuntimeError('--run-root must be outside controlled/ to keep the source package immutable.')
    return legacy, runtime


def verify_runtime_sources(runtime):
    expected = {p.relative_to(HERE / 'frozen_src').as_posix(): sha(p)
                for p in (HERE / 'frozen_src').rglob('*.py')}
    actual = {p.relative_to(runtime / 'frozen_src').as_posix(): sha(p)
              for p in (runtime / 'frozen_src').rglob('*.py')}
    if actual != expected:
        raise RuntimeError('Runtime frozen source set differs from the distributed dependency closure.')


def set_data_root(legacy):
    os.environ['DA_HCR_DATA_ROOT'] = str(legacy / 'data')


def prepare(args, registry):
    legacy, runtime = runtime_paths(args)
    context = {'legacy_root': str(legacy), 'package_sha256': sha(HERE / 'PACKAGE_SHA256.json'),
               'purpose': 'new local audit of unchanged selected historical numerical source'}
    context_path = runtime / 'PORTABLE_CONTEXT.json'
    if context_path.exists() and read(context_path) != context:
        raise RuntimeError('Run directory belongs to another source/data layout; choose a fresh --run-root.')
    # Fail on missing/wrong weights before loading trusted pickle checkpoints.
    for entry in registry['checkpoints']:
        path = legacy / entry['relative_path']
        if not path.is_file():
            raise FileNotFoundError(f'Required historical checkpoint is external and missing: {path}')
        if sha(path) != entry['sha256']:
            raise RuntimeError(f'Wrong historical checkpoint: {entry["dataset"]}/{entry["method"]}')
    for ds, entry in registry['datasets'].items():
        original = entry['original']
        install_missing_exact(HERE / original['path'],
            legacy / 'data/splits' / ds / 'seed_42/split_manifest.csv', original['sha256'])
        screened = entry['screened']
        install_missing_exact(HERE / screened['path'],
            runtime / 'manifests' / ds / 'seed_42/split_manifest.csv', screened['sha256'])
    for source in sorted((HERE / 'frozen_src').rglob('*.py')):
        install_missing_exact(source, runtime/'frozen_src'/source.relative_to(HERE/'frozen_src'), sha(source))
    verify_runtime_sources(runtime)
    save(context_path, context)
    # The original preflight regenerates paths/counts/source hashes, resolves all
    # images, checks transform identity, and strictly loads all six checkpoints.
    audit = load_module('preflight_remote', HERE / 'code/preflight_remote.py')
    audit.ROOT, audit.LEGACY, audit.DATA = runtime, legacy, legacy / 'data'
    set_data_root(legacy)  # original imports contain the historical default path
    with arguments(['preflight_remote.py', '--hash-images']):
        audit.main()
    b1 = read(runtime / 'artifacts/B1_preflight.json')
    for entry in b1['checkpoints']:
        frozen = next(item for item in registry['checkpoints']
                      if (item['dataset'], item['method']) == (entry['dataset'], entry['method']))
        if entry['sha256'] != frozen['sha256'] or entry['epoch'] != frozen['epoch']:
            raise RuntimeError('Audit found a different historical checkpoint identity.')
        if entry['num_classes'] != frozen['num_classes']:
            raise RuntimeError('Historical checkpoint class count changed.')
        for key, value in frozen['identity_args'].items():
            if entry['args'].get(key) != value:
                raise RuntimeError(f'Historical checkpoint argument mismatch: {key}')
    # Decode every original-size RGB image and apply the original group rules.
    # The generated retained rows must match the distributed screened manifests;
    # no stored success flags are substituted for a local data audit.
    cleaner = load_module('build_clean_manifests', HERE / 'code/build_clean_manifests.py')
    cleaner.run(runtime, legacy / 'data')
    clean = read(runtime / 'artifacts/B1_clean_manifest_report.json')
    for ds, entry in registry['datasets'].items():
        if clean['datasets'][ds]['sha256'] != entry['screened']['sha256']:
            raise RuntimeError(f'Decoded-RGB screening does not recreate the fixed manifest: {ds}')
        if clean['datasets'][ds]['counts'] != entry['screened_counts']:
            raise RuntimeError(f'Screened role counts differ: {ds}')
    print('CONTROLLED_PREPARATION_COMPLETE', runtime)


def evaluator(args):
    legacy, runtime = runtime_paths(args)
    context = read(runtime / 'PORTABLE_CONTEXT.json')
    if context['legacy_root'] != str(legacy) or context['package_sha256'] != sha(HERE / 'PACKAGE_SHA256.json'):
        raise RuntimeError('Prepared source or data layout changed; use a new audited run directory.')
    verify_runtime_sources(runtime)
    sys.path.insert(0, str(HERE / 'code'))
    module = load_module('eval_remote', HERE / 'code/eval_remote.py')
    module.ROOT, module.LEGACY, module.DATA = runtime, legacy, legacy / 'data'
    set_data_root(legacy)
    return module


def run_phase(args):
    module = evaluator(args)
    values = ['eval_remote.py', '--phase', args.phase, '--manifest-mode', args.manifest_mode,
              '--run-id', args.run_id, '--batch-size', str(args.batch_size),
              '--workers', str(args.workers), '--datasets', *args.datasets,
              '--methods', *args.methods]
    with arguments(values):
        module.main()


def approve_previews(args):
    if not args.confirm_inspected or not args.reviewer.strip():
        raise RuntimeError('Inspect every generated real-image composite first; supply --confirm-inspected and --reviewer.')
    module = evaluator(args)
    _, runtime = runtime_paths(args)
    root = runtime / 'artifacts/B2/validation' / args.manifest_mode / args.run_id
    report_path = root / 'validation_report.json'
    report = read(report_path)
    if report.get('status') != 'technical_validation_pass' or report.get('errors'):
        raise RuntimeError('Transformation validation must pass before recording visual inspection.')
    previews = {}
    for image in report['images']:
        image_path = Path(image['composite_preview'])
        if sha(image_path) != image['preview_sha256']:
            raise RuntimeError(f'Preview changed: {image_path}')
        if image_path.name in previews:
            raise RuntimeError('Duplicate preview names.')
        previews[image_path.name] = image['preview_sha256']
    module.save_json(root / 'visual_review.json', {
        'status': 'visual_inspection_pass', 'reviewer': args.reviewer.strip(),
        'reviewed_utc': module.utc(), 'validation_report_sha256': sha(report_path),
        'preview_sha256': previews,
        'declaration': 'Reviewer explicitly confirms inspection of every listed real-image composite; no test-score based condition changes.'})
    print('VISUAL_REVIEW_RECORDED', root / 'visual_review.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'run', 'approve-previews'):
        child = sub.add_parser(command)
        child.add_argument('--legacy-root', type=Path, required=True,
                           help='External project directory containing data/{raw,splits} and outputs/formal_v12.')
        child.add_argument('--run-root', type=Path, required=True,
                           help='External generated-audit/cache/result directory; not controlled/.')
        if command != 'prepare':
            child.add_argument('--manifest-mode', choices=('original', 'clean'), default='clean')
            child.add_argument('--run-id', default='controlled_v1')
        if command == 'run':
            child.add_argument('--phase', choices=('validation', 'smoke', 'full'), required=True)
            child.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
            child.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
            child.add_argument('--batch-size', type=int, default=128)
            child.add_argument('--workers', type=int, default=4)
        if command == 'approve-previews':
            child.add_argument('--reviewer', required=True)
            child.add_argument('--confirm-inspected', action='store_true')
    args = parser.parse_args()
    registry = verify_package()
    if args.command == 'prepare':
        prepare(args, registry)
    elif args.command == 'run':
        run_phase(args)
    else:
        approve_previews(args)


if __name__ == '__main__':
    main()

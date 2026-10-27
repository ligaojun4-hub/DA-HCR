#!/usr/bin/env python3
"""B2: frozen-checkpoint evaluation, not training or model selection.

Perspective conditions are synthetic planar projective distortions inspired by
Zhukov et al.; they are not an exact reproduction or a real 3-D viewpoint study.
Validation uses predetermined train/validation images and never test mAP. Full
evaluation retains negative outcomes and never adjusts strengths to fit scores.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path('/root/autodl-tmp/robust_hash_project_scaffold/robust_hash_project_scaffold')
DATA = LEGACY / 'data'
os.environ['DA_HCR_DATA_ROOT'] = str(DATA)
sys.path.insert(0, str(ROOT / 'frozen_src'))

import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageDraw
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize, to_tensor

from robust_hash.data.datasets import resolve_image_path
from robust_hash.data.transforms import (
    ACTIVE_PROTOCOL, DEG_PROTOCOLS, IMAGENET_MEAN, IMAGENET_STD,
    build_image_tensor,
)
from robust_hash.evaluation.retrieval import compute_map_from_codes
from robust_hash.models.hash_model import RobustHashModel
import revision_protocol as protocol

DATASETS = ('caltech256', 'mit_indoor67', 'flowers102')
METHODS = ('base', 'dasupcon')
SIZE = 224
NOTICE = ('Synthetic planar projective distortions inspired by Zhukov et al.; '
          'not exact reproduction and not physical 3-D viewpoint changes.')
CHECKPOINT_NOTICE = (
    'Historical checkpoints were trained and selected on the original split, '
    'which contains duplicate-image contamination. Evaluating them on a cleaned '
    'manifest is a diagnostic migration only, not independently retrained or '
    'leakage-free evidence. Fresh training on the frozen clean split is required.'
)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value):
    raw = json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(plain(data), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    with temp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: plain(row.get(k)) for k in fieldnames})
    os.replace(temp, path)


def spec_id(spec):
    return str(spec['name'])


def spec_type(spec):
    return str(spec['kind'])


def spec_parameters(spec):
    return spec.get('params', spec)


def evaluation_specs(manifest_mode):
    if manifest_mode == 'original':
        return [s for s in protocol.SPECS if 'standard_severity' in s]
    return list(protocol.SPECS)


def runtime_versions():
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'torchvision': torchvision.__version__, 'numpy': np.__version__,
            'pillow': PIL.__version__, 'pandas': pd.__version__}


def source_fingerprint(b1_path, b1):
    frozen = ROOT / 'frozen_src'
    expected = b1['source_files']
    actual = {str(p.relative_to(frozen)): sha(p)
              for p in sorted(frozen.rglob('*.py'))}
    if actual != expected:
        raise RuntimeError('frozen_src differs from B1; repeat B1 before evaluating.')
    return {'B1_sha256': sha(b1_path), 'evaluator_sha256': sha(__file__),
            'protocol_sha256': sha(protocol.__file__),
            'frozen_source_digest': digest(actual), 'frozen_source_files': actual,
            'specs_digest': digest(protocol.SPECS),
            'old_protocol_digest': digest(DEG_PROTOCOLS),
            'versions': runtime_versions()}


def load_audit():
    path = ROOT / 'artifacts' / 'B1_preflight.json'
    b1 = read_json(path)
    if b1.get('status') != 'technical_audit_complete' or b1.get('errors'):
        raise RuntimeError('B1 technical audit is incomplete or contains errors.')
    if ACTIVE_PROTOCOL != 'formal_v12_monotonic':
        raise RuntimeError('Unexpected old degradation protocol.')
    pairs = {(x['dataset'], x['method']) for x in b1['checkpoints']}
    if pairs != {(d, m) for d in DATASETS for m in METHODS}:
        raise RuntimeError('B1 must verify all six source checkpoints.')
    for ds in DATASETS:
        entry = b1['datasets'][ds]
        if not entry['clean_identity_pass'] or not entry['resolved_all_images']:
            raise RuntimeError(f'B1 data checks not passed: {ds}')
        if sha(entry['manifest']) != entry['sha256']:
            raise RuntimeError(f'Manifest changed since B1: {ds}')
    return b1, source_fingerprint(path, b1)


def manifest_context(args, b1):
    """Bind the exact evaluated split; never silently fall back to old data."""
    result = {'mode': args.manifest_mode, 'datasets': {},
              'checkpoint_interpretation': CHECKPOINT_NOTICE,
              'archival_metric_comparison_enabled': args.manifest_mode == 'original'}
    clean_report = None
    if args.manifest_mode == 'clean':
        report_path = ROOT / 'artifacts' / 'B1_clean_manifest_report.json'
        clean_report = read_json(report_path)
        if clean_report.get('status') != 'clean_manifests_complete':
            raise RuntimeError('Clean manifests must pass their dedicated audit first.')
        result['clean_manifest_report_sha256'] = sha(report_path)
    # Keep context stable when a long run is resumed one dataset at a time.
    for ds in DATASETS:
        original = b1['datasets'][ds]
        if clean_report is None:
            entry = {'manifest': original['manifest'], 'sha256': original['sha256'],
                     'source_sha256': original['sha256']}
        else:
            entry = clean_report['datasets'][ds]
            expected = ROOT / 'manifests' / ds / 'seed_42' / 'split_manifest.csv'
            if Path(entry['manifest']).resolve() != expected.resolve():
                raise RuntimeError(f'Unexpected clean manifest location: {ds}')
            if entry['source_sha256'] != original['sha256']:
                raise RuntimeError(f'Clean manifest derives from a different original split: {ds}')
        if sha(entry['manifest']) != entry['sha256']:
            raise RuntimeError(f'Evaluation manifest hash mismatch: {ds}')
        frame = pd.read_csv(entry['manifest'])
        required = {'rel_path', 'label', 'split'}
        if not required.issubset(frame.columns) or frame.empty:
            raise RuntimeError(f'Invalid manifest schema: {ds}')
        if frame['rel_path'].astype(str).duplicated().any():
            raise RuntimeError(f'Duplicate image IDs in evaluation manifest: {ds}')
        counts = frame['split'].value_counts().to_dict()
        if not all(counts.get(role, 0) > 0 for role in ('train', 'val', 'test_query', 'test_retrieval')):
            raise RuntimeError(f'Missing roles in evaluation manifest: {ds}')
        result['datasets'][ds] = {'manifest': str(entry['manifest']),
            'sha256': entry['sha256'], 'source_sha256': entry['source_sha256'],
            'counts': counts}
    return plain(result)


def phase_root(args, phase):
    return ROOT / 'artifacts' / 'B2' / phase / args.manifest_mode / args.run_id


def diverse_qa_rows(frame, ds, role, count=3):
    """Hash-ranked examples from distinct labels, fixed without examining scores."""
    rows = frame[frame['split'] == role].to_dict('records')
    rows.sort(key=lambda row: hashlib.sha256(
        f'JEI_technical_QA_v1|{ds}|{role}|{row["rel_path"]}'.encode('utf-8')).hexdigest())
    selected, seen = [], set()
    for row in rows:
        if int(row['label']) not in seen:
            selected.append(row)
            seen.add(int(row['label']))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f'Need {count} distinct labels for validation QA: {ds}/{role}')
    return selected


def condition_metadata(spec):
    info = {'condition': spec_id(spec), 'spec': spec, 'image_size': SIZE,
            'scientific_scope': NOTICE}
    if 'standard_severity' in spec:
        severity = spec['standard_severity']
        info.update({'operation_frame': 'native image dimensions, then original tensor pipeline',
                     'old_protocol': ACTIVE_PROTOCOL,
                     'old_parameters': DEG_PROTOCOLS[ACTIVE_PROTOCOL][severity],
                     'noise_key': 'original resolved absolute path + severity + protocol'})
    else:
        info['operation_frame'] = 'PIL RGB bilinear 224x224 work image, then ToTensor/ImageNetNormalize'
        if spec_type(spec) == 'perspective':
            params = spec_parameters(spec)
            H = protocol.perspective_homography(SIZE, SIZE, params['rho'], params['direction'])
            info['H_source_to_output'] = plain(H)
            info['H_output_to_source'] = plain(np.linalg.inv(np.asarray(H)))
            info['boundary_sampling'] = protocol.sampling_boundary_report(H, SIZE, SIZE)
    return info


def make_tensor(image_path, spec):
    if 'standard_severity' in spec:
        return build_image_tensor(image_path, SIZE, spec['standard_severity'])
    image = protocol.apply_new_pil(image_path, spec, image_size=SIZE)
    return normalize(to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)


def tensor_pil(tensor):
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


def validate_images(args, b1, fingerprint):
    """Deterministic technical QA only: no network, test query, or mAP."""
    out = phase_root(args, 'validation')
    existing = out / 'validation_report.json'
    if existing.exists():
        previous = read_json(existing)
        if previous['source_fingerprint'] != plain(fingerprint):
            raise RuntimeError('Validation directory uses different source hashes; use a new run-id.')
        covered = {(x['dataset'], x['role']) for x in previous.get('images', [])}
        requested = {(ds, role) for ds in args.datasets for role in ('train', 'val')}
        if previous.get('status') == 'technical_validation_pass' and requested.issubset(covered):
            print('VALIDATION_ALREADY_COMPLETE', existing, flush=True)
            return
    condition_specs = evaluation_specs(args.manifest_mode)
    report = {'phase': 'validation', 'status': 'running', 'started_utc': utc(),
              'uses_test_queries': False, 'uses_test_map': False,
              'strength_selection': 'fixed before testing; validation images only',
              'source_fingerprint': fingerprint, 'scientific_scope': NOTICE,
              'conditions': [condition_metadata(s) for s in condition_specs],
              'manifest_mode': args.manifest_mode,
              'checkpoint_interpretation': CHECKPOINT_NOTICE,
              'qa_selection_rule': 'SHA256 rank of JEI_technical_QA_v1|dataset|role|rel_path; first three distinct labels per train/val role',
              'images': [], 'errors': []}
    first_perspective = next(s for s in protocol.SPECS if spec_type(s) == 'perspective')
    identity_spec = copy.deepcopy(first_perspective)
    identity_spec['name'] = 'technical_rho_zero_identity'
    if 'params' in identity_spec:
        identity_spec['params']['rho'] = 0.0
    else:
        identity_spec['rho'] = 0.0
    for ds in args.datasets:
        frame = pd.read_csv(fingerprint['evaluation_manifest_context']['datasets'][ds]['manifest'])
        selected = [(role, index, row) for role in ('train', 'val')
                    for index, row in enumerate(diverse_qa_rows(frame, ds, role), start=1)]
        for role, sample_index, row in selected:
            path = resolve_image_path(row, ds)
            print('VALIDATE', ds, role, row['rel_path'], flush=True)
            with Image.open(path) as opened:
                work = opened.convert('RGB').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
            identity = protocol.apply_new_pil(work, identity_spec, image_size=SIZE)
            identity_equal = bool(np.array_equal(np.asarray(work), np.asarray(identity)))
            clean_equal = bool(torch.equal(
                build_image_tensor(path, SIZE, 'clean'),
                normalize(to_tensor(work), IMAGENET_MEAN, IMAGENET_STD)))
            tile_w, tile_h, columns = 250, 275, 5
            rows = math.ceil(len(condition_specs) / columns)
            panel = Image.new('RGB', (columns * tile_w, rows * tile_h + 38), 'white')
            draw = ImageDraw.Draw(panel)
            draw.text((8, 7), f'{ds} / {role} / sample {sample_index} / label {row["label"]}; technical QA, not results', fill='black')
            checks = []
            for n, spec in enumerate(condition_specs):
                tensor = make_tensor(path, spec)
                second = make_tensor(path, spec)
                repeatable = bool(torch.equal(tensor, second))
                finite = bool(torch.isfinite(tensor).all())
                shape_ok = tuple(tensor.shape) == (3, SIZE, SIZE)
                image = tensor_pil(tensor)
                arr = np.asarray(image, dtype=np.float32)
                energy = float(np.square(np.diff(arr, axis=0)).mean() +
                               np.square(np.diff(arr, axis=1)).mean())
                checks.append({'name': spec_id(spec), 'repeatable': repeatable,
                               'finite': finite, 'shape_ok': shape_ok,
                               'rgb_mean': float(arr.mean()), 'gradient_energy': energy})
                x = (n % columns) * tile_w + 7
                y = (n // columns) * tile_h + 38
                image.thumbnail((224, 224), Image.Resampling.BILINEAR)
                panel.paste(image, (x, y))
                draw.text((x, y + 228), spec_id(spec), fill='black')
            png = out / 'composite_previews' / f'{ds}_{role}_{sample_index:02d}.png'
            png.parent.mkdir(parents=True, exist_ok=True)
            panel.save(png)
            item = {'dataset': ds, 'role': role, 'rel_path': row['rel_path'],
                    'label': int(row['label']), 'sample_index': sample_index,
                    'resolved_absolute_path': path, 'image_sha256': sha(path),
                    'identity_rho_zero_equal': identity_equal,
                    'new_workframe_vs_original_clean_tensor_equal': clean_equal,
                    'conditions': checks, 'composite_preview': str(png),
                    'preview_sha256': sha(png)}
            report['images'].append(item)
            if not identity_equal or not clean_equal or not all(
                    c['repeatable'] and c['finite'] and c['shape_ok'] for c in checks):
                report['errors'].append({'dataset': ds, 'role': role, 'technical_checks_failed': True})
            save_json(existing, report)
    report['status'] = 'technical_validation_fail' if report['errors'] else 'technical_validation_pass'
    report['finished_utc'] = utc()
    report['visual_review_note'] = 'Inspect the complete composite previews before full evaluation; no metric-based severity changes.'
    report['visual_review_status'] = 'pending_human_or_agent_image_inspection'
    save_json(existing, report)
    if report['errors']:
        raise RuntimeError(f'Transformation technical QA failed: {existing}')
    print('B2_TRANSFORM_VALIDATION_COMPLETE', existing, flush=True)


class ConditionDataset(Dataset):
    def __init__(self, frame, dataset, spec):
        self.rows = frame.to_dict('records')
        self.dataset, self.spec = dataset, spec

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = resolve_image_path(row, self.dataset)
        return make_tensor(path, self.spec), int(row['label']), str(row['rel_path']), str(path)


def empty_stats():
    return {'n': 0, 'elements': 0, 'sum': 0.0, 'sum_sq': 0.0,
            'min': float('inf'), 'max': -float('inf'), 'zero_count': 0,
            'near_zero_count_abs_lt_1e_minus_3': 0, 'norm_sum': 0.0}


@torch.no_grad()
def extract(model, frame, ds, spec, args):
    loader = DataLoader(ConditionDataset(frame, ds, spec), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=True)
    chunks, legacy, labels, ids, paths = [], [], [], [], []
    stats = empty_stats()
    start = time.monotonic()
    model.eval()
    for i, (images, y, rel, absolute) in enumerate(loader):
        outputs = model.forward_single(images.to('cuda', non_blocking=True))
        z = outputs['z'].float()
        if not torch.isfinite(z).all():
            raise RuntimeError('Nonfinite continuous activations; preserve failure and investigate.')
        binary = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
        chunks.append(binary.to(torch.int8).cpu().numpy())
        legacy.append(outputs['b'].float().cpu().numpy())
        labels.append(y.numpy().astype(np.int64))
        ids.extend(rel)
        paths.extend(absolute)
        stats['n'] += len(y)
        stats['elements'] += z.numel()
        stats['sum'] += z.double().sum().item()
        stats['sum_sq'] += z.double().square().sum().item()
        stats['min'] = min(stats['min'], z.min().item())
        stats['max'] = max(stats['max'], z.max().item())
        stats['zero_count'] += int((z == 0).sum().item())
        stats['near_zero_count_abs_lt_1e_minus_3'] += int((z.abs() < 1e-3).sum().item())
        stats['norm_sum'] += z.norm(dim=1).double().sum().item()
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(loader):
            print(f'EXTRACT {ds} {spec_id(spec)} {stats["n"]}/{len(frame)} '
                  f'{time.monotonic()-start:.1f}s', flush=True)
    if not stats['n']:
        raise RuntimeError('Refuse extraction from an empty role.')
    count = stats['elements']
    mean = stats['sum'] / count
    stats['continuous_mean'] = mean
    stats['continuous_std'] = math.sqrt(max(0., stats['sum_sq'] / count - mean * mean))
    stats['mean_l2_norm'] = stats['norm_sum'] / stats['n']
    stats['elapsed_seconds'] = time.monotonic() - start
    return {'codes': np.concatenate(chunks), 'legacy_ste_codes': np.concatenate(legacy),
            'labels': np.concatenate(labels), 'ids': np.asarray(ids, dtype=str),
            'resolved_paths': np.asarray(paths, dtype=str)}, stats


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def cached_codes(path, metadata_path, context_hash, expected_ids):
    if not Path(path).exists() or not Path(metadata_path).exists():
        return None
    meta = read_json(metadata_path)
    if meta['context_hash'] != context_hash or meta['npz_sha256'] != sha(path):
        raise RuntimeError(f'Cache hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as payload:
        arrays = {k: payload[k] for k in payload.files}
    if not np.array_equal(arrays['ids'], np.asarray(expected_ids, dtype=str)):
        raise RuntimeError(f'Cache query/gallery ordering mismatch: {path}')
    if arrays['codes'].dtype != np.int8 or not np.isin(arrays['codes'], [-1, 1]).all():
        raise RuntimeError(f'Invalid binary code cache: {path}')
    return arrays, meta['activation_diagnostics']


def get_codes(model, frame, ds, spec, args, out, context_hash):
    npz, meta = out / 'codes.npz', out / 'codes_metadata.json'
    cached = cached_codes(npz, meta, context_hash, frame['rel_path'].astype(str).tolist())
    if cached is not None:
        print('REUSE_CODE_CACHE', npz, flush=True)
        return cached
    arrays, diagnostics = extract(model, frame, ds, spec, args)
    save_npz(npz, arrays)
    save_json(meta, {'status': 'codes_complete', 'context_hash': context_hash,
                     'condition': condition_metadata(spec), 'npz_sha256': sha(npz),
                     'activation_diagnostics': diagnostics, 'completed_utc': utc()})
    return arrays, diagnostics


def legacy_score(query, gallery):
    return compute_map_from_codes(
        query_codes=torch.from_numpy(query['legacy_ste_codes']),
        query_labels=torch.from_numpy(query['labels']),
        gallery_codes=torch.from_numpy(gallery['legacy_ste_codes']),
        gallery_labels=torch.from_numpy(gallery['labels']),
        query_paths=query['resolved_paths'].tolist(),
        gallery_paths=gallery['resolved_paths'].tolist(), exclude_self=True)


def verify_completed(path, context_hash):
    if not path.exists():
        return None
    result = read_json(path)
    if result.get('status') != 'condition_complete':
        return None
    if result['context_hash'] != context_hash:
        raise RuntimeError(f'Completed result uses different sources: {path}')
    for name, expected in result['artifact_sha256'].items():
        if sha(path.parent / name) != expected:
            raise RuntimeError(f'Completed result artifact changed: {path.parent/name}')
    return result


def validate_checkpoint(entry, device='cuda'):
    if sha(entry['path']) != entry['sha256']:
        raise RuntimeError(f'Checkpoint changed since B1: {entry["path"]}')
    payload = torch.load(entry['path'], map_location='cpu', weights_only=False)
    state, args = payload['model_state_dict'], payload['args']
    if not any(k.startswith('backbone.layer1.') for k in state):
        raise RuntimeError('Checkpoint is not a ResNet18 state dict.')
    if any(k.startswith('backbone.features.') for k in state):
        raise RuntimeError('Fallback CNN state dict refused.')
    for key in ('dataset', 'baseline', 'method', 'split_seed', 'hash_bits'):
        if args.get(key) != entry['args'].get(key):
            raise RuntimeError(f'Checkpoint argument changed: {key}')
    if args['baseline'] != 'hashnet' or int(args['split_seed']) != 42 or int(args['hash_bits']) != 64:
        raise RuntimeError('Wrong frozen-checkpoint configuration.')
    model = RobustHashModel(int(payload['num_classes']), 64, 'resnet18', False)
    if model.backbone.__class__.__name__ != 'ResNet' or not hasattr(model.backbone, 'layer1'):
        raise RuntimeError('torchvision ResNet construction fell back to a small CNN.')
    model.load_state_dict(state, strict=True)
    del payload, state
    gc.collect()
    return model.to(device).eval()


@torch.no_grad()
def cpu_smoke(args, b1, fingerprint):
    """One train image through one strict historical checkpoint; never test mAP."""
    out = phase_root(args, 'smoke')
    report_path = out / 'cpu_smoke_report.json'
    if report_path.exists():
        previous = read_json(report_path)
        if previous['source_fingerprint'] != plain(fingerprint):
            raise RuntimeError('CPU smoke record uses different sources; use a new run-id.')
        if previous.get('status') == 'cpu_smoke_pass':
            print('CPU_SMOKE_ALREADY_COMPLETE', report_path, flush=True)
            return
    ds, method = args.datasets[0], args.methods[0]
    frame = pd.read_csv(fingerprint['evaluation_manifest_context']['datasets'][ds]['manifest'])
    row = diverse_qa_rows(frame, ds, 'train')[0]
    entry = next(x for x in b1['checkpoints'] if (x['dataset'], x['method']) == (ds, method))
    image_path = resolve_image_path(row, ds)
    spec = next(s for s in protocol.SPECS if s.get('standard_severity') == 'clean')
    print('CPU_SMOKE_FORWARD', ds, method, row['rel_path'], flush=True)
    model = validate_checkpoint(entry, device='cpu')
    start = time.monotonic()
    outputs = model.forward_single(make_tensor(image_path, spec).unsqueeze(0))
    finite = all(bool(torch.isfinite(t).all()) for t in outputs.values())
    shapes = {name: list(t.shape) for name, t in outputs.items()}
    valid_shapes = shapes.get('z') == [1, 64] and shapes.get('b') == [1, 64]
    binary = torch.where(outputs['z'] >= 0, 1, -1).to(torch.int8).numpy()
    report = {'status': 'cpu_smoke_pass' if finite and valid_shapes else 'cpu_smoke_fail',
              'completed_utc': utc(), 'source_fingerprint': fingerprint,
              'manifest_mode': args.manifest_mode, 'checkpoint_interpretation': CHECKPOINT_NOTICE,
              'dataset': ds, 'method': method, 'checkpoint_sha256': entry['sha256'],
              'rel_path': row['rel_path'], 'image_sha256': sha(image_path),
              'uses_test_queries': False, 'uses_test_map': False, 'shapes': shapes,
              'all_outputs_finite': finite, 'expected_code_shapes': valid_shapes,
              'binary_code': binary.tolist(), 'forward_seconds': time.monotonic() - start}
    save_json(report_path, report)
    del outputs, model
    gc.collect()
    if report['status'] != 'cpu_smoke_pass':
        raise RuntimeError(f'CPU forward failed: {report_path}')
    print('CPU_SMOKE_COMPLETE', report_path, flush=True)


def evaluate_condition(query, gallery, clean, spec, entry, diagnostics, out, context_hash, manifest_mode):
    start = time.monotonic()
    measured = protocol.evaluate_codes(query['codes'], query['labels'],
        gallery['codes'], gallery['labels'], query['ids'], gallery['ids'],
        exclude_self=True, chunk_size=256)
    ap = np.asarray(measured['per_query_ap'], dtype=np.float64)
    relevant = np.asarray(measured['relevant_counts'])
    bit = protocol.bit_flip_diagnostics(clean['codes'], query['codes'],
                                       clean['ids'], query['ids'])
    # The protocol returns per-query arrays alongside aggregate diagnostics.
    bit_arrays = {k: v for k, v in bit.items()
                  if isinstance(v, (np.ndarray, list, tuple)) and np.asarray(v).shape == (len(ap),)}
    bit_totals = {k: v for k, v in bit.items() if k not in bit_arrays}
    ap_csv = out / 'per_query_ap.csv'
    save_csv(ap_csv, ['query_id', 'label', 'relevant_count', 'AP', 'used_in_map'],
             ({'query_id': str(q), 'label': int(y), 'relevant_count': int(r),
               'AP': float(a) if math.isfinite(a) else None, 'used_in_map': bool(math.isfinite(a))}
              for q, y, r, a in zip(query['ids'], query['labels'], relevant, ap)))
    bit_csv = out / 'per_query_bit_changes.csv'
    bit_keys = sorted(bit_arrays)
    save_csv(bit_csv, ['query_id'] + bit_keys,
             ({'query_id': str(q), **{k: np.asarray(bit_arrays[k])[i] for k in bit_keys}}
              for i, q in enumerate(query['ids'])))
    old = archive = None
    if manifest_mode == 'original' and 'standard_severity' in spec:
        print('LEGACY_METRIC_RECHECK', entry['dataset'], entry['method'], spec_id(spec), flush=True)
        old = legacy_score(query, gallery)
        archived = entry.get('archived_test_metrics') or {}
        archive = (archived.get('results') or {}).get('mAP_' + spec['standard_severity'])
    aggregates = {k: v for k, v in measured.items() if k not in ('per_query_ap', 'relevant_counts')}
    result = {'status': 'condition_complete', 'completed_utc': utc(), 'phase': 'full',
              'dataset': entry['dataset'], 'method': entry['method'],
              'condition': spec_id(spec), 'condition_type': spec_type(spec),
              'manifest_mode': manifest_mode, 'checkpoint_interpretation': CHECKPOINT_NOTICE,
              'parameters': condition_metadata(spec), 'context_hash': context_hash,
              'checkpoint_sha256': entry['sha256'], 'checkpoint_epoch': entry['epoch'],
              'metrics': aggregates, 'binary_code_diagnostics': protocol.code_diagnostics(query['codes']),
              'activation_diagnostics': diagnostics, 'bit_change_diagnostics': bit_totals,
              'legacy_torch_metrics': old, 'archived_map': archive,
              'legacy_minus_archive': None if old is None or archive is None else old['mAP'] - archive,
              'stable_minus_legacy': None if old is None or measured['mAP'] is None else measured['mAP'] - old['mAP'],
              'metric_seconds': time.monotonic() - start,
              'outcome_policy': 'All conditions retained; no strength changes or discarded negative outcomes.',
              'scientific_scope': NOTICE,
              'artifact_sha256': {p.name: sha(p) for p in [out/'codes.npz', out/'codes_metadata.json', ap_csv, bit_csv]}}
    save_json(out / 'result.json', result)
    return plain(result)


def summaries(root, manifest_mode):
    results = [read_json(p) for p in sorted(root.rglob('result.json'))]
    results = [r for r in results if r.get('status') == 'condition_complete']
    identities = [(r['dataset'], r['method'], r['condition']) for r in results]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f'Duplicate result identities under {root}; do not mix checkpoints or runs.')
    if any(r.get('manifest_mode') != manifest_mode for r in results):
        raise RuntimeError('Refuse combining original and cleaned-manifest results.')
    condition_specs = evaluation_specs(manifest_mode)
    fields = ['dataset', 'method', 'manifest_mode', 'checkpoint_epoch', 'condition', 'condition_type',
              'mAP', 'num_query_used', 'num_query_total', 'num_query_without_relevant',
              'legacy_torch_mAP', 'archived_map', 'legacy_minus_archive', 'stable_minus_legacy',
              'checkpoint_sha256']
    rows = []
    for r in results:
        row = {k: r.get(k) for k in fields}
        row.update({k: r['metrics'].get(k) for k in ('mAP', 'num_query_used', 'num_query_total', 'num_query_without_relevant')})
        row['legacy_torch_mAP'] = (r.get('legacy_torch_metrics') or {}).get('mAP')
        rows.append(row)
    save_csv(root / 'summary.csv', fields, rows)
    summary_rows = []
    for ds, method in sorted({(r['dataset'], r['method']) for r in results}):
        group = [r for r in results if (r['dataset'], r['method']) == (ds, method)]
        by_id = {r['condition']: r for r in group}
        row = {'dataset': ds, 'method': method, 'manifest_mode': group[0]['manifest_mode'],
               'completed_conditions': len(group), 'expected_conditions': len(condition_specs),
               'all_conditions_complete': {spec_id(s) for s in condition_specs}.issubset(by_id),
               'checkpoint_interpretation': CHECKPOINT_NOTICE}
        for family in ('perspective', 'blur', 'brightness'):
            expected = {spec_id(s) for s in condition_specs if spec_type(s) == family}
            available = expected & set(by_id)
            values = [by_id[name]['metrics']['mAP'] for name in sorted(available)]
            finite_values = [value for value in values if value is not None and math.isfinite(value)]
            complete = bool(expected) and available == expected and len(finite_values) == len(expected)
            row[family + '_expected_conditions'] = len(expected)
            row[family + '_completed_conditions'] = len(available)
            row[family + '_valid_conditions'] = len(finite_values)
            row[family + '_complete'] = complete
            # Do not present a partial-condition mean as the registered family outcome.
            row[family + '_mean_mAP'] = float(np.mean(finite_values)) if complete else None
        old_rows = [r for r in group if 'standard_severity' in r['parameters']['spec']]
        for r in old_rows:
            severity = r['parameters']['spec']['standard_severity']
            row['old_' + severity + '_stable_mAP'] = r['metrics']['mAP']
        old_degraded = [row.get('old_' + s + '_stable_mAP') for s in ('weak', 'medium', 'strong')]
        row['old_Ravg_stable'] = float(np.mean(old_degraded)) if all(v is not None for v in old_degraded) else None
        summary_rows.append(row)
    model_fields = ['dataset', 'method', 'manifest_mode', 'completed_conditions', 'expected_conditions',
                    'all_conditions_complete', 'old_clean_stable_mAP',
                    'old_weak_stable_mAP', 'old_medium_stable_mAP', 'old_strong_stable_mAP',
                    'old_Ravg_stable']
    for family in ('perspective', 'blur', 'brightness'):
        model_fields.extend(family + suffix for suffix in (
            '_expected_conditions', '_completed_conditions', '_valid_conditions', '_complete', '_mean_mAP'))
    model_fields.append('checkpoint_interpretation')
    save_csv(root / 'model_summary.csv', model_fields, summary_rows)
    return results


def full_evaluation(args, b1, fingerprint):
    if not torch.cuda.is_available():
        raise RuntimeError('Full evaluation requires CUDA; CPU fallback is refused.')
    validation_path = phase_root(args, 'validation') / 'validation_report.json'
    validation = read_json(validation_path)
    if validation.get('status') != 'technical_validation_pass' or validation['source_fingerprint'] != plain(fingerprint):
        raise RuntimeError('Matching transformation technical validation must pass first.')
    validated_ds = {x['dataset'] for x in validation['images']}
    if not set(args.datasets).issubset(validated_ds):
        raise RuntimeError('Validate every requested dataset before full evaluation.')
    visual_review_path = validation_path.parent / 'visual_review.json'
    visual_review = read_json(visual_review_path)
    if visual_review.get('status') != 'visual_inspection_pass':
        raise RuntimeError('Inspect all real-image composite previews before full evaluation.')
    if visual_review.get('validation_report_sha256') != sha(validation_path):
        raise RuntimeError('Visual inspection refers to a different technical validation report.')
    expected_previews = {Path(item['composite_preview']).name: item['preview_sha256']
                         for item in validation['images']}
    if len(expected_previews) != len(validation['images']):
        raise RuntimeError('Duplicate preview filenames in technical validation report.')
    if visual_review.get('preview_sha256') != expected_previews:
        raise RuntimeError('Visual inspection must cover every preview hash in the validation report.')
    for item in validation['images']:
        if sha(item['composite_preview']) != item['preview_sha256']:
            raise RuntimeError(f'Inspected preview was modified: {item["composite_preview"]}')
    root = phase_root(args, 'full')
    report = {'phase': 'full', 'status': 'running', 'started_utc': utc(),
              'source_fingerprint': fingerprint, 'scientific_scope': NOTICE,
              'manifest_mode': args.manifest_mode, 'checkpoint_interpretation': CHECKPOINT_NOTICE,
              'datasets': args.datasets, 'methods': args.methods,
              'gpu': torch.cuda.get_device_name(0), 'runtime': vars(args),
              'cuda_runtime': torch.version.cuda,
              'cudnn_version': torch.backends.cudnn.version(),
              'float32_matmul_precision': torch.get_float32_matmul_precision(),
              'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
              'validation_report_sha256': sha(validation_path), 'completed_conditions': 0,
              'visual_review_sha256': sha(visual_review_path),
              'uses_full_test_query_and_gallery': True}
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / 'run_report.json'
    if report_path.exists() and read_json(report_path)['source_fingerprint'] != plain(fingerprint):
        raise RuntimeError('Existing run has different source hashes; use a fresh run-id.')
    save_json(report_path, report)
    condition_specs = evaluation_specs(args.manifest_mode)
    clean_spec = next(s for s in condition_specs if s.get('standard_severity') == 'clean')
    start = time.monotonic()
    for ds in args.datasets:
        manifest_entry = fingerprint['evaluation_manifest_context']['datasets'][ds]
        frame = pd.read_csv(manifest_entry['manifest'])
        query_frame = frame[frame['split'] == 'test_query'].reset_index(drop=True)
        gallery_frame = frame[frame['split'] == 'test_retrieval'].reset_index(drop=True)
        for method in args.methods:
            entry = next(x for x in b1['checkpoints'] if x['dataset'] == ds and x['method'] == method)
            model_out = root / ds / method / entry['sha256'][:12]
            model_out.mkdir(parents=True, exist_ok=True)
            context = {'phase': 'full', 'source_fingerprint': fingerprint,
                       'dataset': ds, 'method': method, 'checkpoint_sha256': entry['sha256'],
                       'manifest_mode': args.manifest_mode,
                       'manifest_sha256': manifest_entry['sha256'],
                       'checkpoint_interpretation': CHECKPOINT_NOTICE,
                       'query_role': 'test_query', 'gallery_role': 'test_retrieval',
                       'gallery_order': 'selected manifest row sequence; rel_path secondary Hamming tie',
                       'binary_code': 'z>=0 -> +1; z<0 -> -1, int8', 'image_size': SIZE}
            context_hash = digest(context)
            context_path = model_out / 'context.json'
            if context_path.exists() and read_json(context_path)['context_hash'] != context_hash:
                raise RuntimeError(f'Cache context changed: {model_out}')
            save_json(context_path, {'context_hash': context_hash, **context})
            print('LOAD_VERIFIED_MODEL', ds, method, entry['path'], flush=True)
            model = validate_checkpoint(entry)
            gallery, gallery_diag = get_codes(model, gallery_frame, ds, clean_spec,
                                              args, model_out / 'gallery', context_hash)
            clean, clean_diag = get_codes(model, query_frame, ds, clean_spec,
                                          args, model_out / 'conditions' / spec_id(clean_spec), context_hash)
            save_json(model_out / 'gallery_diagnostics.json', {
                'activation_diagnostics': gallery_diag,
                'binary_code_diagnostics': protocol.code_diagnostics(gallery['codes']),
                'context_hash': context_hash})
            for spec in condition_specs:
                condition_out = model_out / 'conditions' / spec_id(spec)
                previous = verify_completed(condition_out / 'result.json', context_hash)
                if previous is not None:
                    print('SKIP_COMPLETED', ds, method, spec_id(spec), flush=True)
                    continue
                if spec_id(spec) == spec_id(clean_spec):
                    arrays, diagnostics = clean, clean_diag
                else:
                    arrays, diagnostics = get_codes(model, query_frame, ds, spec,
                        args, condition_out, context_hash)
                result = evaluate_condition(arrays, gallery, clean, spec, entry,
                                             diagnostics, condition_out, context_hash, args.manifest_mode)
                print('CONDITION_COMPLETE', ds, method, spec_id(spec),
                      f'mAP={result["metrics"]["mAP"]}', flush=True)
                report['completed_conditions'] = len(summaries(root, args.manifest_mode))
                report['elapsed_seconds'] = time.monotonic() - start
                save_json(report_path, report)
            del model, gallery, clean
            gc.collect()
            torch.cuda.empty_cache()
    all_results = summaries(root, args.manifest_mode)
    required = {(ds, method, spec_id(s)) for ds in args.datasets for method in args.methods for s in condition_specs}
    available = {(r['dataset'], r['method'], r['condition']) for r in all_results}
    report['completed_conditions'] = len(required & available)
    report['expected_conditions'] = len(required)
    report['status'] = 'full_evaluation_complete' if required.issubset(available) else 'incomplete'
    report['finished_utc'] = utc()
    report['elapsed_seconds'] = time.monotonic() - start
    report['interpretation'] = 'Evaluation only; new training and scientific conclusions are not completed by this runner.'
    save_json(report_path, report)
    print('B2_FULL_EVALUATION_COMPLETE', report_path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('validation', 'smoke', 'full'), default='validation')
    parser.add_argument('--pilot', action='store_true', help='Technical validation alias; never evaluates test queries.')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--run-id', default='controlled_v1')
    parser.add_argument('--manifest-mode', choices=('original', 'clean'), default='clean',
                        help='original reproduces four archived conditions; clean evaluates all new conditions using historical weights, not retrained evidence')
    args = parser.parse_args()
    if args.pilot and args.phase != 'validation':
        parser.error('--pilot supports technical validation only, never full test evaluation.')
    if args.batch_size < 1 or args.workers < 0:
        parser.error('Invalid batch size/workers.')
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        parser.error('run-id must contain only letters, digits, underscore, or hyphen.')
    args.datasets = list(dict.fromkeys(args.datasets))
    args.methods = list(dict.fromkeys(args.methods))
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    fingerprint = {}
    try:
        b1, fingerprint = load_audit()
        fingerprint['evaluation_manifest_context'] = manifest_context(args, b1)
        if args.phase == 'validation':
            validate_images(args, b1, fingerprint)
        elif args.phase == 'smoke':
            cpu_smoke(args, b1, fingerprint)
        else:
            full_evaluation(args, b1, fingerprint)
    except Exception as exc:
        failure = ROOT / 'artifacts' / 'B2' / 'failures' / f'{args.manifest_mode}_{args.run_id}_{int(time.time())}_{os.getpid()}.json'
        save_json(failure, {'status': 'technical_failure', 'utc': utc(), 'phase': args.phase,
                           'runtime': vars(args), 'error_type': type(exc).__name__,
                           'error': str(exc), 'source_fingerprint': fingerprint,
                           'outcome_policy': 'Preserved failure; not omitted or classified as a scientific negative.'})
        raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Read-only image scan; write versioned, auditable manifests without deleting images.

Duplicate identity is decoded RGB dimensions+pixels, not a perceptual similarity
threshold. Role retention and conflict rules are fixed before new test scores.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, __version__ as PIL_VERSION

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/root/autodl-tmp/robust_hash_project_scaffold/robust_hash_project_scaffold/data')
DATASETS = ('caltech256', 'mit_indoor67', 'flowers102')
PRIORITY = {'train': 0, 'val': 1, 'test_retrieval': 2, 'test_query': 3}
VERSION = 'rgb-exact-dedup-v1-20261006'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def clean_rows(rows, groups):
    """Group entries are {rel_path}; leave source order and retained roles intact."""
    by_path = {r['rel_path']: r for r in rows}
    assert len(by_path) == len(rows), 'Duplicate relative paths'
    excluded = {}
    group_records = []
    seen = set()
    for index, members in enumerate(groups):
        paths = sorted(m['rel_path'] for m in members)
        assert not seen.intersection(paths), 'Overlapping content groups'
        seen.update(paths)
        assert len(paths) > 1 and all(p in by_path for p in paths)
        group = [by_path[p] for p in paths]
        conflict = len({int(r['label']) for r in group}) > 1
        retained = None if conflict else min(group, key=lambda r: (PRIORITY[r['split']], r['rel_path']))['rel_path']
        reason = 'conflicting_labels_exclude_entire_group' if conflict else 'same_label_keep_one_by_role_priority_then_path'
        group_id = f'group_{index:04d}'
        for r in group:
            if r['rel_path'] != retained:
                excluded[r['rel_path']] = {**r, 'group_id': group_id, 'reason': reason, 'retained_rel_path': retained or ''}
        group_records.append({'group_id': group_id, 'members': group, 'retained_rel_path': retained, 'reason': reason})
    kept = [r for r in rows if r['rel_path'] not in excluded]
    return kept, list(excluded.values()), group_records


def coverage(rows):
    roles = {s: Counter(int(r['label']) for r in rows if r['split'] == s) for s in PRIORITY}
    gallery = roles['test_retrieval']
    unsupported = [r['rel_path'] for r in rows if r['split'] == 'test_query' and not gallery[int(r['label'])]]
    all_labels = {int(r['label']) for r in rows}
    return {'counts': dict(Counter(r['split'] for r in rows)), 'classes': len(all_labels),
            'classes_by_role': {s: len(c) for s, c in roles.items()},
            'missing_classes_by_role': {s: sorted(all_labels - set(c)) for s, c in roles.items()},
            'query_without_relevant_gallery': unsupported,
            'minimum_relevant_gallery_per_query': min((gallery[int(r['label'])] for r in rows if r['split'] == 'test_query'), default=0),
            'class_counts_by_role': {s: dict(sorted(c.items())) for s, c in roles.items()}}


def scan(ds, rows, source_sha, root, data):
    cache = root / 'artifacts' / 'image_digest_cache' / ds
    cache.mkdir(parents=True, exist_ok=True)
    context = {'source_sha256': source_sha, 'pillow': PIL_VERSION, 'version': VERSION,
               'pixel_definition': 'PIL convert RGB; original dimensions; no resize; no EXIF transpose'}
    context_path = cache / 'context.json'
    if context_path.exists():
        assert json.loads(context_path.read_text()) == context, 'Digest cache context changed'
    else:
        save(context_path, context)
    records = []
    for start in range(0, len(rows), 500):
        subset = rows[start:start + 500]
        chunk_path = cache / f'{start:06d}.json'
        resolved = []
        for row in subset:
            candidate = Path(row['abs_path'])
            path = candidate if candidate.is_file() else data / 'raw' / ds / row['rel_path']
            assert path.is_file(), path
            stat = path.stat()
            resolved.append((row, path, stat.st_size, stat.st_mtime_ns))
        if chunk_path.exists():
            chunk = json.loads(chunk_path.read_text())
            expected = [(r['rel_path'], str(p), size, mtime) for r, p, size, mtime in resolved]
            actual = [(r['rel_path'], r['resolved_path'], r['size'], r['mtime_ns']) for r in chunk]
            assert actual == expected, 'Cached images changed; do not silently reuse digest cache'
        else:
            chunk = []
            for row, path, size, mtime in resolved:
                with Image.open(path) as im:
                    im = im.convert('RGB')
                    pixel_sha = hashlib.sha256(f'RGB|{im.width}|{im.height}|'.encode() + im.tobytes()).hexdigest()
                    dims = [im.width, im.height]
                chunk.append({'rel_path': row['rel_path'], 'resolved_path': str(path), 'size': size,
                              'mtime_ns': mtime, 'byte_sha256': sha(path), 'rgb_sha256': pixel_sha, 'dimensions': dims})
            save(chunk_path, chunk)
        records.extend(chunk)
        print(f'DIGEST {ds} {len(records)}/{len(rows)}', flush=True)
    return records


def run(root=ROOT, data=DATA):
    b1 = json.loads((root / 'artifacts' / 'B1_preflight.json').read_text())
    report_path = root / 'artifacts' / 'B1_clean_manifest_report.json'
    report = {'status': 'running', 'version': VERSION, 'generator_sha256': sha(__file__),
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'rule': {'identity': 'exact original-size decoded RGB pixels', 'conflicting_labels': 'remove all members',
                       'same_label': 'keep one; train > val > test_retrieval > test_query, then lexicographic rel_path',
                       'retained_roles': 'unchanged', 'selection': 'independent of model results'},
              'limitations': ['Does not detect resized, re-encoded nonidentical, near-duplicate or same-scene images.',
                             'Historical checkpoints were trained/selected using original data; evaluating them on this manifest does not undo that exposure.',
                             'Class counts change. Do not compare new scores against original archive as if the dataset were unchanged.'],
              'datasets': {}, 'errors': []}
    save(report_path, report)
    for ds in DATASETS:
        source = data / 'splits' / ds / 'seed_42' / 'split_manifest.csv'
        source_sha = sha(source)
        assert source_sha == b1['datasets'][ds]['sha256'], f'{ds} original manifest changed'
        with source.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            rows = list(reader)
        assert coverage(rows)['counts'] == b1['datasets'][ds]['counts']
        records = scan(ds, rows, source_sha, root, data)
        rgb, byte = defaultdict(list), defaultdict(list)
        by_path = {r['rel_path']: r for r in rows}
        for record in records:
            rgb[record['rgb_sha256']].append(by_path[record['rel_path']])
            byte[record['byte_sha256']].append(by_path[record['rel_path']])
        groups = [v for key, v in sorted(rgb.items()) if len(v) > 1]
        kept, exclusions, group_report = clean_rows(rows, groups)
        checks = coverage(kept)
        assert checks['classes'] == b1['datasets'][ds]['classes'], 'A whole class would disappear'
        assert not any(checks['missing_classes_by_role'].values()), 'A role would lose a class; requires review'
        assert not checks['query_without_relevant_gallery'], 'Query without relevant gallery'
        assert len(kept) + len(exclusions) == len(rows)
        kept_ids = {r['rel_path'] for r in kept}
        assert all(sum(r['rel_path'] in kept_ids for r in group) <= 1 for group in groups)
        target = root / 'manifests' / ds / 'seed_42' / 'split_manifest.csv'
        # Once a completed manifest is used, a changed generator must not replace it silently.
        if target.exists():
            with target.open(newline='', encoding='utf-8') as f:
                assert list(csv.DictReader(f)) == kept, 'Refuse to change frozen manifest'
        else:
            write_csv(target, columns, kept)
        exclusions_path = target.parent / 'excluded_rows.csv'
        write_csv(exclusions_path, columns + ['group_id', 'reason', 'retained_rel_path'], exclusions)
        save(target.parent / 'duplicate_groups.json', group_report)
        entry = {'source_manifest': str(source), 'source_sha256': source_sha, 'manifest': str(target), 'sha256': sha(target),
                 'source_counts': coverage(rows)['counts'], **checks, 'excluded_rows': len(exclusions),
                 'excluded_counts': dict(Counter(r['split'] for r in exclusions)),
                 'duplicate_byte_groups': sum(len(v) > 1 for v in byte.values()), 'duplicate_rgb_groups': len(groups),
                 'conflicting_label_groups': sum(len({r['label'] for r in v}) > 1 for v in groups),
                 'cross_role_rgb_groups': sum(len({r['split'] for r in v}) > 1 for v in groups),
                 'decoded_all_images': len(records), 'all_image_files_preserved': True,
                 'exclusions': str(exclusions_path), 'exclusions_sha256': sha(exclusions_path),
                 'digest_cache': str(root / 'artifacts' / 'image_digest_cache' / ds)}
        report['datasets'][ds] = entry
        save(report_path, report)
        print('CLEAN_MANIFEST', ds, entry['counts'], 'excluded', len(exclusions), flush=True)
    report['status'] = 'clean_manifests_complete'
    report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save(report_path, report)
    print('CLEAN_MANIFESTS_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args()
    run(args.root, args.data)

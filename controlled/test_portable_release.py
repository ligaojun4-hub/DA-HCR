"""Source-only checks for release identity and failure behavior; no datasets/torch."""
import csv
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
import revision_protocol as protocol
import run_controlled as portable


class PortableReleaseTests(unittest.TestCase):
    def test_absolute_and_relative_frozen_import_closure(self):
        frozen = HERE / 'frozen_src'
        for source in frozen.rglob('*.py'):
            module_name = '.'.join(source.relative_to(frozen).with_suffix('').parts)
            package = module_name.rsplit('.', 1)[0]
            for node in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom):
                    name = node.module or ''
                    if node.level:
                        name = importlib.util.resolve_name('.' * node.level + name, package)
                    if not name.startswith('robust_hash'):
                        continue
                    target = frozen.joinpath(*name.split('.'))
                    self.assertTrue(target.with_suffix('.py').is_file() or
                                    (target / '__init__.py').is_file(),
                                    f'{source.relative_to(frozen)} imports missing {name}')

    def test_exact_six_checkpoint_identity_and_epoch_matrix(self):
        registry = portable.verify_package()
        actual = {(x['dataset'], x['method']): x['epoch'] for x in registry['checkpoints']}
        self.assertEqual(actual, {('caltech256', 'base'): 10, ('caltech256', 'dasupcon'): 30,
            ('mit_indoor67', 'base'): 10, ('mit_indoor67', 'dasupcon'): 10,
            ('flowers102', 'base'): 20, ('flowers102', 'dasupcon'): 30})
        self.assertEqual(len(registry['checkpoints']), 6)
        self.assertTrue(all(len(x['sha256']) == 64 for x in registry['checkpoints']))

    def test_suite_has_exact_paper_condition_counts(self):
        counts = {kind: sum(s['kind'] == kind for s in protocol.SPECS)
                  for kind in ('legacy', 'perspective', 'blur', 'brightness')}
        self.assertEqual(counts, {'legacy': 4, 'perspective': 12, 'blur': 3, 'brightness': 6})
        self.assertEqual(len(protocol.SPECS), 25)
        self.assertEqual(len({s['name'] for s in protocol.SPECS}), 25)

    def test_manifests_keep_published_roles_and_screened_ids_subset(self):
        registry = portable.verify_package()
        for ds, record in registry['datasets'].items():
            parsed = {}
            for kind, expected_counts in [('original', record['original_counts']),
                                          ('screened', record['screened_counts'])]:
                path = HERE / record[kind]['path']
                with path.open(encoding='utf-8', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                actual = {role: sum(r['split'] == role for r in rows) for role in expected_counts}
                self.assertEqual(actual, expected_counts, (ds, kind))
                parsed[kind] = {r['rel_path']: r for r in rows}
                self.assertEqual(len(parsed[kind]), len(rows))
            self.assertLessEqual(set(parsed['screened']), set(parsed['original']))
            self.assertTrue(all(parsed['original'][key] == row for key, row in parsed['screened'].items()))

    def test_missing_exact_install_does_not_overwrite_other_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root/'source.csv', root/'target.csv'
            source.write_bytes(b'expected bytes\n')
            checksum = portable.sha(source)
            portable.install_missing_exact(source, target, checksum)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            target.write_bytes(b'other project bytes\n')
            with self.assertRaisesRegex(RuntimeError, 'Refuse overwriting'):
                portable.install_missing_exact(source, target, checksum)
            self.assertEqual(target.read_bytes(), b'other project bytes\n')


if __name__ == '__main__':
    unittest.main(verbosity=2)

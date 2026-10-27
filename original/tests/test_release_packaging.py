"""CPU checks for packaging guards, archived arguments and path identity.

These do not replace model training or a full dataset evaluation.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleasePackagingTests(unittest.TestCase):
    def test_all_saved_configs_reach_real_parser_options(self):
        replay=load_file("paper_replay",ROOT/"replay_config.py")
        entries=replay.archived_entries()
        self.assertEqual(len(entries),71)
        for key in entries:
            entry,cfg,script,args=replay.prepare_command(key)
            self.assertEqual(cfg["num_workers"],20)
            self.assertEqual(cfg["eval_num_workers"],20)
            self.assertIn("--num-workers",args)
            self.assertNotIn("equal_supcon",key)
        relation=replay.prepare_command("positive_relation_same_image_only")[1]
        self.assertEqual(relation["temperature"],.1)
        self.assertEqual(relation["eval_epochs"],"10,14,18,22,26,30")
        equal=replay.prepare_command("equal_positive_dsh")[1]
        self.assertEqual(equal["same_image_weight"],1)
        self.assertEqual(equal["same_class_weight"],1)

    def test_output_guard_preserves_existing_run(self):
        replay=load_file("paper_replay2",ROOT/"replay_config.py")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"run"
            replay.assert_fresh_output(path)
            path.mkdir()
            replay.assert_fresh_output(path)
            (path/"last.pt").write_bytes(b"keep original checkpoint")
            with self.assertRaises(ValueError):
                replay.assert_fresh_output(path)
            self.assertEqual((path/"last.pt").read_bytes(),b"keep original checkpoint")

    def test_relocated_file_keeps_original_noise_identity(self):
        paths=load_file("paper_paths",ROOT/"src/robust_hash/utils/paths.py")
        identity="/root/autodl-tmp/robust_hash_project_scaffold/robust_hash_project_scaffold/data/raw/flowers102/class_001/example.png"
        tree=ast.parse((ROOT/"src/robust_hash/data/transforms.py").read_text(encoding="utf-8"))
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("_seed_from_key","_rng_for")]
        space={"hashlib":hashlib,"np":np,"Path":Path,"ACTIVE_PROTOCOL":"formal_v12_monotonic"}
        exec(compile(ast.Module(body=functions,type_ignores=[]),"frozen_rng_functions","exec"),space)
        expected=space["_rng_for"](identity,"medium").normal(0,10,(3,4,3))
        pixels=np.arange(36,dtype=np.uint8).reshape(3,4,3)
        with tempfile.TemporaryDirectory() as first,tempfile.TemporaryDirectory() as second:
            resolved=[]
            for base in (first,second):
                dest=Path(base)/"raw/flowers102/class_001/example.png"
                dest.parent.mkdir(parents=True)
                Image.fromarray(pixels).save(dest)
                with patch.dict(os.environ,{"DAHCR_DATA_ROOT":base}):
                    actual=paths.resolve_source_image(identity)
                    np.testing.assert_array_equal(np.asarray(Image.open(actual)),pixels)
                    np.testing.assert_array_equal(space["_rng_for"](identity,"medium").normal(0,10,(3,4,3)),expected)
                    resolved.append(actual)
            self.assertNotEqual(resolved[0],resolved[1])

    def test_missing_relocation_fails_without_rewriting_identity(self):
        paths=load_file("paper_paths2",ROOT/"src/robust_hash/utils/paths.py")
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"DAHCR_DATA_ROOT":tmp}):
            with self.assertRaises(FileNotFoundError):
                paths.resolve_source_image("/missing/data/raw/caltech256/class/not_here.png")
            with self.assertRaises(ValueError):
                paths.resolve_source_image("/missing/data/raw/../not_allowed.png")


if __name__=="__main__":
    unittest.main()

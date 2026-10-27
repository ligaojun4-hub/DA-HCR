"""Small CPU-only tests for replay path handling; does not train a model."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("aux_replay",ROOT/"replay.py")
replay=importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class AuxiliaryPackagingTests(unittest.TestCase):
    def test_only_split_root_changes(self):
        original=json.loads((ROOT/"caltech256_seed42_split.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            relocated=replay.relocated_split(tmp)
        self.assertNotEqual(original["data_root"],relocated["data_root"])
        relocated["data_root"]=original["data_root"]
        self.assertEqual(relocated,original)

    def test_command_retains_auxiliary_workers_and_settings(self):
        cmd=replay.planned_command("/data/caltech256","/outputs/fresh")
        self.assertEqual(cmd[cmd.index("--num-workers")+1],"8")
        self.assertEqual(cmd[cmd.index("--batch-size")+1],"128")
        self.assertEqual(cmd[cmd.index("--quant-weight")+1],"0.001")
        self.assertEqual(cmd[cmd.index("--train-degrade")+1],"medium")
        self.assertIn("--exclude-clutter",cmd)

    def test_existing_outputs_and_missing_images_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            replay.fresh_output(root)
            (root/"old.pt").write_bytes(b"old")
            with self.assertRaises(ValueError):
                replay.fresh_output(root)
            with self.assertRaises(FileNotFoundError):
                replay.validate_images({"data_root":tmp,"items":{"train":[["missing.jpg",0]]}})
            self.assertEqual((root/"old.pt").read_bytes(),b"old")


if __name__=="__main__":
    unittest.main()

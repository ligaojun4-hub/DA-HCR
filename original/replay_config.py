"""Print or explicitly run an archived paper configuration without retuning it.

Default is a dry print. --run invokes the byte-preserved selected trainer, with
only its output directory redirected to a fresh user-selected destination.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parent


def archived_entries():
    entries = json.loads((ROOT / "experiment_index.json").read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in entries}


def prepare_command(identifier):
    entries = archived_entries()
    if identifier not in entries:
        raise ValueError(f"Unknown experiment {identifier}; use --list")
    entry = entries[identifier]
    path = ROOT / entry["config"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["config_sha256"]:
        raise ValueError("Archived configuration hash differs from the release index")
    config = json.loads(path.read_text(encoding="utf-8"))
    script = ROOT / entry["script"]
    tree = ast.parse(script.read_text(encoding="utf-8"))
    options = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                    options.add(arg.value)
    args = []
    for key, value in config.items():
        option = "--" + key.replace("_", "-")
        if option not in options:
            raise ValueError(f"Saved configuration contains unsupported parameter {key}; no value was silently omitted")
        if isinstance(value, bool):
            if value:
                args.append(option)
        elif isinstance(value, (str,int,float)):
            args.extend([option, str(value)])
        else:
            raise ValueError(f"Unsupported saved value for {key}: {value!r}")
    return entry, config, script, args


def assert_fresh_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output directory is not empty; choose a fresh directory instead of overwriting an old run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list",action="store_true")
    parser.add_argument("--experiment")
    parser.add_argument("--data-root",type=Path,help="Directory containing raw/<dataset>/<image>; original split identities are retained")
    parser.add_argument("--output-dir",type=Path,help="Fresh destination for this one training run")
    parser.add_argument("--centers",type=Path,help="Explicit original SHC or SAAT center file; no replacement centers are invented")
    parser.add_argument("--saat-deterministic-centers",action="store_true",help="Explicitly choose the archived deterministic sine-based SAAT fallback; historical env identity is not certified")
    parser.add_argument("--run",action="store_true",help="Actually train using the saved configuration; default only prints")
    args = parser.parse_args()
    if args.list:
        for entry in archived_entries().values():
            print(entry["id"] + "\t" + entry["group"])
        return
    if not args.experiment:
        parser.error("--experiment is required unless --list is used")
    entry, config, script, saved_args = prepare_command(args.experiment)
    print(shlex.join([sys.executable,str(script),*saved_args]))
    if not args.run:
        return
    if not args.data_root or not args.output_dir:
        parser.error("--run requires --data-root and --output-dir")
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The archived execution platform was Linux; replay on Linux with the actual PyTorch/torchvision runtime")
    assert_fresh_output(args.output_dir)
    if not (args.data_root / "raw" / config["dataset"]).is_dir():
        raise FileNotFoundError("Data root must contain raw/" + config["dataset"])
    if config["baseline"] == "shc" and not args.centers:
        parser.error("SHC replay requires its original train-derived --centers file; see the archived generator scripts")
    if config["baseline"] == "saat" and not args.centers and not args.saat_deterministic_centers:
        parser.error("SAAT center identity is not saved in config.json; provide --centers or explicitly select --saat-deterministic-centers")
    if args.centers and not args.centers.is_file():
        raise FileNotFoundError(args.centers)
    os.environ["DAHCR_DATA_ROOT"] = str(args.data_root.resolve())
    os.environ["DAHCR_SPLIT_ROOT"] = str(ROOT / "data/splits")
    # Do not inherit a stale center file from some unrelated experimental session.
    os.environ.pop("DAHCR_SHC_CENTER_PATH",None)
    os.environ.pop("SAAT_CENTER_PATH",None)
    if args.centers:
        var = "DAHCR_SHC_CENTER_PATH" if config["baseline"] == "shc" else "SAAT_CENTER_PATH"
        os.environ[var] = str(args.centers.resolve())
    os.environ.pop("DAHCR_FORCE_RUN_DIR",None)
    sys.path.insert(0,str(ROOT / "src"))
    # Loading as a non-main module avoids running before output redirection.
    # Numeric functions, saved arguments, optimizer and model code are untouched.
    namespace = {"__name__":"paper_replay_trainer", "__file__":str(script)}
    exec(compile(script.read_text(encoding="utf-8"),str(script),"exec"),namespace)
    output = args.output_dir.resolve()
    namespace["build_run_dir"] = lambda *unused, **unused_kw: output
    sys.argv = [str(script),*saved_args]
    namespace["main"]()


if __name__ == "__main__":
    main()

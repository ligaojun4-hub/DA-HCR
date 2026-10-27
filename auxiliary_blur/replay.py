"""Print or explicitly replay the separate archived blur-augmentation run.

Default is a dry print and creates no files. --run makes a path-only copy of
the exact original split JSON inside a fresh output directory, verifies every
listed image exists, and invokes the byte-preserved standalone trainer using
all archived arguments. No losses, split memberships or numeric settings change.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def fresh_output(path):
    path=Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output is not empty; choose a fresh directory")


def relocated_split(data_root):
    split=json.loads((ROOT/"caltech256_seed42_split.json").read_text(encoding="utf-8"))
    split["data_root"]=str(Path(data_root).resolve())
    return split


def planned_command(data_root,output_dir):
    args=json.loads((ROOT/"run_record.json").read_text(encoding="utf-8"))["args"]
    output=Path(output_dir).resolve()
    args["data_root"]=str(Path(data_root).resolve())
    args["out_dir"]=str(output)
    args["split_json"]=str(output/"split_replay.json")
    command=[sys.executable,str(ROOT/"run_caltech256_hashnet_degaug.py")]
    for key,value in args.items():
        if key=="exclude_clutter":
            command.append("--exclude-clutter" if value else "--include-clutter")
        elif key in ("pretrained","amp"):
            command.extend(["--"+key,str(int(value))])
        else:
            command.extend(["--"+key.replace("_","-"),str(value)])
    return command


def validate_images(split):
    root=Path(split["data_root"])
    missing=[]
    for rows in split["items"].values():
        for relative,label in rows:
            path=Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unexpected non-relative image identifier in archived split")
            if not (root/path).is_file():
                missing.append(relative)
    if missing:
        raise FileNotFoundError(f"{len(missing)} archived images unavailable; examples: {missing[:5]}. No rows were dropped or replaced.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True,help="Actual Caltech class-folder root, containing 001.ak47 etc.")
    parser.add_argument("--output-dir",type=Path,required=True,help="Fresh location; old runs are never overwritten")
    parser.add_argument("--run",action="store_true",help="Actually train; omitted means print only")
    args=parser.parse_args()
    fresh_output(args.output_dir)
    command=planned_command(args.data_root,args.output_dir)
    print(shlex.join(command))
    if not args.run:
        return
    split=relocated_split(args.data_root)
    validate_images(split)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"split_replay.json").write_text(json.dumps(split,indent=2)+"\n",encoding="utf-8")
    subprocess.run(command,check=True)


if __name__=="__main__":
    main()

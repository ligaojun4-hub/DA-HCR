"""Run the archived SHC train-only center generator with relocated image files.

Caltech uses its original projection seed 2025; the separate Indoor/Flowers
generator retains seed 2025 + class_count + bits. No centers are generated until
this command is invoked explicitly. Existing output files are never overwritten.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=("caltech256","mit_indoor67","flowers102"),required=True)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    if out.exists() or out.with_suffix(".json").exists():
        raise ValueError("Center output already exists; choose a fresh output filename")
    if not (args.data_root/"raw"/args.dataset).is_dir():
        raise FileNotFoundError("Data root must contain raw/"+args.dataset)
    os.environ["DAHCR_DATA_ROOT"]=str(args.data_root.resolve())
    os.environ["DAHCR_SPLIT_ROOT"]=str(ROOT/"data/splits")
    sys.path.insert(0,str(ROOT/"src"))
    from robust_hash.utils.paths import resolve_source_image
    if args.dataset=="caltech256":
        source=ROOT/"scripts/prepare_shc_trainonly_centers.py"
        argv=[str(source),str(out)]
    else:
        source=ROOT/"scripts/prepare_shc_trainonly_centers_general.py"
        argv=[str(source),"--dataset",args.dataset,"--split-seed","42","--bits","64","--out",str(out)]
    namespace={"__name__":"paper_center_generator","__file__":str(source)}
    exec(compile(source.read_text(encoding="utf-8"),str(source),"exec"),namespace)
    # Both archived generators read their fixed original manifest from cwd.
    # Only the actual file-open location is redirected; rows/labels/order stay fixed.
    namespace["resolve_image_path"]=lambda row,unused_root: resolve_source_image(row["abs_path"])
    os.chdir(ROOT)
    sys.argv=argv
    namespace["main"]()


if __name__=="__main__":
    main()

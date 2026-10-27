# Separate historical blur-augmentation experiment

This is the standalone Caltech256–HashNet experiment reported in AppendixB/Table16 on p.38 of the supplied48-page manuscript. It is separate from the main noise-protocol Base/DA-HCR comparisons and from the new fixed-checkpoint blur test. It has no DA-HCR consistency term.

The actual standalone source, exact split JSON and saved arguments/results were recovered from the author's AutoDL project. Earlier collection overlooked the relative **`scripts/outputs/`** location; the materials were not missing from AutoDL. No retraining or benchmark recomputation was needed to recover these files. A separate CPU smoke check verified the original checkpoint's strict loading and one-image forward pass; this is not a new mAP experiment.

## Contents and evidence

- `run_caltech256_hashnet_degaug.py`: byte-preserved actual source.
- `caltech256_seed42_split.json`: byte-preserved separate historical partition.
- `run_record.json`: byte-preserved metadata containing actual arguments, selected epoch and result values.
- `SOURCE_PROVENANCE.json`: source/split/record SHA-256 values and original checkpoint identity. The44.9MB checkpoint is not committed.
- `verification.json`: checks Table16 rounding, saved test-record agreement, epoch25 validation selection and split counts against the recovered history/curves. History and result caches are not otherwise copied into the code repository.
- `replay.py`: optional packaging helper that prints by default and makes a path-only split copy only when `--run` is explicitly supplied.

Actual saved settings:64bits,seed42,30epochs, medium training pairs only, train/eval batch128/256, **8workers**, backbone/head learning rates5e-5/5e-4, weight decay1e-4, quantization .001, validation every5epochs, ImageNet-pretrained ResNet18 and AMP enabled. The source retains its original AdamW optimizer and cosine scheduler. The main experiment's20workers are unchanged; this separate run must retain its actual8workers.

Blur-chain settings `(resize scale, JPEG quality, blur sigma)` are weak `(.75,70,.5)`, medium `(.50,50,1.0)` and strong `(.25,30,1.5)`. Stored test results are:

| Clean | Weak | Medium | Strong | Mean of three degraded |
|---:|---:|---:|---:|---:|
| .4486822324 | .4457687264 | .4324461225 | .3819039339 | .4200395943 |

All five round to Table16. The highest recorded validation degraded-average mAP occurs at epoch25. These internal archive checks do not certify a fresh numerical rerun.

## Data and safe use

Use Linux and the repository's PyTorch/torchvision/Pillow/NumPy dependencies. For a dependency-free help view or a dry command print:

```bash
python auxiliary_blur/replay.py --help
python auxiliary_blur/replay.py \
  --data-root /path/to/caltech256 \
  --output-dir /path/to/new_aux_blur_run
```

The data root directly contains class folders such as `001.ak47/`. The exact historical split contains256classes and train17910/validation2977/query2977/gallery5982 entries. It retains66original `.ipynb_checkpoints` image filenames:38train,4validation,7query and17gallery. Those recorded entries are preserved rather than silently removed. Distinct identifiers do not establish distinct image content. This separate historical partition has not undergone the new decoded-RGB screening used for the controlled fixed-checkpoint evaluation. The raw images are not in the repository; matching original image files must be supplied to reproduce this exact historical partition.

The original script loads `data_root` from an existing split JSON; passing its `--data-root` alone does **not** override that loaded value. The helper creates a separate `split_replay.json` changing only this root location, while preserving every class, role, identifier and label in the bundled original JSON. It checks that every listed image exists and stops if any are missing; it does not drop unavailable rows, generate replacement partitions or overwrite an existing run.

To explicitly replay training after supplying the exact images:

```bash
python auxiliary_blur/replay.py \
  --data-root /path/to/caltech256 \
  --output-dir /path/to/new_aux_blur_run \
  --run
```

The actual standalone invocation represented by the wrapper is:

```bash
python auxiliary_blur/run_caltech256_hashnet_degaug.py \
  --data-root /path/to/caltech256 \
  --out-dir /path/to/new_aux_blur_run \
  --split-json /path/to/new_aux_blur_run/split_replay.json \
  --seed 42 --bits 64 --epochs 30 \
  --batch-size 128 --eval-batch-size 256 --num-workers 8 \
  --input-size 224 --lr-backbone 5e-5 --lr-head 5e-4 \
  --weight-decay 1e-4 --beta 1.0 --quant-weight 0.001 \
  --train-degrade medium --eval-interval 5 \
  --train-frac 0.6 --val-frac 0.1 --query-frac 0.1 \
  --exclude-clutter --pretrained 1 --amp 1
```

The referenced replay JSON must already exist and match the original partition. The wrapper is the safer entrypoint because the standalone script otherwise can generate a new split when a specified split file does not exist. The original trainer and its numeric behavior remain unchanged.

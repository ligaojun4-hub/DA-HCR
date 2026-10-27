# Original manuscript experiments

This directory contains selected original training and evaluation code recovered from the author's AutoDL project, plus archived configurations and split manifests. It supports the 48-page manuscript supplied on 2026-10-26, SHA-256 `3fcba1ce8e226bb213825af6a076177481f2900d2aa78e89d34fca20df49d40f`. It excludes the later matched three-arm Caltech/Flowers retraining extension. The controlled perspective/brightness/blur evaluation is a separate runtime elsewhere in this repository.

No training or full-dataset inference was run when assembling this source release. The presence of source and configurations does not certify exact regeneration of archived numerical results. The manuscript itself reports unresolved differences between archived and recalculated historical mAP.

## What is included

`original/experiment_index.csv` and `.json` map **71 original configurations** to their training entrypoint, source path, SHA-256 and manuscript pages:

| Group | Configurations | Manuscript |
|---|---:|---|
| Five established objectives, Base and DA-HCR, three datasets | 30 | pp.15-17,21-22 |
| Adapted SHC-inspired, NHL-style and SAAT-inspired, Base and DA-HCR | 18 | pp.16-17,21 |
| Original DSH/DPSH/DHN equal-positive comparisons | 3 | p.23 |
| HashNet original same-instance-only / same-class-only explorations | 2 | p.24 |
| Sequential alpha / temperature screening | 6 | p.25 |
| HashNet 32 / 128 bit comparisons, using the main64-bit runs separately | 4 | pp.25-26 |
| Original additional split/seed43 / 44 HashNet runs | 8 | p.37 |

`dasupcon` is the archived software name for DA-HCR. Original equal-positive comparisons already printed in the manuscript remain included. They are distinct from the excluded later matched HashNet three-arm repetitions.

Caltech DSH main configurations are taken from the archived replacement run. The SHC Caltech Base uses its fixed train-only-center baseline and DA-HCR uses the selected T06 configuration; NHL Caltech uses Base plus selected T04. Remaining adapted settings are taken from the fixed transferred runs. Source identity is recorded in `original/SOURCE_PROVENANCE.json`; source paths refer to the original remote collection and do not contain account credentials.

## Setup and original data identities

Use Linux with Python, PyTorch and matching torchvision supporting ImageNet-pretrained ResNet-18, plus NumPy, Pillow, pandas, PyYAML, matplotlib and tqdm as listed at repository level. The original generic requirements file is not an exact historical environment lock. This release refuses the old silent small-CNN fallback: an unavailable ResNet-18 is an error.

Obtain the three public datasets separately and preserve the class-folder layout represented by the supplied split CSVs. Choose a data root containing:

```text
/path/to/data/raw/caltech256/<class>/<image>
/path/to/data/raw/mit_indoor67/<class>/<image>
/path/to/data/raw/flowers102/<class>/<image>
```

The seven selected original CSVs are under `original/data/splits`: Caltech42/43/44, Indoor42/43/44, Flowers42. These are custom experiment partitions, not the datasets' official classification splits. Do not replace them with newly generated partitions. The extra split/seed experiments change image pools and are not simply initialization repeats.

The original composite Gaussian noise is keyed by the **original absolute image identity**. The path shim maps only the file-open location to the new data root; it retains the original CSV identity in `_rng_for`. Rewriting `abs_path` in the CSV changes generated noise and is not an exact replay. Raw data and checkpoints are deliberately not committed.

## Print or replay a saved configuration

From the repository root:

```bash
python original/replay_config.py --list
python original/replay_config.py --experiment main_caltech256_hashnet_dasupcon
```

Both commands only print; they do not train. To explicitly train one archived run into a fresh directory:

```bash
python original/replay_config.py \
  --experiment main_caltech256_hashnet_dasupcon \
  --data-root /path/to/data \
  --output-dir /path/to/new_runs/caltech_hashnet_dahcr \
  --run
```

The launcher passes every saved parameter explicitly, verifies the archived config hash, and refuses a nonempty output directory. It does not tune parameters, reduce workers or batch size, select a different backbone, or resume/overwrite another experiment. Training and validation workers are **20** in all71 included archived configs. The original script parser defaults differ; running a bare trainer without saved arguments is not the paper's configuration. For original component exploration retain its temperature0.1 and validation10/14/18/22/26/30; do not normalize these to the main temperature0.2/validation10/20/30.

## Adapted semantic centers

SHC requires `DAHCR_SHC_CENTER_PATH`. The actual archived train-only generators are included. Caltech uses projection seed2025, while the separate Indoor/Flowers generator uses2025+class_count+bits. Do not interchange them or substitute a generic Hadamard/random CSQ-center script.

To generate centers from exactly the original training rows, with the archived numerical procedure:

```bash
python original/prepare_centers.py \
  --dataset caltech256 --data-root /path/to/data \
  --out /path/to/new_centers/caltech256_trainonly.pt
python original/replay_config.py \
  --experiment adapted_caltech256_shc_dasupcon \
  --data-root /path/to/data \
  --centers /path/to/new_centers/caltech256_trainonly.pt \
  --output-dir /path/to/new_runs/caltech_shc_dahcr --run
```

The center generator's source is retained byte for byte; the wrapper only maps its image-open paths and working directory. The original generator can skip unreadable image rows. Supply complete valid datasets and inspect its emitted per-class counts; a skipped-image regeneration is not certified as identical to the archived center file. Recovering the original center files and their hashes is preferable when exact identities are required.

SAAT-inspired code can read `SAAT_CENTER_PATH`; without it the archived implementation constructs deterministic sine-based class codes. `config.json` does not record this environment variable, so the actual historical center-file state cannot be certified from these saved configs alone. The replay launcher therefore requires either `--centers FILE` or the explicit `--saat-deterministic-centers` choice. This does not change the SAAT loss or manufacture a historical environment record.

## Historical retrieval and code stability

For a supplied compatible original checkpoint, the original full-gallery mAP evaluator retains the historical `torch.argsort` tie behavior:

```bash
cd original
export DAHCR_DATA_ROOT=/path/to/data
export PYTHONPATH="$PWD/src"
python scripts/07_eval_checkpoint.py \
  --checkpoint /path/to/best_val.pt --dataset caltech256 --split-seed 42 \
  --image-size 224 --eval-batch-size 512 --eval-num-workers 20 --save-json
```

`scripts/08_eval_pk_recall.py` contains the archived retrieval-depth precision/recall calculation; invoke its `--help` from `original/` with `PYTHONPATH=src`. `analysis/v12_hashnet_bit_stability_forward_single_v2.py` preserves the original Caltech HashNet code-stability analysis and exposes its own command-line options. It strictly loads the supplied model. Its outputs are separate from the new controlled-evaluation stable ranking metric; do not silently recompute an archived table with a different tie policy.

## Packaging-only changes

- `utils/paths.py`: environment-selectable data/split roots and original-identity-preserving image-open remapping.
- `data/transforms.py`: only the file opened by Pillow is remapped; transform operations, original identity and noise seed stay unchanged.
- `models/backbone.py`: fail when the required torchvision ResNet-18 cannot be constructed, instead of silently training a different CNN.
- `replay_config.py`: exact saved-argument forwarding, explicit center mode and fresh-output redirection. The numeric trainer functions, losses, optimizer settings and original metric remain unchanged.
- `prepare_centers.py`: invokes the actual dataset-specific archived generator with relocated images and a fresh output filename.

## Known source gaps and verification limits

- The separate auxiliary blur-training experiment in Appendix B (pp.37-38) is provided under `auxiliary_blur/`, with its original script, independent JSON partition and archived run record. It uses batch128/eval256/medium-only settings and eight workers according to its own recorded arguments. These are historical settings for a separate experiment; the 71 configurations in `original/` remain unchanged with 20 workers. Do not use it as a matched replacement for the main protocol.
- The exact historical SAAT center environment/file identity and original center-file hashes are not proven by `config.json` alone.
- Figure-specific plotting scripts/source bindings were not all recovered as independently runnable generators. The quantitative retrieval and code-stability routines are included; manuscript reconstruction is not part of this repository.
- No raw dataset, pretrained weight, training checkpoint, large result cache, later extension, cloud management helper or private account information is included.
- Local CPU packaging tests cover all71config-to-parser mappings, workers20, non-overwrite behavior, relocated image pixels and preserved noise identity. Full scientific imports/model forwards/full-data scoring need the real PyTorch environment; no claim of a fresh successful training or exact numerical reproduction is made.

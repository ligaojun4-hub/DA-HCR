# Source-release validation

Validation date: 2026-10-26. These checks concern the distributed code and its packaging. They do not establish a new experimental result or certify every manuscript claim.

## Checks completed

- Seventeen known-answer geometry and retrieval-metric tests from the controlled evaluator passed.
- Five controlled-package tests passed, including package integrity, checkpoint identities, manifest role counts, non-overwrite behavior, and absolute/relative internal import closure.
- Four original-package tests passed. All 71 saved configurations map to the selected parsers with their explicit saved arguments and 20 training/evaluation workers.
- The Git index was compared byte-for-byte with the source package; no line-ending normalization was allowed. Historical numerical source hashes and the packaging modifications are recorded separately in each runtime's provenance file.
- In the actual AutoDL scientific Python environment, all nine safe original CLI/import checks passed: configuration listing, the center-wrapper help, five selected trainer help commands, and two evaluation help commands. No training command was run by these checks.
- Each of the six original historical HashNet checkpoints passed SHA-256 verification, strict ResNet-18 state-dict loading, and a CPU forward pass on one real training image in clean and medium-degraded form. Outputs had shape `(2, 64)`, finite activations, and strictly binary hash codes.
- The recovered auxiliary blur experiment passed three additional packaging tests and independent dry-run/non-overwrite checks. Its script, split and run record match the recovered source hashes; all five Table16 values and the validation-selected epoch25 match the original records. On AutoDL, all 29,846 recorded image paths exist; the recorded checkpoint passed its SHA-256 check, strict state-dict loading and a real-image clean/weak/medium/strong CPU forward pass of shape `(4, 64)`. This is an identity and smoke check, not a full mAP recomputation or a decoded-RGB duplicate audit of that separate partition.

## Environment used for the AutoDL checks

| Package | Observed version |
|---|---|
| Python | 3.12.3 |
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| NumPy | 2.3.2 |
| Pillow | 11.3.0 |
| pandas | 3.0.2 |

The checks used the CPU; a CUDA-capable PyTorch build being installed does not mean a GPU benchmark was run. These are the versions observed during release checks, not a recovered lock of the original historical training environment.

## Checks not performed

No full training, 150-condition controlled benchmark, full-dataset rescoring, or new model selection was performed for this source release. SAAT's historical center environment and all figure-specific generation scripts remain limited as documented in `original_training.md`. The separate auxiliary blur experiment's source, JSON partition and archived record were recovered from the original `scripts/outputs` layout; its stored test scores match Table 16 when rounded to four decimals, and its recorded validation curve selects epoch25. Recovery and record comparison do not constitute a new end-to-end numerical rerun. Cross-platform and path-sensitive degradation behavior must not be assumed identical without verification.

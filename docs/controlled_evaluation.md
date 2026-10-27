# Fixed-checkpoint controlled evaluation

This directory reproduces the **evaluation procedure** in manuscript Sec.4.7, Sec.6 and AppendicesC-E (supplied48-pagePDF SHA-256 `3fcba1ce8e226bb213825af6a076177481f2900d2aa78e89d34fca20df49d40f`). It contains no later matched-three-arm retraining or additional loss experiments. There is no training command here.

## What is provided

- The actual AutoDL `eval_remote.py`, `revision_protocol.py`, `preflight_remote.py`, decoded-RGB manifest builder, and geometry/metric tests, copied byte-for-byte from the executed historical evaluation workspace.
- Only the reached model/data/legacy retrieval modules under `controlled/frozen_src/`. Unrelated training objectives and experiment runners are excluded.
- Three original and three RGB-screened seed42 split CSVs. They identify dataset-relative image paths, class labels and roles; they contain no image pixels. Archived `abs_path` fields are retained because the original noise seed and resolver use path strings.
- A six-checkpoint identity registry, including SHA-256, selected epoch and required configuration fields. It contains no model weights or numerical retrieval-result files.
- `run_controlled.py`, a new release wrapper that remaps filesystem paths and regenerates technical audits. The numerical evaluator, transformations, model and metrics are unchanged. `SOURCE_PROVENANCE.json` distinguishes those byte-identical files from packaging. `PACKAGE_SHA256.json` fingerprints the complete controlled package.

The old preflight fingerprint covered the entire old project. It is **not** falsely reused as a new pass for this smaller release. Preparation regenerates the preflight against this release's selected dependency closure and the local images/checkpoints; its source-report hashes remain recorded separately in the registry.

## External inputs

Install the original datasets from their official sources and obtain the six historical weights from the authors. Weights/raw images are deliberately outside this source-code repository. A new training checkpoint is not a substitute for these validation-selected files.

The wrapper accepts an external project directory with this layout:

```text
PROJECT/
  data/raw/caltech256/<dataset-relative files>
  data/raw/mit_indoor67/<dataset-relative files>
  data/raw/flowers102/<dataset-relative files>
  data/splits/<dataset>/seed_42/split_manifest.csv
  outputs/formal_v12/
    caltech256_5baseline_exact30/hashnet_{base,dasupcon}/run/best_val.pt
    mit_indoor67_5baseline_exact30_with_dsh/hashnet_{base,dasupcon}/run/best_val.pt
    flowers102_5baseline_exact30_with_dsh/hashnet_{base,dasupcon}/run/best_val.pt
```

Preparation installs a bundled original CSV only if it is absent; it refuses to overwrite a different existing split. Raw images and existing checkpoints are never modified. Generated audits, previews, binary-code caches and scores are written to a separate `--run-root` and should not be committed.

Selected epochs (Base/DA-HCR): Caltech25610/30; MIT-Indoor6710/10; Flowers10220/30. Every checkpoint must match its registry SHA-256 before loading. The historical strict state-dict load and ResNet18/no-fallback guards remain enabled.

## Environment

The original archived evaluator imports PyTorch, torchvision, NumPy, Pillow and pandas. The repository's current requirements specify minimum versions rather than a fully pinned environment; choose a mutually compatible PyTorch/torchvision build. Runtime versions are recorded by the regenerated audit. CUDA is required by the unchanged full-evaluation phase; the technical geometry/metric tests require only NumPy and Pillow. Do not assume the historical code's optional small-CNN fallback is valid for a real evaluation: preflight and checkpoint loading explicitly reject it.

## Source-only tests

From the repository root:

```bash
python controlled/code/test_protocol_metrics.py
python controlled/test_portable_release.py
python controlled/run_controlled.py --help
```

These checks cover independent corner mappings, the Pillow pixel-center convention, no-fill boundaries, deterministic identity, brightness clipping, known-answer full-galleryAP, gallery-order invariance under ties, self exclusion, empty/no-relevant accounting, bit-flip identities, exact condition counts, manifest role counts, checkpoint epoch identities and refusal to overwrite other files. They do not replace a real-image/checkpoint run.

## Run the fixed evaluation

Set paths explicitly; the following names are placeholders for local directories:

```bash
PROJECT=/path/to/external/original_project
RUN=/path/to/external/controlled_run

python controlled/run_controlled.py prepare --legacy-root "$PROJECT" --run-root "$RUN"
python controlled/run_controlled.py run --legacy-root "$PROJECT" --run-root "$RUN" --phase validation --manifest-mode clean
python controlled/run_controlled.py run --legacy-root "$PROJECT" --run-root "$RUN" --phase smoke --manifest-mode clean
```

Preparation audits all original image entries, strictly loads six weights, checks source/split hashes and clean-tensor identity, then decodes every original-sizeRGB image and regenerates the exact screened manifests. Those retained rows must match the supplied screened CSVs. This stage can take time because it actually reads the datasets; it does not manufacture a preflight pass from saved success flags.

Validation creates complete25-condition composite previews for predetermined train/validation images; it does not use test mAP to choose condition strengths. Inspect all files in `$RUN/artifacts/B2/validation/clean/controlled_v1/composite_previews/`. After inspecting every preview, explicitly record the inspection, then run inference:

```bash
python controlled/run_controlled.py approve-previews --legacy-root "$PROJECT" --run-root "$RUN" --manifest-mode clean --reviewer "Your name" --confirm-inspected
python controlled/run_controlled.py run --legacy-root "$PROJECT" --run-root "$RUN" --phase full --manifest-mode clean --batch-size 128 --workers 4
```

The preview inspection requirement is retained from the executed evaluator. The wrapper does not automatically claim a person has inspected images. It records the exact validation-report and every preview hash, which the evaluator verifies before inference.

The screened full suite evaluates six models×25conditions=150model-condition cases: clean; originalweak/medium/strong; four perspective directions×three strengths; independentblur0.5/1/2; signedbrightness±15/±30/±45. Test gallery images remain clean. The evaluator caches strictly binary codes and writes per-queryAP, identifiers and bit changes, preserving every condition and failures. Re-running with the same immutable context reuses validated completed cache entries; changing sources/layout requires a new audited run directory.

For the separate original-partition recalculation, repeat validation, smoke, explicit preview inspection and full phases with `--manifest-mode original`. This mode includes only the four original conditions (24model-condition cases), and reports both historical `torch.argsort` scoring and the stable identifier-tie recalculation. It is not merged with the screened suite.

## Protocol and interpretation boundaries

New perspective/blur/brightness transforms operate individually on bilinearly resized224×224RGB images before normalization. The original composite transform acts at native resolution before the network resize. A source trapezoid is mapped to the full output rectangle using the documented half-pixel-adjusted inverse homography; this is a planar crop/magnification test, not a rendered3Dviewpoint.

Controlled scoring uses strict `z>=0 -> +1`, full-galleryAP, identical-identifier self exclusion, and lexicographic unique gallery-ID tie breaking. Clean-to-degradedBFR is computed against each model's own clean codes. Original legacy scoring remains a separate function with its original tie behavior.

**Path-sensitive legacy noise:** the unchanged original transform deterministically seeds noise from the resolved image-path string, severity and protocol. Its resolver first uses an existing archived `abs_path`, then falls back to `PROJECT/data/raw/<dataset>/<rel_path>`. Moving data to another absolute path can change the exact original-composite noise realization even though parameters are unchanged. Retaining the historical absolute layout is required for the same path-based draw; the wrapper does not silently rewrite numerical seed semantics. New geometric/photometric transforms are path independent.

The supplied manuscript already reports that re-evaluating historical HashNet weights differs slightly from archived original mAP (maximum0.001644), with unresolved cause. This source release does not claim to eliminate that discrepancy or promise bit-for-bit cross-platform results. Screened evaluation also cannot undo historical training/validation duplicate exposure. Higher absolute mAP, reduced degradation sensitivity and lowerBFR remain distinct quantities.

## Verification performed for packaging

Copied controlled numerical source hashes match the current AutoDL source archive; the evaluator, protocol, original preflight, clean-manifest builder and original protocol tests also match the previously archived executed snapshot. The17original geometry/metric tests,5portable packaging tests and syntax/CLI checks are run during release preparation. The packaging tests inspect both absolute and relative local imports, including package initializers. A full GPU inference rerun is not implied by these checks. The local packaging environment lacks PyTorch/torchvision and the external datasets/weights, so a real checkpoint-forward smoke test and150-case rerun have not been performed locally for this release.

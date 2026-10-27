# DA-HCR

Experiment code accompanying **Degradation-aware hash consistency regularization for robust deep hashing retrieval**.

The release is scoped to the supplied manuscript. It contains the historical paired-view training implementation, the paper's configurations and split manifests, and the fixed-checkpoint controlled degradation evaluation. The code was collected from the authors' AutoDL workspace. Dataset images, trained network weights, progress notebooks, manuscript files, and unrelated projects are not included.

## Contents

- `original/`: original training, ablation, and retrieval code; archived run configurations; original data partitions.
- `controlled/`: the separately frozen runtime used for the paper's perspective, blur, and brightness evaluation, with checkpoint identities and original/screened partitions.
- `docs/`: source provenance, command details, and limitations of replaying the historical experiments.

The two runtimes are kept separate because their frozen dependency versions differ. Do not replace one runtime's `robust_hash` package with the other.

## Setup

Use a Python environment with a matching PyTorch/torchvision installation. Install the appropriate PyTorch build for your device first, then:

```bash
python -m pip install -r requirements.txt
python original/replay_config.py --help
python controlled/run_controlled.py --help
```

The dependency list describes the required packages; it is not a lock file for the historical training environment. Training requires the ResNet-18 backbone described in the manuscript. A small fallback network is not an equivalent experiment.

## Data

Obtain the original images from their dataset providers:

- [Caltech256](https://doi.org/10.22002/D1.20087)
- [MIT Indoor67](https://web.mit.edu/torralba/www/indoor.html)
- [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

Use the included custom retrieval partitions rather than substituting official classification splits. See the runtime documentation for data-root remapping and the historical image-path convention used by deterministic degradations. The source release does not grant rights to redistribute dataset images.

## Experiments

Read [original training instructions](docs/original_training.md) before replaying an archived configuration. Main established-objective experiments use the saved settings, including 30 epochs, 256 training pairs, and 20 training/evaluation workers. Exploratory runs retain their separate settings. The exact archived inputs for the separate auxiliary blur-training experiment were not recovered, so that experiment is not presented as runnable here; the independently applied blur evaluation is included in `controlled/`.

Read [controlled evaluation instructions](docs/controlled_evaluation.md) for the six historical HashNet checkpoints, decoded-RGB duplicate screening, and the 25 evaluation conditions. Checkpoint hashes identify the required weights; the network weight files themselves are not committed. The original and screened partitions and their metric conventions must remain distinct.

## Method attribution and interpretation

DA-HCR uses a weighted supervised contrastive formulation. The baseline objectives and SHC-inspired, NHL-style, and SAAT-inspired adaptations are those described in the manuscript; the adapted implementations are not presented as the respective authors' official software. See [method and software attribution](docs/attribution.md).

The historical checkpoints were trained and selected using the original partitions. Later duplicate screening does not make that training leakage-free. This repository does not claim that source release or a smoke test proves exact regeneration of every archived table value. The manuscript reports differences between historical scores and later recalculation; preserve those distinctions when using the code.

## Release checks

See [validation performed for this source release](docs/validation.md). Checks include numerical protocol tests, saved-configuration forwarding, actual AutoDL imports, and CPU forward passes using the six SHA-verified historical checkpoints. They are not a new training run or a rerun of the complete benchmark.

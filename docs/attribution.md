# Method and software attribution

The manuscript is the reference for the mathematical definitions and experimental interpretation of this release.

- Supervised contrastive learning: Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020. DA-HCR modifies positive weighting within this formulation; the underlying supervised contrastive framework is prior work.
- DSH, DPSH, DHN, HashNet, and GreedyHash: prior hashing objectives cited in the accompanying manuscript. This repository contains the common experimental implementations used for the manuscript, rather than a claim of independently reproducing every official implementation.
- SHC-inspired, NHL-style, and SAAT-inspired: adaptations described in the manuscript. Their names do not imply official software or complete implementations of every component in the cited methods.
- Controlled degradations: synthetic planar perspective transformations, independently applied Gaussian blur, and additive brightness changes as defined in the manuscript. The perspective suite is inspired by the editor-cited work of Zhukov et al. and is not an exact reproduction or a physical 3-D viewpoint benchmark.
- PyTorch, torchvision, NumPy, pandas, Pillow, PyYAML, matplotlib, and tqdm are external dependencies. Their respective authors and licenses apply to their software. ImageNet-pretrained ResNet-18 weights are obtained separately through torchvision and are not included here.

Existing source notices are retained. No blanket third-party license is inferred from algorithm citations. This source release does not assign an invented permissive license to code for which a license record was not recovered, and it does not redistribute dataset images or network checkpoints.

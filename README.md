# NeurAlign: Unified Brain Surface and Volume Registration (ICLR 2026)
### [Project Page](https://people.csail.mit.edu/abulnaga/neuralign) | [Paper](https://arxiv.org/abs/2512.19928) | [OpenReview](https://openreview.net/forum?id=7FvUJu63zq)

![Example result on MRI image and surface](https://people.csail.mit.edu/abulnaga/neuralign/teaser.png)

NeurAlign is an image registration model that unifies brain surface and volume registration. NeurAlign aligns both cortical and subcortical structures in 3D space without requiring input meshes. This repository contains code to run inference (registration) on your data. Training code coming soon. 


This repository contains:
- Source code (in `./src/`) for the registration model and inference script.
- Pre-trained model weights in `./models/`. Two models are provided, trained with different regularization strengths: `model_k1.pt` ($\kappa$=1) and `model_k10.pt` ($\kappa$=10).


## Setup

Clone the repository and install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/mabulnaga/neuralign
cd neuralign
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .
```

## Registration (Model Inference)

We provide a sample pair of images for registration.

To register the included example image pair using the default model (`k10`), run from the repo root:

```bash
python src/register.py \
    --moving  data/HLN-12-11_norm.mgz \
    --fixed   data/HLN-12-1_norm.mgz  \
    --model   models/model_k10.pt \
    --out_dir results/
```

This uses the following paths:
- **Moving image**: `data/HLN-12-11_norm.mgz`
- **Fixed image**: `data/HLN-12-1_norm.mgz`
- **Model**: `models/model_k10.pt`

Outputs are saved to `results/`:
- `results/HLN-12-11_norm_to_HLN-12-1_norm.nii.gz` — moving image warped into fixed space
- `results/HLN-12-11_norm_to_HLN-12-1_norm_warp.nii.gz` — forward displacement field (x, y, z components)


All arguments and their defaults:

| Argument | Default | Description |
|---|---|---|
| `--moving` | `data/HLN-12-11_norm.mgz` | Moving (source) image — `.nii`, `.nii.gz`, or `.mgz` |
| `--fixed` | `data/HLN-12-1_norm.mgz` | Fixed (target) image — `.nii`, `.nii.gz`, or `.mgz` |
| `--model` | `models/model_k10.pt` | Pre-trained model: `model_k1.pt` ($\kappa$=1) or `model_k10.pt` ($\kappa$=10) |
| `--out_dir` | `results` | Directory to save output files |

## Training 
Code coming soon

## Repository Structure

```
.
├── data/
│   ├── HLN-12-11_norm.mgz      # Example moving image
│   └── HLN-12-1_norm.mgz      # Example fixed image
├── models/
│   ├── model_k1.pt             # Pre-trained weights, $\kappa$=1
│   └── model_k10.pt            # Pre-trained weights, $\kappa$=10 (default in paper)
├── src/
│   ├── register.py             # Registration script
│   ├── model.py                # VxmDense network (self-contained)
│   └── layers.py               # SpatialTransformer, VecInt, ResizeTransform
├── pyproject.toml
└── README.md
```


## Model

The model is based on the [VoxelMorph](https://github.com/voxelmorph/voxelmorph) architecture. We release two variants of our model, `model_k1.pt` and `model_k10.pt`. `k` refers to the Dice loss weight, as in Equation (5) in the paper. A larger `k` produces higher structural alignment at the cost of field regularity. The paper results are all based on `model_k10.pt`. See Table A.1 in the paper for additional details.




## Citation

If you use this code, please consider citing our work:

```
@inproceedings{abulnaga2026neuralign,
    title={Unified Brain Surface and Volume Registration},
    author={Abulnaga, S Mazdak and Hoopes, Andrew and Hoffmann, Malte and Magnet, Robin and Ovsjanikov, Maks and Z{\"o}llei, Lilla and Guttag, John and Fischl, Bruce and Dalca, Adrian},
    booktitle={Proceedings of the International Conference on Learning Representations},
    year={2026}
    }
```

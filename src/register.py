"""
register.py  —  Run 3-D registration with NeurAlign on one brain MRI pair.

Run from the repo root:

    python src/register.py

or with custom paths / model choice:

    python src/register.py \\
        --moving  data/Colin27-1_norm.mgz \\
        --fixed   data/HLN-12-1_norm.mgz  \\
        --weights   models/model_k10.pt \\
        --out_dir results/

Outputs (saved to --out_dir)
-----------------------------
  warped.nii.gz  : moving image warped into fixed space
  flow.nii.gz    : forward displacement field 
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import torch

# Add repo root to path so vxm package is importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

from vxm import networks as vxm_networks


# ---------------------------------------------------------------------------
# load model weights
# ---------------------------------------------------------------------------

def load_checkpt(checkpoint_fpath, model, optimizer=None):
    if hasattr(model, 'model'):
        model = model.model
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_volume(path):
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img


def preprocess(volume):
    return torch.from_numpy(volume / 255.0).unsqueeze(0).unsqueeze(0).float()


def get_stem(path):
    """Strip directory and all known extensions from a filename."""
    name = os.path.basename(path)
    for ext in ('.nii.gz', '.nii', '.mgz'):
        if name.endswith(ext):
            return name[:-len(ext)]
    return os.path.splitext(name)[0]


def save_volume(data, ref_img, path):
    """Save array as NIfTI reusing the affine from ref_img."""
    nib.save(nib.Nifti1Image(data, ref_img.affine), path)
    print(f"  Saved  → {path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="3-D brain MRI registration with NeurAlign",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--moving",  default="data/Colin27-1_norm.mgz")
    p.add_argument("--fixed",   default="data/HLN-12-1_norm.mgz")
    p.add_argument("--model", default="models/model_k10.pt",
                   help="Model Weight path")
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    weights_path = args.model
    print(f"Model  : ({weights_path})")

    print("\nLoading images ")
    moving_np, moving_img = load_volume(args.moving)
    fixed_np,  fixed_img  = load_volume(args.fixed)
    assert moving_np.shape == fixed_np.shape, (
        f"Shape mismatch: moving {moving_np.shape} vs fixed {fixed_np.shape}")
    inshape = tuple(moving_np.shape)
    print(f"  Shape : {inshape}")

    print("\nBuilding model ")
    model = vxm_networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[[64, 256, 256, 256], [256, 256, 256, 64]],
        bidir=True,
        int_steps=7,
        src_feats=1,
        trg_feats=1,
        int_downsize=1,
    )
    model, _, epoch = load_checkpt(weights_path, model, optimizer=None)
    model = model.to(device).eval()
    print(f"  Checkpoint epoch : {epoch}")

    print("\nRunning registration ")
    moving_t = preprocess(moving_np).to(device)
    fixed_t  = preprocess(fixed_np).to(device)
    with torch.no_grad():
        warped_t, pos_flow_t, _ = model(moving_t, fixed_t)

    os.makedirs(args.out_dir, exist_ok=True)
    moving_stem = get_stem(args.moving)
    fixed_stem  = get_stem(args.fixed)
    base = f"{moving_stem}_to_{fixed_stem}"

    print(f"\nSaving to '{args.out_dir}/' ")
    warped_np = warped_t.squeeze().cpu().numpy() * 255.0
    save_volume(warped_np.astype(np.float32), fixed_img,
                os.path.join(args.out_dir, f"{base}.nii.gz"))
    flow_np = pos_flow_t.squeeze().permute(1, 2, 3, 0).cpu().numpy()
    warp_path = os.path.join(args.out_dir, f"{base}_warp.nii.gz")
    nib.save(nib.Nifti1Image(flow_np.astype(np.float32), fixed_img.affine), warp_path)
    print(f" Output saved  to {warp_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

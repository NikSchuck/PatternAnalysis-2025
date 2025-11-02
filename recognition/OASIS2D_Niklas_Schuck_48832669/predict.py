# Inference + visualisation for Improved UNet on OASIS 2D slices.
# Saves per-sample PNGs showing image, ground-truth, prediction.

from pathlib import Path
import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from modules import ImprovedUNet
from dataset import build_loaders

base_dir = Path(r"C:\Users\nschu\OneDrive\Desktop\COMP3710\Final Report\OASIS Dataset")
split = "test" # choose from "train", "validate", "test"
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = Path("predictions")
max_samples = 16
base_ch = 32
depth = 4
dropout = 0.1

# util

def to_numpy_img(x):
    """
    x: (1,H,W) or (H,W). Returns float HxW in [0,1] for display
    """
    if x.ndim == 3:
        x = x[0]
    x = x.detach().cpu().float().numpy()
    # normalise for display
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x

def to_greyscale(mask, num_classes):
    """
    Convert integer mask HxW to color HxWx3 for plotting
    fixed palette for reproducibility
    """
    # simple fixed palette
    base = np.array([
        [0, 0, 0],        # 0 background
        [145, 145, 145],  # 1 grey
    ], dtype=np.uint8)
    base = np.vstack([base] * int(num_classes / len(base)))
    pal = base[:num_classes]
    out = pal[mask.clip(min=0, max=num_classes - 1)]
    return out

def save_triplet(img, gt_mask, pred_mask, out_path, num_classes):
    """Save side-by-side panels."""
    img_np = to_numpy_img(img)
    gt_np = to_greyscale(gt_mask, num_classes)
    pr_np = to_greyscale(pred_mask, num_classes)

    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
    ax.set_title("image"); ax.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(gt_np)
    ax.set_title("ground truth"); ax.axis("off")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(pr_np)
    ax.set_title("prediction"); ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# main

def main():

    # Check if CUDA is available
    print(f'Using device: {device}')
    
    # Data loaders
    loaders = build_loaders(base_dir=base_dir, batch_size=1)
    loader = loaders[split]

    # Load checkpoint
    ckpt = torch.load("outputs/best.ckpt", map_location=device)
    cfg = ckpt.get("config", {})
    num_classes = int(cfg.get("num_classes", 256))

    # Model
    model = ImprovedUNet(in_channels=1, base_ch=base_ch, depth=depth, dropout=dropout, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Inference loop
    saved = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)

            logits = model(x)                           # (1,C,H,W)
            pred = torch.argmax(logits, dim=1)[0]       # (H,W)

            # Save panel
            img = x[0].cpu()
            gt_mask = y[0].cpu().numpy().astype(np.int64)
            pr_mask = pred.cpu().numpy().astype(np.int64)

            img_name = Path(batch["image_path"][0]).stem
            out_path = out_dir / f"{split}_{img_name}.png"
            save_triplet(img, gt_mask, pr_mask, out_path, num_classes)

            saved += 1
            if saved >= max_samples:
                break

    print(f"[done] saved {saved} visualisations to {out_dir}")


if __name__ == "__main__":
    main()

import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import matplotlib.pyplot as plt

from modules import ImprovedUNet
from dataset import build_loaders

base_dir = Path(r"C:\Users\nschu\OneDrive\Desktop\COMP3710\Final Report\OASIS Dataset") # local training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 2
batch_size = 8
lr = 1e-3
weight_decay = 1e-4
out_dir = Path("Outputs")
base_ch = 32
depth = 4
dropout = 0.1

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def infer_num_classes(dl):
    """ peek a few batches to infer max label """
    max_label = 0
    it = iter(dl)
    for i in range(3):
        try:
            batch = next(it)
        except StopIteration:
            break
        y = batch["mask"]
        max_label = max(max_label, int(y.max().item()))
    return max_label + 1


def dice_per_class(logits, targets, num_classes):
    """
    Computes Dice per class on (B,C,H,W) logits and (B,H,W) targets.
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)  # (B,H,W)
        dices = []
        for c in range(num_classes):
            pred_c = (preds == c).float()
            targ_c = (targets == c).float()
            # handle empty target class
            denom = pred_c.sum() + targ_c.sum()
            if denom.item() == 0:
                dices.append(torch.tensor(1.0, device=logits.device))
                continue
            dice = 2.0 * (pred_c * targ_c).sum() / (denom + 1e-8)
            dices.append(dice)
        return torch.stack(dices).mean(), [d.item() for d in dices]


def save_curves(hist, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 4))
    xs = range(1, len(hist["train_loss"]) + 1)
    plt.plot(xs, hist["train_loss"], label="train_loss")
    plt.plot(xs, hist["val_loss"], label="val_loss")
    plt.plot(xs, hist["val_dice"], label="val_dice")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


# training

def run_epoch(model, loader, criterion, num_classes, optimizer, device=device):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0
    dices = []

    for batch in tqdm(loader, desc="train" if is_train else "eval"):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        mean_dice, _ = dice_per_class(logits, y, num_classes)
        dices.append(mean_dice.item())

    avg_loss = total_loss / max(total_batches, 1)
    avg_dice = float(np.mean(dices)) if dices else 0.0
    return avg_loss, avg_dice


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    losses, dices = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            md, _ = dice_per_class(logits, y, num_classes)
            dices.append(md.item())
    return float(np.mean(losses) if losses else 0.0), float(np.mean(dices) if dices else 0.0)


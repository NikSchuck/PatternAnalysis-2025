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
epochs = 4
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

def infer_num_classes(pick_set):
    """ 
    peek a few batches to infer max label
    input: loader["train" "test" or "validate"] to access dataset
    """
    max_label = 0
    it = iter(pick_set)
    for i in range(3):
        try:
            batch = next(it)
        except StopIteration:
            break
        y = batch["mask"]
        max_label = max(max_label, int(y.max().item()))
    return max_label + 1


def dice_per_class(logits, targets, num_classes):
    """ Computes Dice per class on (B,C,H,W) logits and (B,H,W) targets. """
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
    """ training curves for model visualisation """
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
    """ run one epoch """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0
    dices = []

    for batch in tqdm(loader, desc="train" if is_train else "eval"):
        x = batch["image"].to(device)
        y = batch["mask"].to(device)

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
    """ evaluates the mean loss and dice values from the run """
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
    return np.mean(losses), np.mean(dices)


# main

def main():

    # Check if CUDA is available
    print(f'Using device: {device}')

    # Data
    loaders = build_loaders(base_dir=base_dir, batch_size=batch_size, num_workers=2, pin_memory=True)

    # Number of classes is inferred
    num_classes = infer_num_classes(loaders["train"])
    print(f"[info] inferred num_classes = {num_classes}")

    # Model
    model = ImprovedUNet(in_channels=1, num_classes=num_classes, base_ch=base_ch, depth=depth, dropout=dropout).to(device)

    # Loss + Opt
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # Train
    best_val_dice = -1.0
    hist = {"train_loss": [], "val_loss": [], "val_dice": []}

    os.makedirs("outputs", exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        tr_loss, tr_dice = run_epoch(model, loaders["train"], criterion, num_classes, optimizer)
        val_loss, val_dice = evaluate(model, loaders["validate"], criterion, device, num_classes)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(val_loss)
        hist["val_dice"].append(val_dice)

        print(f"train: loss={tr_loss:.4f} dice={tr_dice:.4f}")
        print(f"valid: loss={val_loss:.4f} dice={val_dice:.4f}")

        # save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_path = Path(out_dir / "best.ckpt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                },
                ckpt_path,
            )
            print(f"[info] saved best checkpoint to {ckpt_path}")

        scheduler.step(val_dice)

    save_curves(hist, out_dir)

    # Test with best
    ckpt = torch.load(out_dir / "best.ckpt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_dice = evaluate(model, loaders["test"], criterion, device, num_classes)
    print(f"\nTest: loss={test_loss:.4f} dice={test_dice:.4f}")


if __name__ == "__main__":
    main()

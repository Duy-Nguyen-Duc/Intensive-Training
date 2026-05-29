"""Assignment B — ResNet50: train-from-scratch vs. transfer learning.

Trains two ResNet50 models on a >=5-class image dataset and compares them on
**accuracy** and **training time**, then plots the learning curves:

  1. SCRATCH  — ResNet50 with random weights, every layer trained.
  2. TRANSFER — ResNet50 pretrained on ImageNet, backbone FROZEN, only a new
                classifier head trained (feature extraction; see cnn_core.md §5).

Dataset
-------
Default: EuroSAT (10 classes, auto-downloaded by torchvision; also a popular
Kaggle dataset). To use any Kaggle dataset instead, download it into an
ImageFolder layout and pass --data-dir:

    # one-time Kaggle setup:
    pip install kaggle                      # put kaggle.json in ~/.kaggle/
    kaggle datasets download -d <owner/dataset> -p ./data --unzip
    # then point at the folder of class-subfolders (>=5 classes):
    python3 resnet50_transfer.py --data-dir ./data/<extracted_dir>

Run (default EuroSAT):
    python3 resnet50_transfer.py
Useful flags: --epochs 8  --train-size 6000  --val-size 2000  --batch 64
"""

from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")            # headless: write PNG, no display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),               # data augmentation
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_datasets(args):
    train_tf, eval_tf = build_transforms()

    if args.data_dir:
        # Any Kaggle dataset arranged as class-subfolders -> ImageFolder.
        # Two views over the same files so each split gets the right transform.
        full_train = datasets.ImageFolder(args.data_dir, transform=train_tf)
        full_eval = datasets.ImageFolder(args.data_dir, transform=eval_tf)
        classes = full_train.classes
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(full_train), generator=g).tolist()
        n_val = int(0.2 * len(perm))
        tr = Subset(full_train, perm[n_val:])
        va = Subset(full_eval, perm[:n_val])
    else:
        # Default: EuroSAT (10 classes), auto-download + random split.
        base_tr = datasets.EuroSAT(root="./data", download=True, transform=train_tf)
        base_va = datasets.EuroSAT(root="./data", download=True, transform=eval_tf)
        classes = base_tr.classes
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(base_tr), generator=g).tolist()
        n_val = min(args.val_size, len(perm) // 5)
        n_train = min(args.train_size, len(perm) - n_val)
        tr = Subset(base_tr, perm[:n_train])
        va = Subset(base_va, perm[n_train:n_train + n_val])

    return tr, va, classes


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

def build_model(mode, num_classes):
    """mode in {'scratch', 'transfer'}."""
    if mode == "scratch":
        model = resnet50(weights=None, num_classes=num_classes)
    else:  # transfer: pretrained backbone, frozen, new head
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for p in model.parameters():
            p.requires_grad = False                      # freeze backbone
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # new trainable head
    return model.to(DEVICE)


def build_optimizer(mode, model):
    if mode == "scratch":
        # Full network from random init: SGD + momentum + weight decay.
        return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9,
                               weight_decay=5e-4)
    # Transfer: only the head's params have requires_grad=True.
    head_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(head_params, lr=1e-3)


# ----------------------------------------------------------------------------
# Train / eval
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss_sum += crit(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def train_model(mode, train_loader, val_loader, num_classes, epochs):
    print(f"\n=== {mode.upper()} ===")
    model = build_model(mode, num_classes)
    opt = build_optimizer(mode, model)
    crit = nn.CrossEntropyLoss()
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable:,}")

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        tr_loss, tr_acc = loss_sum / total, correct / total
        va_loss, va_acc = evaluate(model, val_loader)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        print(f"epoch {epoch:2d} | train acc {tr_acc:.4f} | val acc {va_acc:.4f} "
              f"| val loss {va_loss:.4f}")

    train_time = time.perf_counter() - t0
    print(f"{mode} training time: {train_time:.1f}s | final val acc {history['val_acc'][-1]:.4f}")
    return history, train_time


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def plot_curves(hist_scratch, hist_transfer, path):
    epochs = range(1, len(hist_scratch["val_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, hist_scratch["val_acc"], "o-", label="scratch (val)")
    ax1.plot(epochs, hist_transfer["val_acc"], "s-", label="transfer (val)")
    ax1.set(title="Validation Accuracy", xlabel="epoch", ylabel="accuracy")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, hist_scratch["val_loss"], "o-", label="scratch (val)")
    ax2.plot(epochs, hist_transfer["val_loss"], "s-", label="transfer (val)")
    ax2.set(title="Validation Loss", xlabel="epoch", ylabel="loss")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle("ResNet50: scratch vs. transfer learning")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved learning curves -> {path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="ImageFolder dir (Kaggle dataset). Default: EuroSAT download.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train-size", type=int, default=6000)
    ap.add_argument("--val-size", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plot", default="learning_curves.png")
    args = ap.parse_args()

    print(f"device: {DEVICE}")
    train_set, val_set, classes = get_datasets(args)
    num_classes = len(classes)
    print(f"classes ({num_classes}): {classes}")
    print(f"train={len(train_set)}  val={len(val_set)}  batch={args.batch}  epochs={args.epochs}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    hist_scratch, t_scratch = train_model("scratch", train_loader, val_loader,
                                          num_classes, args.epochs)
    hist_transfer, t_transfer = train_model("transfer", train_loader, val_loader,
                                            num_classes, args.epochs)

    plot_curves(hist_scratch, hist_transfer, args.plot)

    # ---- comparison summary ----
    s_acc, t_acc = hist_scratch["val_acc"][-1], hist_transfer["val_acc"][-1]
    print("\n" + "=" * 60)
    print(f"{'':12}{'final val acc':>16}{'train time (s)':>18}")
    print(f"{'scratch':12}{s_acc:>16.4f}{t_scratch:>18.1f}")
    print(f"{'transfer':12}{t_acc:>16.4f}{t_transfer:>18.1f}")
    print("-" * 60)
    print(f"transfer vs scratch: {t_acc - s_acc:+.4f} accuracy, "
          f"{t_scratch / max(t_transfer, 1e-9):.2f}x the train time for scratch")
    print("=" * 60)


if __name__ == "__main__":
    main()

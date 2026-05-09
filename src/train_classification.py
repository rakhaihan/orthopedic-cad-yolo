import yaml
import argparse
import os
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import albumentations as A

try:
    from src.model import ResNetClassifier
    from src.dataset import XrayClassificationDataset
except ModuleNotFoundError:
    # Support direct execution: python src/train_classification.py
    from model import ResNetClassifier
    from dataset import XrayClassificationDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_weighted_loss(train_ds: XrayClassificationDataset, device: torch.device, enabled: bool) -> nn.Module:
    if not enabled:
        return nn.CrossEntropyLoss()

    class_counts = torch.zeros(2, dtype=torch.float32)
    for _, label in train_ds.samples:
        if 0 <= label < 2:
            class_counts[label] += 1

    class_counts = torch.clamp(class_counts, min=1.0)
    class_weights = class_counts.sum() / (2.0 * class_counts)
    return nn.CrossEntropyLoss(weight=class_weights.to(device))


def train(cfg_path: str) -> None:
    """Train binary fracture classifier from folder-structured dataset."""
    cfg_file = Path(cfg_path)
    if not cfg_file.is_absolute():
        cfg_file = (PROJECT_ROOT / cfg_file).resolve()

    with cfg_file.open() as f:
        cfg = yaml.safe_load(f)
    c = cfg["training"]["cls"]
    model = ResNetClassifier(backbone=cfg["model"]["cls_backbone"], num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    splits_dir = Path(cfg["data"]["splits_dir"])
    if not splits_dir.is_absolute():
        splits_dir = (PROJECT_ROOT / splits_dir).resolve()
    train_split = splits_dir / "train.txt"
    val_split = splits_dir / "val.txt"
    if train_split.exists() and val_split.exists():
        train_root = str(train_split)
        val_root = str(val_split)
    else:
        cls_dir = Path(cfg["data"]["classification_dir"])
        if not cls_dir.is_absolute():
            cls_dir = (PROJECT_ROOT / cls_dir).resolve()
        train_root = str(cls_dir / "train")
        val_root = str(cls_dir / "val")

    train_tfms = A.Compose(
        [
            A.Resize(c["img_size"], c["img_size"]),
            A.ShiftScaleRotate(
                shift_limit=0.04,
                scale_limit=0.08,
                rotate_limit=12,
                border_mode=0,
                p=0.6,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(std_range=(0.02, 0.06), p=1.0),
                ],
                p=0.25,
            ),
            A.Normalize(),
        ]
    )
    val_tfms = A.Compose(
        [
            A.Resize(c["img_size"], c["img_size"]),
            A.Normalize(),
        ]
    )

    train_ds = XrayClassificationDataset(train_root, img_size=c["img_size"], transforms=train_tfms)
    val_ds = XrayClassificationDataset(val_root, img_size=c["img_size"], transforms=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=c["batch_size"], shuffle=False, num_workers=0)

    criterion = _build_weighted_loss(train_ds, device, bool(c.get("use_class_weights", True)))
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(c["lr"]),
        weight_decay=float(c.get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(c.get("lr_reduce_factor", 0.5)),
        patience=int(c.get("lr_patience", 3)),
    )
    early_stop_patience = int(c.get("early_stopping_patience", 7))
    min_improve = float(c.get("min_delta", 1e-4))
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(c["epochs"]):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        val_acc = (correct / total) if total else 0.0
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{c['epochs']} | train_loss={train_loss:.4f} "
            f"| val_acc={val_acc:.4f} | lr={current_lr:.6f}"
        )

        if val_acc > (best_val_acc + min_improve):
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0
            runs_dir = (PROJECT_ROOT / "runs").resolve()
            os.makedirs(runs_dir, exist_ok=True)
            best_path = runs_dir / "classification_resnet50.pt"
            torch.save(model.state_dict(), str(best_path))
            print(f"[BEST] Saved checkpoint to {best_path}")
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1}: no val_acc improvement "
                f"for {early_stop_patience} epoch(s)."
            )
            break

    runs_dir = (PROJECT_ROOT / "runs").resolve()
    os.makedirs(runs_dir, exist_ok=True)
    last_path = runs_dir / "classification_resnet50_last.pt"
    torch.save(model.state_dict(), str(last_path))
    print(f"Saved last classifier weights to {last_path}")
    print(f"Best val_acc={best_val_acc:.4f} at epoch={best_epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)

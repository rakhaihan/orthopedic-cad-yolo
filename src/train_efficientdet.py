"""EfficientDet-D3 training entrypoint (transfer learning)."""

import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import EfficientDetWrapper
from src.dataset import XrayClassificationDataset


def train(cfg_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]["efficientdet"]
    img_size = cfg["training"]["cls"]["img_size"]
    data_dir = cfg["data"]["classification_dir"]
    splits_dir = Path(cfg["data"]["splits_dir"])
    train_split = splits_dir / "train.txt"
    val_split = splits_dir / "val.txt"

    if train_split.exists() and val_split.exists():
        train_ds = XrayClassificationDataset(str(train_split), img_size=img_size)
        val_ds = XrayClassificationDataset(str(val_split), img_size=img_size)
    else:
        train_ds = XrayClassificationDataset(f"{data_dir}/train", img_size=img_size)
        val_ds = XrayClassificationDataset(f"{data_dir}/val", img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientDetWrapper(
        model_name=tcfg["model_name"], pretrained=cfg["model"]["efficientdet_pretrained"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(tcfg["lr"]))

    for epoch in range(tcfg["epochs"]):
        model.model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.forward(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        model.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model.forward(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        acc = (correct / total) if total else 0.0
        print(f"Epoch {epoch + 1}/{tcfg['epochs']} | loss={loss_sum:.4f} | val_acc={acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)

import yaml
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import ResNetClassifier
from src.dataset import XrayClassificationDataset
import torch.nn as nn

def train(cfg_path: str) -> None:
    """Train binary fracture classifier from folder-structured dataset."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    c = cfg["training"]["cls"]
    model = ResNetClassifier(backbone=cfg["model"]["cls_backbone"], num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_root = f'{cfg["data"]["classification_dir"]}/train'
    val_root = f'{cfg["data"]["classification_dir"]}/val'

    train_ds = XrayClassificationDataset(train_root, img_size=c["img_size"])
    val_ds = XrayClassificationDataset(val_root, img_size=c["img_size"])
    train_loader = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=c["batch_size"], shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(c["lr"]))

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
        print(f"Epoch {epoch + 1}/{c['epochs']} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    os.makedirs("runs", exist_ok=True)
    torch.save(model.state_dict(), "runs/classification_resnet50.pt")
    print("Saved classifier weights to runs/classification_resnet50.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)

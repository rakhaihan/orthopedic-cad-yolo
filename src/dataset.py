import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class XrayDetectionDataset(Dataset):
    """Dataset for detection records containing boxes and labels."""

    def __init__(self, records: List[Dict], img_dir: str, transforms: Optional[A.Compose] = None):
        self.records = records
        self.img_dir = img_dir
        self.transforms = transforms or A.Compose([A.Normalize()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.img_dir, rec["file_name"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        boxes = rec.get("boxes", [])
        labels = rec.get("labels", [])
        if self.transforms:
            augmented = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4)), "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)}
        return img, target


class XrayClassificationDataset(Dataset):
    """Simple image-folder dataset with class subdirectories."""

    def __init__(self, root_dir: str, img_size: int = 224, transforms: Optional[A.Compose] = None):
        self.root_dir = Path(root_dir)
        self.samples: List[Tuple[Path, int]] = []
        classes = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {c.name: i for i, c in enumerate(classes)}
        self.transforms = transforms or A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(),
            ]
        )
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            for image_path in class_dir.rglob("*"):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((image_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = self.transforms(image=image)["image"]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image, torch.tensor(label, dtype=torch.long)

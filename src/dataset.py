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
        self.class_to_idx: Dict[str, int] = {}
        self.transforms = transforms or A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(),
            ]
        )
        self._build_samples()

    def _build_samples(self) -> None:
        # Support either:
        # 1) image-folder style directory: root/class_name/*.jpg
        # 2) split file (.txt): each line is an absolute/relative image path
        if self.root_dir.is_file():
            self._build_samples_from_split_file(self.root_dir)
            return

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Classification dataset path not found: {self.root_dir}. "
                "Expected a class directory or split text file."
            )

        classes = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        if not classes:
            raise ValueError(f"No class subdirectories found in: {self.root_dir}")

        self.class_to_idx = {c.name: i for i, c in enumerate(classes)}
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            for image_path in class_dir.rglob("*"):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((image_path, class_idx))

    def _build_samples_from_split_file(self, split_file: Path) -> None:
        split_lines = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        class_names = sorted({Path(line).parent.name for line in split_lines})
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for line in split_lines:
            image_path = Path(line)
            if not image_path.exists():
                # Resolve relative split entries against project root (cwd).
                image_path = (Path.cwd() / line).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Image referenced in split file does not exist: {line}")

            class_name = image_path.parent.name
            if class_name not in self.class_to_idx:
                raise ValueError(f"Unable to infer class from image path: {image_path}")
            self.samples.append((image_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError(f"No image samples found in split file: {split_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = self.transforms(image=image)["image"]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image, torch.tensor(label, dtype=torch.long)

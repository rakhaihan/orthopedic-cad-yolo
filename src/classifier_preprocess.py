"""Classifier input preprocessing aligned with XrayClassificationDataset (Albumentations)."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch

# Matches albumentations.A.Normalize() defaults (max_pixel_value=255).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess_rgb_tensor(
    rgb: np.ndarray,
    img_size: int = 224,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Resize (INTER_AREA), scale to [0,1], then ImageNet normalization — same pipeline as validation transforms.
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)


def imagenet_normalize_stats() -> Tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )

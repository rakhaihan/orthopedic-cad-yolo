"""Persist / load classifier class indices for CAM target selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


LABEL_MAP_FILENAME = "classification_label_map.yaml"


def infer_fracture_class_index(class_to_idx: dict[str, int]) -> int:
    """
    Pick the logits index whose name best matches a *fractured / positive* class.
    Robust to folder naming like fractured / non_fractured / fracture / positive.
    """
    if not class_to_idx:
        return 0

    normalized = {_normalize_name(k): int(v) for k, v in class_to_idx.items()}

    for key in ("fractured", "fracture", "positive", "pos", "yes", "1"):
        if key in normalized:
            return normalized[key]

    for name, idx in normalized.items():
        if name.startswith("non") and "frac" in name:
            continue
        if "frac" in name and "non" not in name:
            return idx

    return min(normalized.values())


def _normalize_name(name: str) -> str:
    return (
        name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .strip()
    )


def save_label_map(path: Path, class_to_idx: dict[str, int], fracture_class_idx: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = fracture_class_idx if fracture_class_idx is not None else infer_fracture_class_index(class_to_idx)
    payload: dict[str, Any] = {
        "class_to_idx": dict(sorted(class_to_idx.items(), key=lambda kv: kv[1])),
        "cam_fracture_class_index": idx,
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, default_flow_style=False)


def load_label_map(path: Path) -> tuple[dict[str, int], int] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "class_to_idx" not in data:
        return None
    class_to_idx = {str(k): int(v) for k, v in data["class_to_idx"].items()}
    idx = int(data.get("cam_fracture_class_index", infer_fracture_class_index(class_to_idx)))
    return class_to_idx, idx

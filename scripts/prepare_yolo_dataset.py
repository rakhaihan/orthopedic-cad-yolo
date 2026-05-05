#!/usr/bin/env python3
"""Prepare YOLO-standard dataset folders from split txt files."""

from pathlib import Path
import argparse
import shutil


def prepare_split(split_name: str, split_file: Path, labels_src: Path, out_root: Path) -> None:
    images_out = out_root / "images" / split_name
    labels_out = out_root / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_paths = [Path(line.strip()) for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept_images = 0
    kept_labels = 0
    missing_images = 0
    missing_labels = 0

    for image_path in image_paths:
        if not image_path.exists():
            missing_images += 1
            continue
        dst_image = images_out / image_path.name
        shutil.copy2(image_path, dst_image)
        kept_images += 1

        label_name = f"{image_path.stem}.txt"
        src_label = labels_src / label_name
        dst_label = labels_out / label_name
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
            kept_labels += 1
        else:
            # Negative (non-fracture) samples are valid without label files.
            missing_labels += 1

    print(
        f"{split_name}: images={kept_images}, labels={kept_labels}, "
        f"missing_images={missing_images}, no_label={missing_labels}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/splits", help="Directory containing train.txt/val.txt/test.txt")
    parser.add_argument("--labels-src", default="data/processed/labels_yolo", help="Directory with source YOLO label txt files")
    parser.add_argument("--out-root", default="data/yolo", help="Output root for YOLO dataset")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    labels_src = Path(args.labels_src)
    out_root = Path(args.out_root)

    for split in ("train", "val", "test"):
        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        prepare_split(split, split_file, labels_src, out_root)

    print(f"Prepared YOLO dataset at: {out_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def normalize_filename(value: str) -> str:
    name = str(value).strip().replace("\\", "/")
    lower = name.lower()
    if lower.endswith(".jpg.jpg"):
        return name[:-4]
    if not lower.endswith(VALID_EXTS):
        return f"{name}.jpg"
    return name


def resolve_relpath(filename: str, row: pd.Series, images_dir: Path) -> str:
    """Resolve relative image path for flat or class-subfolder dataset layouts."""
    fractured_val = int(row.get("fractured", row.get("label", 0)))
    candidates = [filename]
    if fractured_val == 1:
        candidates.extend([f"Fractured/{filename}", f"fractured/{filename}"])
    else:
        candidates.extend(
            [
                f"Non_fractured/{filename}",
                f"Non_Fractured/{filename}",
                f"non_fractured/{filename}",
            ]
        )
    for rel in candidates:
        if (images_dir / rel).exists():
            return rel.replace("\\", "/")
    return filename.replace("\\", "/")


def create_splits(
    csv_path,
    out_dir,
    seed=42,
    train_frac=0.7,
    val_frac=0.15,
    image_prefix="data/processed/images_640",
    images_dir="data/processed/images_640",
):
    df = pd.read_csv(csv_path)
    if "file_name" not in df.columns:
        source_col = "image_id" if "image_id" in df.columns else df.columns[0]
        df["file_name"] = df[source_col].astype(str).map(normalize_filename)
    else:
        df["file_name"] = df["file_name"].astype(str).map(normalize_filename)
    images_dir = Path(images_dir)
    df["relative_path"] = df.apply(lambda row: resolve_relpath(row["file_name"], row, images_dir), axis=1)

    # Stratify by anatomy region and fracture label if present.
    region_col = "anatomy_region" if "anatomy_region" in df.columns else None
    label_col = "label" if "label" in df.columns else ("fracture" if "fracture" in df.columns else None)
    if region_col and label_col:
        df["strata"] = df[region_col].astype(str) + "_" + df[label_col].astype(str)
    elif region_col:
        df["strata"] = df[region_col].astype(str)
    elif label_col:
        df["strata"] = df[label_col].astype(str)
    else:
        df["strata"] = "all"

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_frac),
        random_state=seed,
        stratify=df["strata"],
    )

    val_ratio_in_temp = val_frac / (1.0 - train_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=seed,
        stratify=temp_df["strata"],
    )

    train_files = train_df["relative_path"].tolist()
    val_files = val_df["relative_path"].tolist()
    test_files = test_df["relative_path"].tolist()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {'train': train_files, 'val': val_files, 'test': test_files}
    # simpan splits.json
    with open(out_dir / 'splits.json', 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)

    # tulis train/val/test .txt (satu path per baris)
    mapping = {"train": train_files, "val": val_files, "test": test_files}
    for name, lst in mapping.items():
        file_path = out_dir / f"{name}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            kept = 0
            dropped = 0
            for p in lst:
                full_path = (Path(image_prefix) / p).as_posix()
                if Path(full_path).exists():
                    f.write(full_path + "\n")
                    kept += 1
                else:
                    dropped += 1
        print(f"{name}.txt -> kept={kept}, dropped_missing={dropped}")

    print("Splits created in", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to dataset.csv")
    p.add_argument("--out", required=True, help="Output directory for splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-prefix", default="data/processed/images_640", help="Prefix path written to split txt files.")
    p.add_argument("--images-dir", default="data/processed/images_640", help="Directory used to validate image existence.")
    args = p.parse_args()
    create_splits(
        args.csv,
        args.out,
        seed=args.seed,
        image_prefix=args.image_prefix,
        images_dir=args.images_dir,
    )

#!/usr/bin/env python3
from PIL import Image, ImageFile
import os, sys, json
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = False

def validate_dir(img_dir, out_json="data/splits/invalid_images.json"):
    img_dir = Path(img_dir)
    invalid = []
    total = 0
    for root, _, files in os.walk(img_dir):
        for fn in files:
            if fn.lower().endswith((".jpg",".jpeg",".png")):
                total += 1
                p = Path(root) / fn
                try:
                    with Image.open(p) as im:
                        im.verify()   # quick check
                except Exception as e:
                    invalid.append({"path": str(p), "error": str(e)})
    print(f"Checked {total} files, found {len(invalid)} invalid")
    if invalid:
        os.makedirs(Path(out_json).parent, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(invalid, f, indent=2)
        print("Invalid list saved to", out_json)
    return invalid

if __name__ == "__main__":
    img_dir = sys.argv[1] if len(sys.argv)>1 else "data/raw/images"
    validate_dir(img_dir)

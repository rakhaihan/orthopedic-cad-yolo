import json
import os
from PIL import Image
from pathlib import Path

def make_crops(coco_json, images_dir, out_dir, split_files=None):
    coco = json.load(open(coco_json))
    id2img = {img['id']: img for img in coco['images']}
    for ann in coco['annotations']:
        img = id2img[ann['image_id']]
        fname = img['file_name']
        img_path = Path(images_dir) / fname
        if not img_path.exists():
            continue
        im = Image.open(img_path).convert('RGB')
        x,y,w,h = ann['bbox']
        crop = im.crop((x, y, x+w, y+h))
        cls = ann.get('category_id', 0)
        out_sub = Path(out_dir) / str(cls)
        out_sub.mkdir(parents=True, exist_ok=True)
        out_path = out_sub / f"{Path(fname).stem}_{ann['id']}.jpg"
        crop.save(out_path)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    make_crops(args.coco, args.images, args.out)

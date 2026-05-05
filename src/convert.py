import json
import os
from pathlib import Path

def coco_to_yolo(coco_json_path, images_dir, out_labels_dir, class_map=None):
    """
    Convert COCO bbox annotations to YOLO txt files.
    class_map: dict mapping coco category_id -> yolo class index
    """
    coco = json.load(open(coco_json_path))
    images = {img['id']: img for img in coco['images']}
    anns_by_image = {}
    for ann in coco['annotations']:
        anns_by_image.setdefault(ann['image_id'], []).append(ann)
    os.makedirs(out_labels_dir, exist_ok=True)
    for img_id, img in images.items():
        fname = img['file_name']
        w, h = img['width'], img['height']
        anns = anns_by_image.get(img_id, [])
        lines = []
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            x_center = (x + bw/2) / w
            y_center = (y + bh/2) / h
            w_rel = bw / w
            h_rel = bh / h
            cls = class_map.get(ann.get('category_id',0), 0) if class_map else ann.get('category_id',0)
            lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_rel:.6f} {h_rel:.6f}")
        label_path = Path(out_labels_dir) / (Path(fname).stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    coco_to_yolo(args.coco, args.images, args.out)

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import argparse
from src.convert import coco_to_yolo

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    coco_to_yolo(args.coco, args.images, args.out)

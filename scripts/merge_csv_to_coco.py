import json
import pandas as pd
import argparse

def merge(csv_path, coco_path, out_path):
    df = pd.read_csv(csv_path)
    if 'file_name' not in df.columns:
        df['file_name'] = df['image_id'].astype(str) + '.jpg'
    meta = df.set_index('file_name').to_dict(orient='index')
    coco = json.load(open(coco_path))
    for img in coco['images']:
        img_meta = meta.get(img['file_name'], {})
        img['meta'] = img_meta
    json.dump(coco, open(out_path, 'w'), indent=2)
    print("Saved merged COCO to", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--coco", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    merge(args.csv, args.coco, args.out)

import json, os
from PIL import Image

coco = json.load(open('Annotations/COCO JSON/COCO_fracture_masks.json'))
img_dir = 'images'  # sesuaikan path

# 1) cek semua file ada
file_names = {img['file_name'] for img in coco['images']}
missing = [f for f in file_names if not os.path.exists(os.path.join(img_dir, f))]
print('missing files:', missing)

# 2) cek bbox dalam batas gambar
id2img = {img['id']: img for img in coco['images']}
for ann in coco['annotations']:
    img = id2img[ann['image_id']]
    w,h = img['width'], img['height']
    x,y,ww,hh = ann['bbox']
    assert x>=0 and y>=0 and x+ww<=w+1 and y+hh<=h+1, f"bbox OOB {ann['id']}"
print('COCO basic validation passed')

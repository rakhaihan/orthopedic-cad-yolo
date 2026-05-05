import json
def test_coco_exists():
    coco = json.load(open('data/raw/annotations/COCO_fracture_masks.json'))
    assert 'images' in coco and 'annotations' in coco

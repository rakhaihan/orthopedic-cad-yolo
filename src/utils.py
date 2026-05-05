import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orthopedic-cad-yolo")

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

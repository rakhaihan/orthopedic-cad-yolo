import argparse
from src.model import YOLOWrapper, ResNetClassifier
import torch
import cv2
import numpy as np

def detect_image(yolo_model_path, image_path, conf=0.25):
    yolo = YOLOWrapper(model_path=yolo_model_path)
    results = yolo.predict(image_path, conf=conf)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    res = detect_image(args.model, args.image)
    print(res)

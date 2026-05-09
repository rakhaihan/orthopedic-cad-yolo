"""YOLOv8 detection training entrypoint."""

import argparse
from pathlib import Path
import yaml

from src.model import YOLOWrapper


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train(cfg_path: str) -> None:
    cfg_file = Path(cfg_path)
    if not cfg_file.is_absolute():
        cfg_file = (PROJECT_ROOT / cfg_file).resolve()

    with cfg_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ycfg = cfg["training"]["yolo"]
    yolo_data_yaml = Path(cfg["data"]["yolo_data_yaml"])
    if not yolo_data_yaml.is_absolute():
        yolo_data_yaml = (PROJECT_ROOT / yolo_data_yaml).resolve()

    model_seed = cfg["model"].get("yolo_model", "yolov8m.pt")
    device = cfg.get("device", "cpu")

    model = YOLOWrapper(model_path=model_seed)
    model.train(
        data_yaml=str(yolo_data_yaml),
        epochs=int(ycfg["epochs"]),
        imgsz=int(ycfg["img_size"]),
        batch=int(ycfg["batch_size"]),
        device=str(device),
        degrees=float(ycfg.get("augmentation", {}).get("degrees", 0.0)),
        fliplr=float(ycfg.get("augmentation", {}).get("fliplr", 0.5)),
        hsv_v=float(ycfg.get("augmentation", {}).get("hsv_v", 0.0)),
        mosaic=float(ycfg.get("augmentation", {}).get("mosaic", 0.0)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)

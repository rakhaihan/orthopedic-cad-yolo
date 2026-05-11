"""YOLOv8 detection training entrypoint."""

import argparse
from pathlib import Path
import yaml
import torch

from src.model import YOLOWrapper


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error" in msg


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
    initial_batch = int(ycfg["batch_size"])
    initial_imgsz = int(ycfg["img_size"])
    min_batch = int(ycfg.get("min_batch_size", 2))
    min_imgsz = int(ycfg.get("min_img_size", 416))

    model = YOLOWrapper(model_path=model_seed)
    batch = initial_batch
    imgsz = initial_imgsz

    while True:
        try:
            print(f"[INFO] Starting YOLO training with batch={batch}, imgsz={imgsz}, device={device}")
            model.train(
                data_yaml=str(yolo_data_yaml),
                epochs=int(ycfg["epochs"]),
                imgsz=imgsz,
                batch=batch,
                device=str(device),
                degrees=float(ycfg.get("augmentation", {}).get("degrees", 0.0)),
                fliplr=float(ycfg.get("augmentation", {}).get("fliplr", 0.5)),
                hsv_v=float(ycfg.get("augmentation", {}).get("hsv_v", 0.0)),
                mosaic=float(ycfg.get("augmentation", {}).get("mosaic", 0.0)),
            )
            return
        except RuntimeError as exc:
            if str(device).lower() != "cuda" or not _is_cuda_oom(exc):
                raise

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            next_batch = max(min_batch, batch // 2)
            if next_batch < batch:
                print(f"[WARN] CUDA OOM. Retrying with smaller batch: {batch} -> {next_batch}")
                batch = next_batch
                continue

            # Batch cannot be reduced further; reduce image size as final fallback.
            next_imgsz = max(min_imgsz, int(imgsz * 0.8))
            if next_imgsz < imgsz:
                print(f"[WARN] CUDA OOM at batch={batch}. Retrying with smaller imgsz: {imgsz} -> {next_imgsz}")
                imgsz = next_imgsz
                continue

            raise RuntimeError(
                "CUDA OOM persists even at minimal configured batch/img size. "
                "Set `training.yolo.batch_size` lower, reduce `training.yolo.img_size`, "
                "or train on CPU."
            ) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)

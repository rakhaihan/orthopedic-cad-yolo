from ultralytics import YOLO
import timm
import torch.nn as nn
import torch
from typing import Any, Dict, Optional

class YOLOWrapper:
    """Thin wrapper around Ultralytics YOLO for train/predict."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("yolov8m.pt")

    def train(
        self,
        data_yaml: str,
        epochs: int = 30,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "cuda",
        degrees: float = 0.0,
        fliplr: float = 0.5,
        hsv_v: float = 0.0,
        mosaic: float = 0.0,
    ) -> Any:
        """Train YOLO model with common augmentation controls."""
        return self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            degrees=degrees,
            fliplr=fliplr,
            hsv_v=hsv_v,
            mosaic=mosaic,
        )

    def predict(self, imgs: str, conf: float = 0.25) -> Any:
        return self.model.predict(source=imgs, conf=conf)


class EfficientDetWrapper:
    """Detector wrapper using timm EfficientDet variants."""

    def __init__(self, model_name: str = "tf_efficientdet_d3", pretrained: bool = True):
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=2)

    def to(self, device: torch.device) -> "EfficientDetWrapper":
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ResNetClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet50", num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.net = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

import torch
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from typing import Literal

def cam_for_model(model, input_tensor, target_layer, method: Literal["gradcam", "eigencam"] = "gradcam"):
    """Generate CAM mask with Grad-CAM or EigenCAM."""
    cam_cls = GradCAM if method == "gradcam" else EigenCAM
    cam = cam_cls(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    return grayscale_cam

def overlay_cam(img_rgb, cam_mask):
    """Overlay normalized CAM mask on RGB image."""
    cam_image = show_cam_on_image(img_rgb, cam_mask, use_rgb=True)
    return cam_image

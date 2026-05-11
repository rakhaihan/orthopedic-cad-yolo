import inspect
from typing import Iterable, Literal, Sequence

import numpy as np
import torch

try:
    from pytorch_grad_cam import GradCAM, EigenCAM, GradCAMPlusPlus, LayerCAM
except ImportError:  # pragma: no cover - optional combos
    GradCAMPlusPlus = None  # type: ignore[misc, assignment]
    LayerCAM = None  # type: ignore[misc, assignment]

CAMMethod = Literal["gradcam", "gradcam++", "eigencam", "layercam"]


def _resolve_cam_cls(method: CAMMethod):
    raw = method.lower().strip().replace("_", "").replace(" ", "")
    if "layercam" in raw:
        return LayerCAM if LayerCAM is not None else GradCAM
    if "eigen" in raw:
        return EigenCAM
    if "gradcam++" in method.lower() or "gradcamplus" in raw or "plusplus" in raw:
        return GradCAMPlusPlus if GradCAMPlusPlus is not None else GradCAM
    return GradCAM


def cam_for_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    method: CAMMethod = "gradcam",
    targets: Sequence | Iterable | None = None,
) -> np.ndarray:
    """Generate CAM mask with Grad-CAM variants or EigenCAM.

    Targets: pytorch_grad_cam ClassifierOutputTarget list, etc. If None,
    CAM library uses model's top predicted class gradients (often misleading).
    """
    cam_cls = _resolve_cam_cls(method)
    ctor_params = inspect.signature(cam_cls.__init__).parameters
    ctor_kwargs = {}
    if "use_cuda" in ctor_params:
        ctor_kwargs["use_cuda"] = torch.cuda.is_available()

    cam = cam_cls(model=model, target_layers=[target_layer], **ctor_kwargs)

    fwd_params = inspect.signature(cam.__call__).parameters
    kwargs = dict(input_tensor=input_tensor, targets=targets)
    if "aug_smooth" in fwd_params:
        kwargs["aug_smooth"] = False
    if "eigen_smooth" in fwd_params:
        kwargs["eigen_smooth"] = False

    grayscale_cam = cam(**kwargs)[0]
    getattr(cam, "release_hooks", lambda: None)()

    out = grayscale_cam.astype(np.float32)
    omin, omax = float(out.min()), float(out.max())
    if omax - omin > 1e-6:
        out = (out - omin) / (omax - omin)
    return out


def bbox_guided_classifier_cam(
    model: torch.nn.Module,
    rgb_image_uint8: np.ndarray,
    detect_result,
    target_layer,
    preprocess_fn,
    *,
    img_size: int,
    device: torch.device,
    method: CAMMethod,
    fracture_class_idx: int,
) -> np.ndarray | None:
    """
    Crop each detected box with margin, compute CAM focused on fracture logit, upsample paste with max pooling.
    """
    try:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        return None

    boxes = getattr(detect_result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    h, w = rgb_image_uint8.shape[:2]
    agg = np.zeros((h, w), dtype=np.float32)

    xyxy = boxes.xyxy.detach().cpu().numpy()
    targets = [ClassifierOutputTarget(fracture_class_idx)]

    for box in xyxy:
        x1, y1, x2, y2 = box.astype(np.float64)
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        px = max(12.0, 0.12 * bw)
        py = max(12.0, 0.12 * bh)
        xi1 = int(max(0, np.floor(x1 - px)))
        yi1 = int(max(0, np.floor(y1 - py)))
        xi2 = int(min(w, np.ceil(x2 + px)))
        yi2 = int(min(h, np.ceil(y2 + py)))

        crop = rgb_image_uint8[yi1:yi2, xi1:xi2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            continue

        input_tensor = preprocess_fn(crop, img_size, device)

        cam_map = cam_for_model(
            model,
            input_tensor,
            target_layer,
            method=method,
            targets=targets,
        )
        ch, cw = cam_map.shape[:2]

        cw_target = xi2 - xi1
        ch_target = yi2 - yi1
        if cw_target <= 0 or ch_target <= 0:
            continue

        roi_w = xi2 - xi1
        roi_h = yi2 - yi1
        resized = cv2_resize_like(cam_map, (roi_w, roi_h))
        agg[yi1:yi2, xi1:xi2] = np.maximum(agg[yi1:yi2, xi1:xi2], resized)


    m = float(np.max(agg))
    if m < 1e-6:
        return None
    return agg / m


def cv2_resize_like(arr: np.ndarray, size_xy: tuple[int, int]) -> np.ndarray:
    import cv2

    cw_target, ch_target = size_xy
    return cv2.resize(arr, (cw_target, ch_target), interpolation=cv2.INTER_LINEAR)


def overlay_cam(img_rgb: np.ndarray, cam_mask: np.ndarray):
    """Overlay normalized CAM mask on RGB image."""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    rgb01 = img_rgb.astype(np.float32) / 255.0
    cam_image = show_cam_on_image(rgb01, cam_mask, use_rgb=True)
    return cam_image

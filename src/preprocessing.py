import os
import argparse
from pathlib import Path
import cv2
from src.utils import ensure_dir
from typing import Tuple

def apply_clahe_gray(img):
    """Apply CLAHE on grayscale image for better local contrast."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def process_image(in_path: Path, out_path: Path, size: Tuple[int, int] = (640, 640), gaussian_kernel: int = 5):
    """Read, denoise, normalize contrast, resize, and save image."""
    img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {in_path}")
    img = apply_clahe_gray(img)
    if gaussian_kernel > 1:
        img = cv2.GaussianBlur(img, (gaussian_kernel, gaussian_kernel), 0)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # Keep detector input channel-compatible with RGB backbones.
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_path), img)

def batch_process(
    input_dir: str,
    output_dir: str,
    size: Tuple[int, int] = (640, 640),
    gaussian_kernel: int = 5,
    exts=(".jpg", ".png", ".jpeg"),
):
    ensure_dir(output_dir)
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(exts):
                in_path = Path(root) / fn
                rel = in_path.relative_to(input_dir)
                out_path = Path(output_dir) / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    process_image(in_path, out_path, size, gaussian_kernel=gaussian_kernel)
                except Exception as e:
                    print("Error processing", in_path, e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--gaussian-kernel", type=int, default=5, help="Odd kernel size for Gaussian denoising.")
    args = parser.parse_args()
    kernel = args.gaussian_kernel if args.gaussian_kernel % 2 == 1 else args.gaussian_kernel + 1
    batch_process(args.input, args.output, tuple(args.size), gaussian_kernel=kernel)

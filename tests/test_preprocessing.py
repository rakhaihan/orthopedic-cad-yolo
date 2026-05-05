from src.preprocessing import apply_clahe_gray
import numpy as np

def test_clahe():
    img = (np.random.rand(100,100)*255).astype('uint8')
    out = apply_clahe_gray(img)
    assert out.shape == img.shape

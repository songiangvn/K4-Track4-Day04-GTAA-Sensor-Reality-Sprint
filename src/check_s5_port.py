"""Kiểm tra zoom_blur (OpenCV) trong benchmark.py khớp với bản gốc của S5.

Bản gốc: ImageMotionBlurFrontBack.zoom_blur + clipped_zoom trong Camera_corruptions.py của
github.com/thu-ml/3D_Corruptions_AD @ 48c23f7 (dùng scipy.ndimage.zoom). File gốc import thêm
imgaug, mmdet3d, Automold nên không import thẳng được; logic hai hàm được chép lại nguyên văn dưới đây.

Chạy:  python src/check_s5_port.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import zoom as scizoom

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import BDD100K_ROOT, zoom_blur


def s5_clipped_zoom(img, zoom_factor):
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2
    return scizoom(img[top0:top0 + ch0, top1:top1 + ch1], (zoom_factor, zoom_factor, 1), order=1)


def s5_zoom_blur(x, corruption):
    c = np.arange(1, 1 + corruption, 0.005)
    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        layer = s5_clipped_zoom(x, zoom_factor)[:x.shape[0], :x.shape[1], :]
        out[:layer.shape[0], :layer.shape[1]] += layer
    return (np.clip((x + out) / (len(c) + 1), 0, 1) * 255).astype(np.uint8)


def main() -> None:
    images = sorted((BDD100K_ROOT / "data").glob("*.jpg"))[:3]
    for path in images:
        image = cv2.imread(str(path))
        for severity in (1, 3, 5):
            corruption = 0.02 * severity
            diff = np.abs(s5_zoom_blur(image, corruption).astype(int) - zoom_blur(image, corruption).astype(int))
            print(f"{path.name}  severity {severity} (zoom {corruption:.2f}): "
                  f"mean|diff| = {diff.mean():.3f}, max|diff| = {diff.max()} mức xám")


if __name__ == "__main__":
    main()

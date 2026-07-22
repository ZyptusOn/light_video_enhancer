"""Pixel-format helpers for the D3D11 video-processor bridge."""

import cv2
import numpy as np


def bgr_to_nv12(bgr: np.ndarray, align_w: int = 0, align_h: int = 0) -> np.ndarray:
    """Convert BGR24 to tightly packed NV12, padding odd dimensions safely."""
    if bgr.ndim != 3 or bgr.shape[2] < 3:
        raise ValueError("DXVA 输入必须是 HxWx3 BGR 图像")
    height, width = bgr.shape[:2]
    target_width = max((width + 1) // 2 * 2, align_w)
    target_height = max((height + 1) // 2 * 2, align_h)
    target_width = (target_width + 1) // 2 * 2
    target_height = (target_height + 1) // 2 * 2
    source = np.ascontiguousarray(bgr[:, :, :3], dtype=np.uint8)
    if target_width != width or target_height != height:
        source = cv2.copyMakeBorder(
            source, 0, target_height - height, 0, target_width - width,
            cv2.BORDER_REPLICATE,
        )
    yuv = cv2.cvtColor(source, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].reshape(-1)
    half_height, half_width = target_height // 2, target_width // 2
    u = yuv[:, :, 1].reshape(half_height, 2, half_width, 2).mean(
        axis=(1, 3)
    ).astype(np.uint8).reshape(-1)
    v = yuv[:, :, 2].reshape(half_height, 2, half_width, 2).mean(
        axis=(1, 3)
    ).astype(np.uint8).reshape(-1)
    uv = np.empty(u.size * 2, dtype=np.uint8)
    uv[0::2], uv[1::2] = u, v
    return np.ascontiguousarray(np.concatenate((y, uv)))

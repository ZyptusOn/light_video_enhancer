"""Cheap frame-pair classification shared by RIFE execution modes."""

from typing import List

import cv2
import numpy as np


PAIR_NORMAL = 0
PAIR_STATIC = 1
PAIR_SCENE_CUT = 2

SSIM_IDENTICAL = 0.996
SSIM_SCENE_CUT = 0.20


def _thumbnail_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("场景检测输入必须是 HxWx3 BGR 图像")
    gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(
        np.float32) / 255.0


def thumbnail_ssim(frame0: np.ndarray, frame1: np.ndarray) -> float:
    """Return a low-cost global SSIM score over 32x32 luminance thumbnails."""
    first = _thumbnail_gray(frame0)
    second = _thumbnail_gray(frame1)
    mean0 = float(first.mean())
    mean1 = float(second.mean())
    centered0 = first - mean0
    centered1 = second - mean1
    variance0 = float(np.mean(centered0 * centered0))
    variance1 = float(np.mean(centered1 * centered1))
    covariance = float(np.mean(centered0 * centered1))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    denominator = ((mean0 * mean0 + mean1 * mean1 + c1) *
                   (variance0 + variance1 + c2))
    if denominator <= 0.0:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(((2.0 * mean0 * mean1 + c1) *
                  (2.0 * covariance + c2)) / denominator)


def classify_pair(frame0: np.ndarray, frame1: np.ndarray) -> int:
    score = thumbnail_ssim(frame0, frame1)
    if score > SSIM_IDENTICAL:
        return PAIR_STATIC
    if score < SSIM_SCENE_CUT:
        return PAIR_SCENE_CUT
    return PAIR_NORMAL


def skipped_intermediates(frame0: np.ndarray, frame1: np.ndarray,
                          multiplier: int, pair_mode: int) -> List[np.ndarray]:
    """Create intermediates without cross-scene blending when RIFE is skipped."""
    if pair_mode == PAIR_STATIC:
        return [frame0.copy() for _ in range(multiplier - 1)]
    if pair_mode == PAIR_SCENE_CUT:
        return [(frame0 if step * 2 <= multiplier else frame1).copy()
                for step in range(1, multiplier)]
    raise ValueError("普通帧对不能使用跳过推理输出")

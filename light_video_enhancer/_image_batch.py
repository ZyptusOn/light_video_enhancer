"""Shared image-directory helpers for portable NCNN backends.

The NCNN command line tools accept whole directories. Keeping the files in
one workspace lets adjacent NCNN stages pass results directly without a
Python ``imread``/``imwrite`` round trip between them.
"""

import os
import re
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def make_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def image_paths(directory: str) -> List[str]:
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith(_IMAGE_EXTENSIONS)
    ]


def write_frames(frames: Iterable[np.ndarray], directory: str,
                 label: str = "NCNN") -> int:
    make_directory(directory)
    count = 0
    for index, frame in enumerate(frames):
        path = os.path.join(directory, "%08d.png" % index)
        if not cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
            raise RuntimeError("无法写入 %s 临时帧: %s" % (label, path))
        count += 1
    return count


def validate_outputs(directory: str, expected: int, label: str) -> List[str]:
    files = image_paths(directory)
    if len(files) != expected:
        raise RuntimeError("%s 仅输出 %d/%d 帧" % (label, len(files), expected))
    return files


def read_frames(directory: str, expected: int,
                size: Optional[Tuple[int, int]] = None,
                label: str = "NCNN") -> List[np.ndarray]:
    files = validate_outputs(directory, expected, label)
    output: List[np.ndarray] = []
    for path in files:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("无法读取 %s 输出: %s" % (label, path))
        if size is not None and frame.shape[1::-1] != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
        output.append(np.ascontiguousarray(frame))
    return output


_NCNN_JOBS_PATTERN = re.compile(r"^[1-9]\d*:[1-9]\d*(?:,[1-9]\d*)*:[1-9]\d*$")


def _engine_environment(name: str, engine: str) -> str:
    suffix = "".join(char if char.isalnum() else "_"
                     for char in engine.upper()).strip("_")
    if suffix:
        scoped = os.environ.get("%s_%s" % (name, suffix), "").strip()
        if scoped:
            return scoped
    return os.environ.get(name, "").strip()


def ncnn_jobs(src_width: int, src_height: int,
              dst_width: Optional[int] = None,
              dst_height: Optional[int] = None,
              engine: str = "") -> str:
    """Choose load:process:save concurrency, with a benchmark override.

    ``LVE_NCNN_JOBS`` applies globally and an engine-specific variable such as
    ``LVE_NCNN_JOBS_RIFE`` takes precedence. Invalid values are ignored so a
    stale profile cannot prevent the portable CLI fallback from starting.
    """
    overridden = _engine_environment("LVE_NCNN_JOBS", engine)
    if overridden and _NCNN_JOBS_PATTERN.fullmatch(overridden):
        return overridden
    largest = src_width * src_height
    if dst_width and dst_height:
        largest = max(largest, dst_width * dst_height)
    if largest <= 1280 * 720:
        return "4:4:4"
    if largest <= 2560 * 1440:
        return "3:3:3"
    return "2:2:2"


def ncnn_tile(engine: str = "") -> int:
    """Return an optional measured tile override; zero keeps CLI auto sizing."""
    value = _engine_environment("LVE_NCNN_TILE", engine)
    if not value:
        return 0
    try:
        parsed = int(value)
    except ValueError:
        return 0
    return parsed if parsed == 0 or parsed >= 32 else 0

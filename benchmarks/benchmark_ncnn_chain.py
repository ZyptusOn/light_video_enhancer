"""Compare legacy Python round-trips with the NCNN directory chain fast path."""

import argparse
import gc
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from light_video_enhancer.config import ProcessConfig
from light_video_enhancer.fi import create_fi_engine
from light_video_enhancer.pipeline import VideoEnhancer
from light_video_enhancer.sr import create_sr_engine


def frames(width, height, count):
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    result = []
    for index in range(count):
        frame = np.empty((height, width, 3), np.uint8)
        frame[:, :, 0] = (x.astype(np.uint16) + index * 7) % 256
        frame[:, :, 1] = (y.astype(np.uint16) + index * 5) % 256
        frame[:, :, 2] = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2 + index * 3) % 256
        left = (index * 19) % max(1, width - 64)
        cv2.rectangle(frame, (left, height // 3),
                      (min(width - 1, left + 64), min(height - 1, height // 3 + 64)),
                      (240, 40, 180), -1)
        result.append(frame)
    return result


def engines(width, height, scale, sr_name, gpu):
    fi = create_fi_engine("rife_ncnn", ncnn_gpu=gpu)
    fi.initialize(width, height, 2)
    sr = create_sr_engine(sr_name, ncnn_gpu=gpu, quality="balanced")
    sr.initialize(width, height, width * scale, height * scale)
    return fi, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--legacy-batch", type=int, default=9)
    parser.add_argument("--sr", choices=("realcugan", "realesrgan", "esrgan"),
                        default="realcugan")
    args = parser.parse_args()
    source = frames(args.width, args.height, args.frames)

    fi, sr = engines(args.width, args.height, args.scale, args.sr, args.gpu)
    start = time.perf_counter()
    legacy = []
    first_chunk = True
    for chunk in VideoEnhancer._chunks(source, args.legacy_batch):
        transformed = sr.process_batch(fi.interpolate_batch(chunk))
        if not first_chunk:
            transformed = transformed[1:]
        legacy.extend(transformed)
        first_chunk = False
    legacy_seconds = time.perf_counter() - start
    fi.release()
    sr.release()
    if len(legacy) != (args.frames - 1) * 2 + 1:
        raise RuntimeError("unexpected legacy output")
    del legacy
    gc.collect()

    fi, sr = engines(args.width, args.height, args.scale, args.sr, args.gpu)
    config = ProcessConfig(fi_engine="rife_ncnn", sr_engine=args.sr,
                           fi_multiplier=2, sr_quality="balanced",
                           sr_first=False, ncnn_gpu=args.gpu)
    enhancer = VideoEnhancer(config)
    enhancer._src_width, enhancer._src_height = args.width, args.height
    enhancer._dst_width = args.width * args.scale
    enhancer._dst_height = args.height * args.scale
    enhancer._fi_engine, enhancer._sr_engine = fi, sr
    start = time.perf_counter()
    direct = []
    first_chunk = True
    for chunk in VideoEnhancer._chunks(source, enhancer._batch_size()):
        transformed = enhancer._transform(chunk)
        if not first_chunk:
            transformed = transformed[1:]
        direct.extend(transformed)
        first_chunk = False
    enhancer._release_engines()
    direct_seconds = time.perf_counter() - start

    expected = (args.frames - 1) * 2 + 1
    if len(direct) != expected or direct[0].shape[:2] != (
            args.height * args.scale, args.width * args.scale):
        raise RuntimeError("unexpected direct-chain output")
    print("legacy_seconds=%.3f" % legacy_seconds)
    print("direct_seconds=%.3f" % direct_seconds)
    print("speedup=%.3fx" % (legacy_seconds / direct_seconds))


if __name__ == "__main__":
    main()

"""Reproducible short-video smoke benchmark for optional heavyweight SR engines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np


def _frames(path: str, count: int, width: int, height: int):
    capture = cv2.VideoCapture(path)
    result = []
    try:
        while len(result) < count:
            ok, frame = capture.read()
            if not ok:
                break
            result.append(cv2.resize(
                frame, (width, height), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    if len(result) != count:
        raise RuntimeError("decoded %d/%d source frames" % (len(result), count))
    return result


def _sample_gpu(stop: threading.Event, samples: list[dict]) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    while not stop.wait(0.5):
        try:
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=5, creationflags=flags)
            values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
            samples.append({
                "gpu_percent": float(values[0]),
                "vram_mib": float(values[1]),
                "power_w": float(values[2]),
            })
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine", choices=("seedvr2", "flashvsr", "dloral"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--torch-python", required=True)
    parser.add_argument("--quality", default="fast")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--src-width", type=int, default=320)
    parser.add_argument("--src-height", type=int, default=180)
    parser.add_argument("--dst-width", type=int, default=640)
    parser.add_argument("--dst-height", type=int, default=360)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _frames(
        args.input, args.frames, args.src_width, args.src_height)
    if args.engine == "seedvr2":
        from light_video_enhancer.sr.seedvr2 import SeedVR2Engine
        engine = SeedVR2Engine(
            torch_python=args.torch_python, quality=args.quality)
    elif args.engine == "flashvsr":
        from light_video_enhancer.sr.flashvsr import FlashVSREngine
        engine = FlashVSREngine(
            torch_python=args.torch_python, quality=args.quality)
    else:
        from light_video_enhancer.sr.dloral import DLoRALEngine
        engine = DLoRALEngine(
            torch_python=args.torch_python, quality=args.quality)

    initialized = time.perf_counter()
    engine.initialize(
        args.src_width, args.src_height, args.dst_width, args.dst_height)
    init_seconds = time.perf_counter() - initialized
    stop = threading.Event()
    samples: list[dict] = []
    monitor = threading.Thread(
        target=_sample_gpu, args=(stop, samples), daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        outputs = engine.process_batch(frames)
    finally:
        inference_seconds = time.perf_counter() - started
        stop.set()
        monitor.join(timeout=6)
        engine.release()

    for index, frame in enumerate(outputs):
        if not cv2.imwrite(
                str(output_dir / ("%02d.png" % index)), frame,
                [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise RuntimeError("cannot write output frame %d" % index)
    array = np.stack(outputs)
    report = {
        "engine": args.engine,
        "quality": args.quality,
        "input": os.path.abspath(args.input),
        "input_frames": len(frames),
        "output_frames": len(outputs),
        "source_size": [args.src_width, args.src_height],
        "output_size": [args.dst_width, args.dst_height],
        "init_seconds": init_seconds,
        "inference_seconds": inference_seconds,
        "output_fps": len(outputs) / inference_seconds,
        "output_mean": float(array.mean()),
        "output_std": float(array.std()),
        "output_min": int(array.min()),
        "output_max": int(array.max()),
        "gpu_samples": len(samples),
        "peak_gpu_percent": max(
            (item["gpu_percent"] for item in samples), default=None),
        "peak_vram_mib": max(
            (item["vram_mib"] for item in samples), default=None),
        "peak_power_w": max(
            (item["power_w"] for item in samples), default=None),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

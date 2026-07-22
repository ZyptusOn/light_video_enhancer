"""Small reproducible RIFE -> NV-VFX -> NVENC throughput benchmark."""

import argparse
import os
import tempfile
import sys
import time

import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from light_video_enhancer.config import EncodeConfig, ProcessConfig
from light_video_enhancer.ffmpeg_bridge import FFmpegVideoEncoder
from light_video_enhancer.pipeline import VideoEnhancer


def make_input(path: str, frames: int, width: int, height: int) -> None:
    encoder = FFmpegVideoEncoder(
        path, width, height, 30.0, codec="libx264", preset="fast", crf=20)
    encoder.open()
    try:
        yy, xx = np.mgrid[:height, :width]
        for index in range(frames):
            frame = np.empty((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (xx // 4 + index * 7) % 256
            frame[:, :, 1] = (yy // 3 + index * 5) % 256
            frame[:, :, 2] = ((xx + yy) // 8 + index * 11) % 256
            left = (index * 29) % max(1, width - 180)
            frame[height // 3:height // 3 + 120, left:left + 180] = (30, 230, 250)
            encoder.encode(frame)
    finally:
        encoder.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--torch-python", required=True)
    parser.add_argument("--work-dir", default=os.path.join(tempfile.gettempdir(), "lve-benchmark"))
    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    source = os.path.join(args.work_dir, "input_%dx%d_%df.mp4" % (
        args.width, args.height, args.frames))
    output = os.path.join(args.work_dir, "output.mp4")
    if not os.path.isfile(source):
        make_input(source, args.frames, args.width, args.height)
    if os.path.exists(output):
        os.remove(output)
    config = ProcessConfig(
        input_path=source, output_path=output, scale=2.0,
        sr_engine="nvvfx", fi_engine="rife", fi_multiplier=2,
        sr_quality="quality", torch_python=args.torch_python,
        encode=EncodeConfig(codec="hevc_nvenc", preset="fast", crf=25,
                            copy_audio=False, overwrite=True),
    )
    started = time.perf_counter()
    VideoEnhancer(config).run()
    elapsed = time.perf_counter() - started
    output_frames = args.frames * 2 - 1
    print("BENCHMARK elapsed=%.3f input_fps=%.3f output_fps=%.3f bytes=%d" % (
        elapsed, args.frames / elapsed, output_frames / elapsed, os.path.getsize(output)))


if __name__ == "__main__":
    main()

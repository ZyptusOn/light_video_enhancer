import argparse
import os
import sys
from typing import Optional

from ._logging import get_logger
from .config import EncodeConfig, ProcessConfig
from .encoding import CLI_CODEC_CHOICES, canonical_codec

_log = get_logger(__name__)


def _auto_output(input_path: str, scale: float, fi_engine: str,
                 fi_mult: int, container: str, sr_engine: str = "auto") -> str:
    directory = os.path.dirname(os.path.abspath(input_path))
    base = os.path.splitext(os.path.basename(input_path))[0]
    tags = []
    if sr_engine != "none" and scale != 1.0:
        tags.append("x%s" % ("%.2f" % scale).rstrip("0").rstrip("."))
    if fi_engine != "none":
        tags.append("f%d" % fi_mult)
    safe_container = "".join(char for char in container if char.isalnum()).lower() or "mp4"
    return os.path.join(directory, "%s%s.%s" %
                        (base, "_" + "_".join(tags) if tags else "_enhanced", safe_container))


def _ncnn_gpu(value: str) -> Optional[int]:
    lowered = value.lower()
    if lowered == "auto":
        return None
    if lowered == "cpu":
        return -1
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("NCNN 设备应为 auto、cpu 或非负 GPU 编号") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("GPU 编号不能为负数")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lve",
        description="Light Video Enhancer - Windows 视频超分、插帧与转码",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-o", "--output", help="输出视频路径")
    parser.add_argument("-s", "--scale", type=float, default=2.0, help="超分倍率")
    parser.add_argument("-W", "--width", type=int, default=0, help="指定输出宽度")
    parser.add_argument("-H", "--height", type=int, default=0, help="指定输出高度")
    parser.add_argument("--sr-engine", default="auto", choices=[
        "auto", "dxva_vsr", "nvvfx", "realcugan", "realesrgan", "esrgan",
        "bicubic", "lanczos", "none"], help="超分引擎")
    parser.add_argument("--fi-engine", default="auto", choices=[
        "auto", "rife", "rife_ncnn", "dis", "optical_flow",
        "torch_flow", "blend", "none"], help="插帧引擎")
    parser.add_argument("--sr-quality", default="quality",
                        choices=["fast", "balanced", "quality", "ultra"], help="超分质量")
    parser.add_argument("--fi-quality", default="balanced",
                        choices=["ultra", "fast", "balanced", "quality"], help="插帧质量")
    parser.add_argument("--fi-multiplier", type=int, default=2, help="插帧倍率")
    parser.add_argument("--sr-first", action="store_true", help="先超分后插帧（显存和计算量更高）")
    parser.add_argument("--codec", default="auto", choices=CLI_CODEC_CHOICES, help="编码器")
    parser.add_argument("--preset", default="balanced", help="编码器速度/质量预设")
    parser.add_argument("--crf", type=int, default=23, help="CQ/CRF 质量值（越小质量越高，0-63）")
    parser.add_argument("--container", default="mp4", choices=["mp4", "mkv", "mov"], help="输出容器")
    parser.add_argument("--fps", type=float, help="目标输出帧率（会自动补帧/丢帧以保持时长）")
    parser.add_argument("--start", type=float, help="从指定秒数开始")
    parser.add_argument("--duration", type=float, help="只处理指定秒数")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="PyTorch 设备")
    parser.add_argument("--torch-python", help="外部 CUDA PyTorch 的 python.exe")
    parser.add_argument("--ncnn-gpu", type=_ncnn_gpu, default=None,
                        metavar="auto|cpu|INDEX", help="NCNN Vulkan 设备")
    parser.add_argument("--no-audio", action="store_true", help="不复制源音频")
    parser.add_argument("-y", "--overwrite", action="store_true", help="覆盖已有输出")
    parser.add_argument("--keep-partial", action="store_true", help="失败/取消时保留部分文件")
    return parser


def parse_args(argv=None) -> ProcessConfig:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.device == "cpu" and (args.sr_engine == "nvvfx" or args.fi_engine == "torch_flow"):
        parser.error("NVIDIA VFX 和 CUDA 光流不能使用 --device cpu")
    output = args.output or _auto_output(
        args.input, args.scale, args.fi_engine, args.fi_multiplier,
        args.container, args.sr_engine)
    container = os.path.splitext(output)[1].lstrip(".").lower() or args.container
    encode = EncodeConfig(
        codec=canonical_codec(args.codec), preset=args.preset, crf=args.crf,
        pixel_format="yuv420p", container=container,
        copy_audio=not args.no_audio, overwrite=args.overwrite)
    return ProcessConfig(
        input_path=args.input, output_path=output, width=args.width, height=args.height,
        scale=args.scale, sr_engine=args.sr_engine, fi_engine=args.fi_engine,
        sr_quality=args.sr_quality, fi_multiplier=args.fi_multiplier,
        fi_quality=args.fi_quality,
        encode=encode, fps=args.fps, start_time=args.start, duration=args.duration,
        device=args.device, torch_python=args.torch_python, sr_first=args.sr_first,
        ncnn_gpu=args.ncnn_gpu, keep_partial=args.keep_partial)

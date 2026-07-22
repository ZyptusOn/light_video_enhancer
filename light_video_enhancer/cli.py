import argparse
import os
import sys
from typing import Optional

from ._logging import get_logger
from .config import EncodeConfig, ProcessConfig
from .encoding import CLI_CODEC_CHOICES, canonical_codec
from .i18n import tr

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
        raise argparse.ArgumentTypeError(tr(
            "NCNN 设备应为 auto、cpu 或非负 GPU 编号",
            "NCNN device must be auto, cpu, or a non-negative GPU index")) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(tr("GPU 编号不能为负数", "GPU index cannot be negative"))
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lve",
        description=tr(
            "Light Video Enhancer - Windows 视频超分、插帧与转码",
            "Light Video Enhancer - video super resolution, interpolation, and transcoding for Windows"),
        epilog=tr(
            "全局语言选项：--language zh-CN|en-US（可放在任意位置）",
            "Global language option: --language zh-CN|en-US (accepted anywhere)"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", help=tr("输入视频路径", "input video path"))
    parser.add_argument("-o", "--output", help=tr("输出视频路径", "output video path"))
    parser.add_argument("-s", "--scale", type=float, default=2.0,
                        help=tr("超分倍率", "super-resolution scale"))
    parser.add_argument("-W", "--width", type=int, default=0,
                        help=tr("指定输出宽度", "explicit output width"))
    parser.add_argument("-H", "--height", type=int, default=0,
                        help=tr("指定输出高度", "explicit output height"))
    parser.add_argument("--sr-engine", default="auto", choices=[
        "auto", "dxva_vsr", "nvvfx", "realcugan", "realesrgan", "esrgan",
        "bicubic", "lanczos", "none"], help=tr("超分引擎", "super-resolution engine"))
    parser.add_argument("--fi-engine", default="auto", choices=[
        "auto", "rife", "rife_ncnn", "dis", "optical_flow",
        "torch_flow", "blend", "none"], help=tr("插帧引擎", "interpolation engine"))
    parser.add_argument("--sr-quality", default="quality",
                        choices=["fast", "balanced", "quality", "ultra"],
                        help=tr("超分质量", "super-resolution quality"))
    parser.add_argument("--fi-quality", default="balanced",
                        choices=["ultra", "fast", "balanced", "quality"],
                        help=tr("插帧质量", "interpolation quality"))
    parser.add_argument("--fi-multiplier", type=int, default=2,
                        help=tr("插帧倍率", "interpolation multiplier"))
    parser.add_argument("--sr-first", action="store_true",
                        help=tr("先超分后插帧（显存和计算量更高）",
                                "run super resolution before interpolation (more VRAM and compute)"))
    parser.add_argument("--codec", default="auto", choices=CLI_CODEC_CHOICES,
                        help=tr("编码器", "video encoder"))
    parser.add_argument("--preset", default="balanced",
                        help=tr("编码器速度/质量预设", "encoder speed/quality preset"))
    parser.add_argument("--crf", type=int, default=23,
                        help=tr("CQ/CRF 质量值（越小质量越高，0-63）",
                                "CQ/CRF quality value (lower is better, 0-63)"))
    parser.add_argument("--container", default="mp4", choices=["mp4", "mkv", "mov"],
                        help=tr("输出容器", "output container"))
    parser.add_argument("--fps", type=float,
                        help=tr("目标输出帧率（会自动补帧/丢帧以保持时长）",
                                "target frame rate (frames are added/dropped to preserve duration)"))
    parser.add_argument("--start", type=float, help=tr("从指定秒数开始", "start time in seconds"))
    parser.add_argument("--duration", type=float, help=tr("只处理指定秒数", "duration to process in seconds"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help=tr("PyTorch 设备", "PyTorch device"))
    parser.add_argument("--torch-python",
                        help=tr("外部 CUDA PyTorch 的 python.exe", "python.exe from an external CUDA PyTorch environment"))
    parser.add_argument("--ncnn-gpu", type=_ncnn_gpu, default=None,
                        metavar="auto|cpu|INDEX", help=tr("NCNN Vulkan 设备", "NCNN Vulkan device"))
    parser.add_argument("--no-audio", action="store_true", help=tr("不复制源音频", "do not copy source audio"))
    parser.add_argument("-y", "--overwrite", action="store_true", help=tr("覆盖已有输出", "overwrite existing output"))
    parser.add_argument("--keep-partial", action="store_true",
                        help=tr("失败/取消时保留部分文件", "keep partial output on failure or cancellation"))
    return parser


def parse_args(argv=None) -> ProcessConfig:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.device == "cpu" and (args.sr_engine == "nvvfx" or args.fi_engine == "torch_flow"):
        parser.error(tr(
            "NVIDIA VFX 和 CUDA 光流不能使用 --device cpu",
            "NVIDIA VFX and CUDA optical flow cannot use --device cpu"))
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

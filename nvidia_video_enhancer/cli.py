import argparse
import os
import sys
from .config import ProcessConfig, EncodeConfig


def _auto_output(input_path: str, scale: float, fi_engine: str,
                 fi_mult: int, container: str) -> str:
    dirname = os.path.dirname(os.path.abspath(input_path))
    base = os.path.splitext(os.path.basename(input_path))[0]
    tags = []
    if scale > 1.0:
        tags.append(f"x{scale:.1f}".rstrip('0').rstrip('.'))
    if fi_engine != "none":
        tags.append(f"f{fi_mult}")
    suffix = "_" + "_".join(tags) if tags else ""
    return os.path.join(dirname, f"{base}{suffix}.{container}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nvidia-video-enhancer",
        description="视频超分 & 插帧工具 (D3D11 VSR / 光流 / RIFE)",
    )
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-o", "--output", default=None,
                        help="输出视频路径 (默认: 输入同目录, 自动命名)")
    parser.add_argument(
        "-s", "--scale", type=float, default=2.0,
        help="超分倍率 (默认 2.0)"
    )
    parser.add_argument(
        "-W", "--width", type=int, default=0,
        help="输出宽度 (覆盖 --scale)"
    )
    parser.add_argument(
        "-H", "--height", type=int, default=0,
        help="输出高度 (覆盖 --scale)"
    )
    parser.add_argument(
        "--sr-engine", choices=["dxva_vsr", "nvvfx", "bicubic", "lanczos"],
        default="nvvfx",
        help="超分引擎"
    )
    parser.add_argument(
        "--fi-engine", choices=["dis", "rife", "torch_flow", "optical_flow", "blend", "none"],
        default="optical_flow",
        help="插帧引擎"
    )
    parser.add_argument(
        "--fi-quality", choices=["ultra", "fast", "balanced", "quality"],
        default="balanced",
        help="光流法质量: ultra(极速)/fast(快)/balanced/quality(最佳)"
    )
    parser.add_argument(
        "--fi-multiplier", type=int, default=2,
        help="插帧倍率 (2→60fps)"
    )
    parser.add_argument(
        "--codec", default="h264_nvenc",
        help="编码器 (h264_nvenc / hevc_nvenc / av1_nvenc)"
    )
    parser.add_argument("--preset", default="p7", help="NVENC preset")
    parser.add_argument("--crf", type=int, default=23, help="质量 (越小越好)")
    parser.add_argument("--container", default="mp4", help="容器 (mp4/mkv/mov)")
    parser.add_argument("--fps", type=float, help="覆盖输出帧率")
    parser.add_argument("--start", type=float, help="起始时间 (秒)")
    parser.add_argument("--duration", type=float, help="持续时长 (秒)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser


def parse_args(argv=None) -> ProcessConfig:
    parser = build_parser()
    args = parser.parse_args(argv or sys.argv[1:])

    output = args.output
    if not output:
        output = _auto_output(
            args.input, args.scale, args.fi_engine,
            args.fi_multiplier, args.container,
        )
        print(f"[信息] 自动输出: {output}")

    encode = EncodeConfig(
        codec=args.codec,
        preset=args.preset,
        crf=args.crf,
        pixel_format="yuv420p",
        container=args.container,
    )

    config = ProcessConfig(
        input_path=args.input,
        output_path=output,
        width=args.width,
        height=args.height,
        scale=args.scale,
        sr_engine=args.sr_engine,
        fi_engine=args.fi_engine,
        fi_multiplier=args.fi_multiplier,
        fi_quality=args.fi_quality,
        encode=encode,
        fps=args.fps,
        start_time=args.start,
        duration=args.duration,
        device=args.device,
    )
    return config

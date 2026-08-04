"""Interactive console entry point for the standalone backend executable."""

import ctypes
import os
import sys
from typing import Iterable, Optional

from . import __version__
from .encoding import CLI_CODEC_CHOICES
from .i18n import get_language, tr


_SR_ENGINES = (
    "auto", "dxva_vsr", "nvvfx", "span", "flashvsr", "seedvr2", "dloral", "osdenhancer", "sparkvsr",
    "realcugan", "realesrgan", "esrgan", "bicubic", "lanczos", "none",
)
_FI_ENGINES = (
    "auto", "rife", "ema_vfi", "vfimamba", "rife_ncnn", "ifrnet_ncnn",
    "dis", "optical_flow", "torch_flow", "blend", "none",
)
_QUALITIES = ("fast", "balanced", "quality", "ultra")


def _console_title() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.kernel32.SetConsoleTitleW(
            "Light Video Enhancer CLI %s" % __version__)
    except (AttributeError, OSError):
        pass


def _read(prompt: str) -> Optional[str]:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def _path(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        return value[1:-1]
    return value


def _choice(label_zh: str, label_en: str, values: Iterable[str],
            default: str) -> Optional[str]:
    choices = tuple(values)
    while True:
        value = _read("%s [%s] (%s): " % (
            tr(label_zh, label_en), default, "/".join(choices)))
        if value is None:
            return None
        value = value or default
        if value in choices:
            return value
        print(tr("请输入列表中的值。", "Enter one of the listed values."))


def _number(label_zh: str, label_en: str, default: str,
            minimum: float, maximum: float) -> Optional[str]:
    while True:
        value = _read("%s [%s]: " % (tr(label_zh, label_en), default))
        if value is None:
            return None
        value = value or default
        try:
            parsed = float(value)
        except ValueError:
            parsed = minimum - 1
        if minimum <= parsed <= maximum:
            return value
        print(tr(
            "请输入 %.1f 到 %.1f 之间的数字。" % (minimum, maximum),
            "Enter a number between %.1f and %.1f." % (minimum, maximum)))


def _yes_no(label_zh: str, label_en: str, default: bool) -> Optional[bool]:
    marker = "Y/n" if default else "y/N"
    while True:
        value = _read("%s [%s]: " % (tr(label_zh, label_en), marker))
        if value is None:
            return None
        if not value:
            return default
        lowered = value.lower()
        if lowered in {"y", "yes", "是", "好"}:
            return True
        if lowered in {"n", "no", "否", "不"}:
            return False
        print(tr("请输入 y 或 n。", "Enter y or n."))


def interactive_arguments() -> Optional[list]:
    """Collect a common processing configuration without hiding advanced CLI flags."""
    _console_title()
    executable = os.path.basename(sys.executable if getattr(sys, "frozen", False) else sys.argv[0])
    print("=" * 68)
    print("Light Video Enhancer CLI %s" % __version__)
    print(tr("语言：中文（跟随系统）", "Language: English (system default)"))
    print(tr(
        "这是可独立运行的完整命令行后端。当前向导覆盖常用选项；",
        "This is the complete standalone command-line backend. The wizard covers common options;"))
    print(tr(
        "全部高级参数请运行：%s --help" % executable,
        "run %s --help for every advanced option." % executable))
    print(tr("可把视频文件直接拖入此窗口。", "You can drag a video file into this window."))
    print("=" * 68)

    while True:
        raw_input = _read(tr("输入视频（q 退出）：", "Input video (q to quit): "))
        if raw_input is None or raw_input.lower() in {"q", "quit", "exit"}:
            return None
        input_path = _path(raw_input)
        if os.path.isfile(input_path):
            break
        print(tr("文件不存在，请重新输入。", "The file does not exist. Try again."))

    raw_output = _read(tr(
        "输出文件（留空自动命名）：", "Output file (blank for automatic name): "))
    if raw_output is None:
        return None
    output_path = _path(raw_output)
    scale = _number("超分倍率", "Super-resolution scale", "2", 1.0, 8.0)
    if scale is None:
        return None
    sr_engine = _choice("超分引擎", "Super-resolution engine", _SR_ENGINES, "auto")
    if sr_engine is None:
        return None
    fi_multiplier = _number("插帧倍率", "Interpolation multiplier", "2", 1.0, 4.0)
    if fi_multiplier is None:
        return None
    fi_engine = _choice("插帧引擎", "Interpolation engine", _FI_ENGINES, "auto")
    if fi_engine is None:
        return None
    sr_quality = _choice("超分质量", "Super-resolution quality", _QUALITIES, "quality")
    if sr_quality is None:
        return None
    fi_quality = _choice("插帧质量", "Interpolation quality", _QUALITIES, "balanced")
    if fi_quality is None:
        return None
    codec = _choice("编码器", "Encoder", CLI_CODEC_CHOICES, "auto")
    if codec is None:
        return None
    overwrite = _yes_no("允许覆盖已有输出", "Overwrite an existing output", False)
    if overwrite is None:
        return None

    arguments = [
        input_path,
        "--scale", scale,
        "--sr-engine", sr_engine,
        "--fi-engine", fi_engine,
        "--sr-quality", sr_quality,
        "--fi-quality", fi_quality,
        "--fi-multiplier", str(int(float(fi_multiplier))),
        "--codec", codec,
    ]
    if output_path:
        arguments.extend(["--output", output_path])
    if overwrite:
        arguments.append("--overwrite")
    print()
    print(tr("即将开始处理。按 Ctrl+C 可取消。", "Processing will start. Press Ctrl+C to cancel."))
    confirmed = _yes_no("继续", "Continue", True)
    return arguments if confirmed else None

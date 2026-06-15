#!/usr/bin/env python3
"""PyInstaller 入口 — 双击 → GUI / 拖拽文件 → 自动处理。

此文件放在 nvidia_video_enhancer/ 同级目录，PyInstaller 从这里打包。
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from nvidia_video_enhancer.utils import print_system_info
from nvidia_video_enhancer.pipeline import VideoEnhancer
from nvidia_video_enhancer.config import ProcessConfig, EncodeConfig
from nvidia_video_enhancer.cli import _auto_output


def _detect_best_engines():
    sr = "bicubic"
    fi = "optical_flow"

    # nvvfx 需要 torch + nvidia-vfx
    try:
        import importlib
        importlib.import_module("torch")
        importlib.import_module("nvvfx")
        sr = "nvvfx"
    except ImportError:
        pass

    # dxva_vsr 需要编译 bridge DLL
    from nvidia_video_enhancer._paths import data_file_exists
    if data_file_exists("bridge", "dxva_vsr_bridge.dll"):
        sr = "dxva_vsr"

    return sr, fi


def main():
    args = sys.argv[1:]
    clean_args = [a for a in args if not a.startswith("--_") and not a.startswith("-psn")]

    if not clean_args:
        print_system_info()
        from nvidia_video_enhancer.gui import main as gui_main
        gui_main()
        return

    print_system_info()

    for input_path in clean_args:
        if input_path == "--gui":
            from nvidia_video_enhancer.gui import main as gui_main
            gui_main()
            continue
        if not os.path.isfile(input_path):
            print(f"[错误] 文件不存在: {input_path}")
            continue

        output_path = _auto_output(input_path, 2.0, "optical_flow", 2, "mp4")
        print(f"[信息] 输入: {input_path}")
        print(f"[信息] 输出: {output_path}")

        sr_engine, fi_engine = _detect_best_engines()
        print(f"[信息] 超分: {sr_engine}, 插帧: {fi_engine}")

        encode = EncodeConfig(codec="h264_nvenc", preset="p7", crf=23, container="mp4")
        config = ProcessConfig(
            input_path=input_path,
            output_path=output_path,
            scale=2.0,
            sr_engine=sr_engine,
            fi_engine=fi_engine,
            fi_multiplier=2,
            encode=encode,
            device="cuda",
        )
        enhancer = VideoEnhancer(config)
        try:
            enhancer.run()
        except Exception as e:
            print(f"[错误] {e}")
            import traceback
            traceback.print_exc()

    print("\n处理完成，按 Enter 退出...")
    try:
        input()
    except EOFError:
        pass


if __name__ == "__main__":
    main()

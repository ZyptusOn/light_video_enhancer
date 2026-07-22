"""PyInstaller entry point: GUI by default, dropped files in batch mode."""

import os
import sys


def main() -> None:
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-psn")]
    if not args or args == ["--gui"]:
        from light_video_enhancer.gui import main as gui_main
        gui_main()
        return

    from light_video_enhancer.cli import _auto_output
    from light_video_enhancer.config import EncodeConfig, ProcessConfig
    from light_video_enhancer.pipeline import VideoEnhancer

    failures = []
    for input_path in args:
        if not os.path.isfile(input_path):
            failures.append("文件不存在: %s" % input_path)
            continue
        output = _auto_output(input_path, 2.0, "auto", 2, "mp4", "auto")
        config = ProcessConfig(
            input_path=input_path,
            output_path=output,
            scale=2.0,
            sr_engine="auto",
            fi_engine="auto",
            fi_multiplier=2,
            encode=EncodeConfig(codec="auto", preset="balanced"),
        )
        try:
            VideoEnhancer(config).run()
        except Exception as exc:
            failures.append("%s: %s" % (input_path, exc))
    if failures:
        message = "\n".join(failures)
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("部分文件处理失败", message, parent=root)
            root.destroy()
        except Exception:
            print(message, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

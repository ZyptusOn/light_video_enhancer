#!/usr/bin/env python3
"""NVE GUI — 轻量图形界面，自动检测引擎可用性。"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from .utils import check_engine_availability
from .pipeline import VideoEnhancer
from .config import ProcessConfig, EncodeConfig


def run_in_thread(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()


class NVEGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Enhancer")
        self.geometry("720x580")
        self.resizable(True, True)
        self.configure(padx=12, pady=12)

        self._running = False
        self._caps = check_engine_availability()
        self._sr_widgets = {}
        self._fi_widgets = {}
        self._build_ui()
        self._update_engine_state()

    def _build_ui(self):
        # ---------- 文件选择 ----------
        f1 = ttk.Labelframe(self, text="输入 / 输出", padding=8)
        f1.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(f1, text="输入:").grid(row=0, column=0, sticky="w")
        self._input_var = tk.StringVar()
        ttk.Entry(f1, textvariable=self._input_var, width=50).grid(row=0, column=1, padx=4)
        ttk.Button(f1, text="浏览", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(f1, text="输出:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._output_var = tk.StringVar()
        ttk.Entry(f1, textvariable=self._output_var, width=50).grid(row=1, column=1, padx=4, pady=(4, 0))
        ttk.Button(f1, text="浏览", command=self._browse_output).grid(row=1, column=2, pady=(4, 0))

        # ---------- 超分 ----------
        f2 = ttk.Labelframe(self, text="超分辨率", padding=8)
        f2.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(f2, text="引擎:").grid(row=0, column=0, sticky="w")
        self._sr_var = tk.StringVar(value=self._best_sr())
        self._sr_combo = ttk.Combobox(f2, textvariable=self._sr_var, state="readonly", width=12)
        self._sr_combo.grid(row=0, column=1, padx=4, sticky="w")

        ttk.Label(f2, text="倍率:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._scale_var = tk.DoubleVar(value=2.0)
        self._sr_spin = ttk.Spinbox(f2, textvariable=self._scale_var, from_=1.0, to=4.0,
                                    increment=0.5, width=6)
        self._sr_spin.grid(row=1, column=1, padx=4, pady=(4, 0), sticky="w")

        # ---------- 插帧 ----------
        f3 = ttk.Labelframe(self, text="插帧", padding=8)
        f3.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(f3, text="引擎:").grid(row=0, column=0, sticky="w")
        self._fi_var = tk.StringVar(value=self._best_fi())
        self._fi_combo = ttk.Combobox(f3, textvariable=self._fi_var, state="readonly", width=12)
        self._fi_combo.grid(row=0, column=1, padx=4, sticky="w")

        ttk.Label(f3, text="倍率:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._fi_mult_var = tk.IntVar(value=2)
        self._fi_spin = ttk.Spinbox(f3, textvariable=self._fi_mult_var, from_=2, to=4,
                                    increment=1, width=6)
        self._fi_spin.grid(row=1, column=1, padx=4, pady=(4, 0), sticky="w")

        ttk.Label(f3, text="质量:").grid(row=1, column=2, sticky="w", pady=(4, 0), padx=(16, 0))
        self._fi_quality_var = tk.StringVar(value="balanced")
        ttk.Combobox(f3, textvariable=self._fi_quality_var,
                     values=["ultra", "fast", "balanced", "quality"],
                     state="readonly", width=10).grid(
            row=1, column=3, padx=4, pady=(4, 0), sticky="w")

        # ---------- 编码 ----------
        f4 = ttk.Labelframe(self, text="编码", padding=8)
        f4.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(f4, text="编码器:").grid(row=0, column=0, sticky="w")
        self._codec_var = tk.StringVar(value="h264_nvenc")
        ttk.Combobox(f4, textvariable=self._codec_var,
                     values=["h264_nvenc", "hevc_nvenc", "av1_nvenc"],
                     state="readonly", width=12).grid(row=0, column=1, padx=4, sticky="w")

        ttk.Label(f4, text="CRF:").grid(row=0, column=2, sticky="w", padx=(16, 0))
        self._crf_var = tk.IntVar(value=23)
        ttk.Spinbox(f4, textvariable=self._crf_var, from_=15, to=35, width=6).grid(row=0, column=3, padx=4, sticky="w")

        ttk.Label(f4, text="Preset:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._preset_var = tk.StringVar(value="p7")
        ttk.Combobox(f4, textvariable=self._preset_var,
                     values=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                     state="readonly", width=12).grid(row=1, column=1, padx=4, pady=(4, 0), sticky="w")

        ttk.Label(f4, text="容器:").grid(row=1, column=2, sticky="w", pady=(4, 0), padx=(16, 0))
        self._container_var = tk.StringVar(value="mp4")
        ttk.Combobox(f4, textvariable=self._container_var,
                     values=["mp4", "mkv", "mov"], state="readonly", width=6).grid(
            row=1, column=3, padx=4, pady=(4, 0), sticky="w")

        # ---------- 按钮 ----------
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=(0, 8))
        self._btn = ttk.Button(btn_frame, text="▶ 开始处理", command=self._start)
        self._btn.pack(side=tk.RIGHT)

        # ---------- 进度 ----------
        self._progress = ttk.Progressbar(self, mode="indeterminate")
        self._progress.pack(fill=tk.X, pady=(0, 8))

        # ---------- 日志 ----------
        self._log = scrolledtext.ScrolledText(self, height=10, state="normal", wrap="word")
        self._log.pack(fill=tk.BOTH, expand=True)

    # ====== 引擎检测 ======

    def _update_engine_state(self):
        caps = self._caps

        self._sr_map = {}
        sr_items = []
        for key in ("nvvfx", "dxva_vsr", "bicubic", "lanczos"):
            if key == "nvvfx":
                ok = caps["worker"] and caps["torch_cuda"] and caps["nvvfx"]
                label = f"NVIDIA VFX SDK{' (需torch+nvidia-vfx)' if not ok else ''}"
            elif key == "dxva_vsr":
                ok = caps["worker"] and caps["vsr_dll"]
                label = f"DXVA VSR{' (需bridge DLL)' if not ok else ''}"
            elif key == "bicubic":
                ok = True; label = "双三次"
            else:
                ok = True; label = "Lanczos"
            self._sr_map[label] = key
            sr_items.append(label)

        self._sr_combo["values"] = sr_items
        best_sr = self._best_sr()
        for label, key in self._sr_map.items():
            if key == best_sr:
                self._sr_var.set(label)
                break

        self._fi_map = {}
        fi_items = []
        for key in ("dis", "torch_flow", "optical_flow", "rife", "blend", "none"):
            if key == "rife":
                ok = caps["torch_cuda"]
                label = f"RIFE AI{' (需PyTorch)' if not ok else ''}"
            elif key == "torch_flow":
                ok = caps["torch_cuda"]
                label = f"GPU光流(SVP风格){' (需PyTorch)' if not ok else ''}"
            elif key == "dis":
                try:
                    import cv2; cv2.DISOpticalFlow_create
                    ok = True; label = "DIS 光流(SVP风格)"
                except Exception:
                    ok = False; label = "DIS 光流(需contrib)"
            elif key == "optical_flow":
                ok = True; label = "光流法(Farneback)"
            elif key == "blend":
                ok = True; label = "混合"
            else:
                ok = True; label = "不插帧"
            self._fi_map[label] = key
            fi_items.append(label)

        self._fi_combo["values"] = fi_items
        self._fi_var.set(fi_items[0])  # 光流法 always first

    def _best_sr(self):
        caps = self._caps
        if caps["worker"] and caps["torch_cuda"] and caps["nvvfx"]:
            return "nvvfx"
        if caps["worker"] and caps["vsr_dll"]:
            return "dxva_vsr"
        return "bicubic"

    def _best_fi(self):
        return "optical_flow"

    # ====== 文件浏览 ======

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="选择输入视频",
            filetypes=[("视频文件", "*.mp4 *.mkv *.mov *.avi *.webm"), ("所有文件", "*.*")])
        if path:
            self._input_var.set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="选择输出路径",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("MKV", "*.mkv"), ("MOV", "*.mov"), ("所有文件", "*.*")])
        if path:
            self._output_var.set(path)

    def _auto_output(self, input_path: str) -> str:
        dirname = os.path.dirname(os.path.abspath(input_path))
        base, _ = os.path.splitext(os.path.basename(input_path))
        tags = []
        s = self._scale_var.get()
        if s > 1.0:
            tags.append(f"x{s:.1f}".rstrip('0').rstrip('.'))
        fi_key = self._fi_map.get(self._fi_var.get(), "optical_flow")
        if fi_key != "none":
            tags.append(f"f{self._fi_mult_var.get()}")
        suffix = "_" + "_".join(tags) if tags else ""
        return os.path.join(dirname, f"{base}{suffix}.{self._container_var.get()}")

    def _log_msg(self, msg: str):
        self._log.insert(tk.END, msg + "\n")
        self._log.see(tk.END)

    # ====== 处理 ======

    def _start(self):
        if self._running:
            return

        sr_label = self._sr_var.get()
        fi_label = self._fi_var.get()
        sr = self._sr_map.get(sr_label, "bicubic")
        fi = self._fi_map.get(fi_label, "optical_flow")
        caps = self._caps

        if sr == "nvvfx" and not (caps["worker"] and caps["torch_cuda"] and caps["nvvfx"]):
            messagebox.showerror("不可用", "NVIDIA VFX SDK 不可用 (需 PyTorch + nvidia-vfx)")
            return
        if sr == "dxva_vsr" and not (caps["worker"] and caps["vsr_dll"]):
            messagebox.showerror("不可用", "DXVA VSR 不可用 (需编译 bridge DLL)")
            return
        if fi == "rife" and not caps["torch_cuda"]:
            messagebox.showerror("不可用", "RIFE AI 不可用 (需 PyTorch CUDA)")
            return

        inp = self._input_var.get().strip()
        out = self._output_var.get().strip()
        if not inp:
            messagebox.showerror("错误", "请选择输入文件")
            return
        if not out:
            out = self._auto_output(inp)
            self._output_var.set(out)
            self._log_msg(f"[信息] 自动输出: {out}")

        self._running = True
        self._btn.configure(text="处理中...", state="disabled")
        self._progress.start()
        self._log.delete("1.0", tk.END)
        run_in_thread(self._process, inp, out, sr, fi)

    def _process(self, inp, out, sr, fi):
        try:
            encode = EncodeConfig(
                codec=self._codec_var.get(),
                preset=self._preset_var.get(),
                crf=self._crf_var.get(),
                container=self._container_var.get(),
            )
            config = ProcessConfig(
                input_path=inp,
                output_path=out,
                scale=self._scale_var.get(),
                sr_engine=sr,
                fi_engine=fi,
                fi_multiplier=self._fi_mult_var.get(),
                fi_quality=self._fi_quality_var.get(),
                encode=encode,
                device="cuda",
            )

            enhancer = VideoEnhancer(config)
            import builtins
            _orig_print = builtins.print

            def gui_print(*args, **kwargs):
                msg = " ".join(str(a) for a in args)
                self.after(0, self._log_msg, msg)
                _orig_print(msg)

            builtins.print = gui_print
            try:
                enhancer.run()
            finally:
                builtins.print = _orig_print

            self.after(0, self._done, True, "")
        except Exception as e:
            self.after(0, self._done, False, str(e))

    def _done(self, ok: bool, err: str):
        self._running = False
        self._progress.stop()
        self._btn.configure(text="▶ 开始处理", state="normal")
        if ok:
            self._log_msg("\n✓ 处理完成!")
        else:
            self._log_msg(f"\n✗ 错误: {err}")


def main():
    import builtins
    _print = builtins.print

    app = NVEGUI()

    caps = app._caps
    _print("[信息] 引擎检测:")
    _print(f"  FFmpeg Worker: {'✓' if caps['worker'] else '✗'}")
    _print(f"  D3D11 Bridge:  {'✓' if caps['vsr_dll'] else '✗'}")
    _print(f"  PyTorch CUDA:  {'✓' if caps['torch_cuda'] else '✗'}")
    _print(f"  nvidia-vfx:    {'✓' if caps['nvvfx'] else '✗'}")
    _print(f"  超分默认: {app._best_sr()}, 插帧默认: {app._best_fi()}")

    app.mainloop()


if __name__ == "__main__":
    main()

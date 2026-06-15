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


_VSR_MAX_W = 4096
_VSR_MAX_H = 2160


class NVEGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Enhancer")
        self.geometry("760x620")
        self.resizable(True, True)
        self.configure(padx=12, pady=12)

        self._running = False
        self._caps = check_engine_availability()
        self._last_input = ""
        self._build_ui()
        self._update_engine_state()

    def _build_ui(self):
        f1 = ttk.Labelframe(self, text="输入 / 输出", padding=8)
        f1.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(f1, text="输入:").grid(row=0, column=0, sticky="w")
        self._input_var = tk.StringVar()
        self._input_var.trace_add("write", self._on_input_changed)
        ttk.Entry(f1, textvariable=self._input_var, width=52).grid(row=0, column=1, padx=4)
        ttk.Button(f1, text="浏览", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(f1, text="输出:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._output_var = tk.StringVar()
        ttk.Entry(f1, textvariable=self._output_var, width=52).grid(row=1, column=1, padx=4, pady=(4, 0))
        ttk.Button(f1, text="浏览", command=self._browse_output).grid(row=1, column=2, pady=(4, 0))

        f2 = ttk.Labelframe(self, text="超分辨率", padding=8)
        f2.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(f2, text="引擎:").grid(row=0, column=0, sticky="w")
        self._sr_var = tk.StringVar()
        self._sr_combo = ttk.Combobox(f2, textvariable=self._sr_var, state="readonly", width=30)
        self._sr_combo.grid(row=0, column=1, padx=4, sticky="w")
        self._sr_combo.bind("<<ComboboxSelected>>", self._on_sr_changed)

        ttk.Label(f2, text="倍率:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._scale_var = tk.DoubleVar(value=2.0)
        self._scale_var.trace_add("write", lambda *_: self._check_sr_limit())
        self._sr_spin = ttk.Spinbox(f2, textvariable=self._scale_var, from_=1.0, to=4.0,
                                    increment=0.5, width=6)
        self._sr_spin.grid(row=1, column=1, padx=4, pady=(4, 0), sticky="w")

        self._sr_warning_var = tk.StringVar()
        self._sr_warning_lbl = ttk.Label(f2, textvariable=self._sr_warning_var, foreground="red")
        self._sr_warning_lbl.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        f3 = ttk.Labelframe(self, text="插帧", padding=8)
        f3.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(f3, text="引擎:").grid(row=0, column=0, sticky="w")
        self._fi_var = tk.StringVar()
        self._fi_combo = ttk.Combobox(f3, textvariable=self._fi_var, state="readonly", width=30)
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

        f4 = ttk.Labelframe(self, text="编码", padding=8)
        f4.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(f4, text="编码器:").grid(row=0, column=0, sticky="w")
        self._codec_var = tk.StringVar(value="h264_nvenc")
        ttk.Combobox(f4, textvariable=self._codec_var,
                     values=["h264_nvenc", "hevc_nvenc", "av1_nvenc"],
                     state="readonly", width=14).grid(row=0, column=1, padx=4, sticky="w")

        ttk.Label(f4, text="CRF:").grid(row=0, column=2, sticky="w", padx=(16, 0))
        self._crf_var = tk.IntVar(value=23)
        ttk.Spinbox(f4, textvariable=self._crf_var, from_=15, to=35, width=6).grid(row=0, column=3, padx=4, sticky="w")

        ttk.Label(f4, text="Preset:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._preset_var = tk.StringVar(value="p7")
        ttk.Combobox(f4, textvariable=self._preset_var,
                     values=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                     state="readonly", width=14).grid(row=1, column=1, padx=4, pady=(4, 0), sticky="w")

        ttk.Label(f4, text="容器:").grid(row=1, column=2, sticky="w", pady=(4, 0), padx=(16, 0))
        self._container_var = tk.StringVar(value="mp4")
        ttk.Combobox(f4, textvariable=self._container_var,
                     values=["mp4", "mkv", "mov"], state="readonly", width=6).grid(
            row=1, column=3, padx=4, pady=(4, 0), sticky="w")

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=(0, 8))
        self._btn = ttk.Button(btn_frame, text="▶ 开始处理", command=self._start)
        self._btn.pack(side=tk.RIGHT)

        self._progress = ttk.Progressbar(self, mode="indeterminate")
        self._progress.pack(fill=tk.X, pady=(0, 8))

        self._log = scrolledtext.ScrolledText(self, height=10, state="normal", wrap="word")
        self._log.pack(fill=tk.BOTH, expand=True)

    def _update_engine_state(self):
        caps = self._caps

        self._sr_available = {}
        sr_items = []
        sr_sel = None

        for key in ("nvvfx", "dxva_vsr", "bicubic", "lanczos"):
            if key == "nvvfx":
                ok = caps["worker"] and caps["torch_cuda"] and caps["nvvfx"]
                label = "NVIDIA VFX SDK"
            elif key == "dxva_vsr":
                ok = caps["worker"] and caps["vsr_dll"]
                label = "DXVA VSR (NVIDIA RTX 视频增强)"
            elif key == "bicubic":
                ok = True; label = "双三次"
            else:
                ok = True; label = "Lanczos"
            if ok:
                self._sr_available[label] = key
                sr_items.append(label)
                if sr_sel is None:
                    sr_sel = label

        self._sr_combo["values"] = sr_items
        if sr_sel:
            self._sr_var.set(sr_sel)
        self._check_sr_limit()

        self._fi_available = {}
        fi_items = []
        fi_sel = None

        for key in ("dis", "torch_flow", "optical_flow", "rife", "blend", "none"):
            if key == "rife":
                ok = caps["torch_cuda"]
                label = "RIFE AI"
            elif key == "torch_flow":
                ok = caps["torch_cuda"]
                label = "GPU 光流 (SVP 风格)"
            elif key == "dis":
                try:
                    import cv2; cv2.DISOpticalFlow_create
                    ok = True; label = "DIS 光流 (SVP 风格)"
                except Exception:
                    ok = False; label = "DIS 光流"
            elif key == "optical_flow":
                ok = True; label = "光流法 (Farneback)"
            elif key == "blend":
                ok = True; label = "混合 (Blend)"
            else:
                ok = True; label = "不插帧"
            if ok:
                self._fi_available[label] = key
                fi_items.append(label)
                if fi_sel is None:
                    fi_sel = label

        self._fi_combo["values"] = fi_items
        if fi_sel:
            self._fi_var.set(fi_sel)

    def _on_sr_changed(self, event=None):
        self._check_sr_limit()

    def _check_sr_limit(self):
        label = self._sr_var.get()
        key = self._sr_available.get(label, "")
        if key != "dxva_vsr":
            self._sr_warning_var.set("")
            return

        try:
            scale = float(self._scale_var.get())
        except (ValueError, tk.TclError):
            scale = 2.0

        src_w = max(self._probe_input_width(), 1920)
        src_h = max(self._probe_input_height(), 1080)
        dst_w = int(src_w * scale)
        dst_h = int(src_h * scale)

        if dst_w > _VSR_MAX_W or dst_h > _VSR_MAX_H:
            self._sr_warning_var.set(
                f"⚠ DXVA VSR 最大支持 {_VSR_MAX_W}×{_VSR_MAX_H}，当前输出 {dst_w}×{dst_h} 可能失败。")
        else:
            self._sr_warning_var.set("")

    def _probe_input_width(self):
        path = self._input_var.get().strip()
        if not path or not os.path.isfile(path):
            return 0
        try:
            from .ffmpeg_bridge import FFmpegVideoDecoder
            dec = FFmpegVideoDecoder(path, use_nvdec=False)
            info = dec.probe()
            return info.get("width", 0)
        except Exception:
            return 0

    def _probe_input_height(self):
        path = self._input_var.get().strip()
        if not path or not os.path.isfile(path):
            return 0
        try:
            from .ffmpeg_bridge import FFmpegVideoDecoder
            dec = FFmpegVideoDecoder(path, use_nvdec=False)
            info = dec.probe()
            return info.get("height", 0)
        except Exception:
            return 0

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="选择输入视频",
            filetypes=[("视频文件", "*.mp4 *.mkv *.mov *.avi *.webm"), ("所有文件", "*.*")])
        if path:
            self._input_var.set(path)
            self._output_var.set("")

    def _on_input_changed(self, *args):
        path = self._input_var.get().strip()
        if path and path != self._last_input:
            self._last_input = path
            self._check_sr_limit()

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
        fi_key = self._fi_available.get(self._fi_var.get(), "none")
        if fi_key != "none":
            tags.append(f"f{self._fi_mult_var.get()}")
        suffix = "_" + "_".join(tags) if tags else ""
        return os.path.join(dirname, f"{base}{suffix}.{self._container_var.get()}")

    def _log_msg(self, msg: str):
        self._log.insert(tk.END, msg + "\n")
        self._log.see(tk.END)

    def _start(self):
        if self._running:
            return

        sr_label = self._sr_var.get()
        fi_label = self._fi_var.get()
        sr = self._sr_available.get(sr_label, "bicubic")
        fi = self._fi_available.get(fi_label, "none")

        inp = self._input_var.get().strip()
        out = self._output_var.get().strip()
        if not inp:
            messagebox.showerror("错误", "请选择输入文件")
            return

        if sr == "dxva_vsr":
            scale = self._scale_var.get()
            sw = self._probe_input_width()
            sh = self._probe_input_height()
            if sw and sh:
                dw = int(sw * scale)
                dh = int(sh * scale)
                if dw > _VSR_MAX_W or dh > _VSR_MAX_H:
                    ok = messagebox.askyesno(
                        "分辨率超限",
                        f"DXVA VSR 最大支持 {_VSR_MAX_W}×{_VSR_MAX_H}。\n\n"
                        f"当前输出 {dw}×{dh} 可能导致处理失败。\n"
                        f"是否继续？")
                    if not ok:
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
    av_sr = app._sr_available
    av_fi = app._fi_available
    _print(f"  可用超分引擎: {', '.join(list(av_sr.keys())[:3])}")
    _print(f"  可用插帧引擎: {', '.join(list(av_fi.keys())[:4])}")

    app.mainloop()


if __name__ == "__main__":
    main()

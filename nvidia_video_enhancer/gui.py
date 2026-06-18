#!/usr/bin/env python3
"""NVE GUI — 轻量图形界面，自动检测引擎可用性。"""

import os
import sys
import threading
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from .utils import check_engine_availability
from .pipeline import VideoEnhancer
from .config import ProcessConfig, EncodeConfig
from ._logging import get_logger, set_gui_handler

_log = get_logger(__name__)


def run_in_thread(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()


class _GuiLogHandler(logging.Handler):
    """将 logging 消息发送到 GUI 文本框。"""

    def __init__(self, gui_callback):
        super().__init__()
        self._callback = gui_callback
        self.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))

    def emit(self, record):
        try:
            msg = self.format(record)
            self._callback(msg)
        except Exception:
            pass


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
        self._torch_python = None
        self._torch_scanned = False

        # 阶段1: 轻量检测 (worker DLL / VSR bridge / ncnn 文件 — 毫秒级)
        caps_fast = check_engine_availability()
        caps_fast["torch_cuda"] = False
        caps_fast["nvvfx"] = False
        self._caps = caps_fast

        self._last_input = ""
        self._probe_cache = {}
        self._build_ui()
        self._update_engine_state()

        # 阶段2: 后台检测 torch + nvvfx (可能需要扫描系统 Python — 秒级)
        run_in_thread(self._detect_torch_python)

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

        self._sr_first_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f3, text="先超分再插帧", variable=self._sr_first_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

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
        torch_ok = caps["torch_cuda"]
        scanning = not self._torch_scanned

        self._sr_available = {}
        sr_items = []
        sr_sel = None

        for key in ("nvvfx", "dxva_vsr", "esrgan", "realcugan", "bicubic", "lanczos", "none"):
            if key == "nvvfx":
                if caps["worker"] and torch_ok and caps["nvvfx"]:
                    ok = True; label = "NVIDIA NGX VSR"
                elif scanning and caps["worker"]:
                    ok = True; label = "NVIDIA NGX VSR (检测中...)"
                else:
                    ok = False; label = ""
            elif key == "dxva_vsr":
                ok = caps["worker"] and caps["vsr_dll"]
                label = "DXVA VSR (NVIDIA RTX 视频增强)"
            elif key == "esrgan":
                ok = torch_ok
                label = "Real-ESRGAN (PyTorch)" if torch_ok else ("Real-ESRGAN (PyTorch) (检测中...)" if scanning else "")
            elif key == "realcugan":
                ok = self._ncnn_sr_available()
                label = "Real-CUGAN ncnn (AI)"
            elif key == "bicubic":
                ok = True; label = "双三次"
            elif key == "lanczos":
                ok = True; label = "Lanczos"
            else:
                ok = True; label = "不超分"
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

        for key in ("dis", "torch_flow", "optical_flow", "rife", "rife_ncnn", "blend", "none"):
            if key == "rife":
                if torch_ok:
                    ok = True; label = "RIFE AI (PyTorch)"
                elif scanning:
                    ok = True; label = "RIFE AI (PyTorch) (检测中...)"
                else:
                    ok = False; label = ""
            elif key == "rife_ncnn":
                ok = self._ncnn_fi_available()
                label = "RIFE ncnn-vulkan"
            elif key == "torch_flow":
                if torch_ok:
                    ok = True; label = "GPU 光流 (SVP 风格)"
                elif scanning:
                    ok = True; label = "GPU 光流 (SVP 风格) (检测中...)"
                else:
                    ok = False; label = ""
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

    def _ncnn_sr_available(self):
        from ._paths import pkg_file_exists
        return pkg_file_exists("ncnn", "realcugan", "realcugan-ncnn-vulkan.exe")

    def _ncnn_fi_available(self):
        from ._paths import pkg_file_exists
        return pkg_file_exists("ncnn", "rife", "rife-ncnn-vulkan.exe")

    def _detect_torch_python(self):
        try:
            from ._env import get_torch_python
            result = get_torch_python()
            if result:
                self._torch_python = result
            self._caps = check_engine_availability(self._torch_python)
        except Exception:
            pass
        finally:
            def _done():
                self._update_engine_state()
                self._torch_scanned = True
            self.after(0, _done)

    def _check_sr_limit(self):
        label = self._sr_var.get()
        key = self._sr_available.get(label, "")
        if key not in ("dxva_vsr", "none"):
            self._sr_warning_var.set("")
            return

        try:
            scale = float(self._scale_var.get())
        except (ValueError, tk.TclError):
            scale = 2.0

        if key == "none":
            self._sr_warning_var.set("")
            return

        info = self._probe_input_info()
        src_w = max(info.get("width", 0), 1920)
        src_h = max(info.get("height", 0), 1080)
        dst_w = int(src_w * scale)
        dst_h = int(src_h * scale)

        if dst_w > _VSR_MAX_W or dst_h > _VSR_MAX_H:
            self._sr_warning_var.set(
                f"⚠ DXVA VSR 最大支持 {_VSR_MAX_W}×{_VSR_MAX_H}，当前输出 {dst_w}×{dst_h} 可能失败。")
        else:
            self._sr_warning_var.set("")

    def _probe_input_info(self) -> dict:
        path = self._input_var.get().strip()
        if not path or not os.path.isfile(path):
            return {"width": 0, "height": 0}
        if path in self._probe_cache:
            return self._probe_cache[path]
        try:
            from .ffmpeg_bridge import FFmpegVideoDecoder
            dec = FFmpegVideoDecoder(path, use_nvdec=False)
            info = dec.probe()
            result = {"width": info.get("width", 0), "height": info.get("height", 0)}
        except Exception:
            result = {"width": 0, "height": 0}
        self._probe_cache[path] = result
        return result

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
            self._probe_cache.pop(path, None)
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
        sr_key = self._sr_available.get(self._sr_var.get(), "none")
        if sr_key != "none":
            s = self._scale_var.get()
            if s > 1.0:
                tags.append(f"x{s:.1f}".rstrip('0').rstrip('.'))
        fi_key = self._fi_available.get(self._fi_var.get(), "none")
        if fi_key != "none":
            tags.append(f"f{self._fi_mult_var.get()}")
        suffix = "_" + "_".join(tags) if tags else ""
        container = self._container_var.get()
        # 确保容器后缀不含非法字符
        safe_container = "".join(c for c in container if c.isalnum() or c in "._-")
        return os.path.join(dirname, f"{base}{suffix}.{safe_container}")

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

        if sr == "none" and fi == "none":
            messagebox.showerror("错误", "未选择任何可用的引擎！")
            return

        inp = self._input_var.get().strip()
        out = self._output_var.get().strip()
        if not inp:
            messagebox.showerror("错误", "请选择输入文件")
            return

        if sr == "nvvfx" or fi in ("rife", "torch_flow"):
            if not self._caps.get("torch_cuda"):
                messagebox.showerror(
                    "PyTorch 不可用",
                    "所选引擎需要 PyTorch + CUDA 环境。\n\n"
                    "请通过 conda 安装:\n"
                    "  conda install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia\n\n"
                    "或选择 ncnn 版本引擎。")
                return

        if sr == "dxva_vsr":
            scale = self._scale_var.get()
            info = self._probe_input_info()
            sw = info.get("width", 0)
            sh = info.get("height", 0)
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
                torch_python=self._torch_python,
                sr_first=self._sr_first_var.get(),
            )

            enhancer = VideoEnhancer(config)

            gui_cb = lambda msg: self.after(0, self._log_msg, msg)
            handler = _GuiLogHandler(gui_cb)
            set_gui_handler(handler)
            try:
                enhancer.run()
            finally:
                set_gui_handler(None)

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

    def _check_detection_done():
        if not app._torch_scanned:
            app.after(100, _check_detection_done)
        else:
            caps = app._caps
            _print("[信息] 引擎检测:")
            _print(f"  FFmpeg Worker: {'✓' if caps['worker'] else '✗'}")
            _print(f"  D3D11 Bridge:  {'✓' if caps['vsr_dll'] else '✗'}")
            _print(f"  nvidia-vfx:    {'✓' if caps['nvvfx'] else '✗'}")
            _print(f"  ncnn RIFE:     {'✓' if caps.get('ncnn_rife', False) else '✗'}")
            _print(f"  ncnn Real-CUG: {'✓' if caps.get('ncnn_cugan', False) else '✗'}")
            av_sr = [k for k in app._sr_available.keys() if "不超分" not in k]
            av_fi = [k for k in app._fi_available.keys() if "不插帧" not in k]
            _print(f"  可用超分引擎: {', '.join(av_sr)}")
            _print(f"  可用插帧引擎: {', '.join(av_fi)}")

    app.after(50, _check_detection_done)
    app.mainloop()


if __name__ == "__main__":
    main()

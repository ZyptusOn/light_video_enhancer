#!/usr/bin/env python3
"""Responsive Tk GUI for Light Video Enhancer (Windows 7 compatible)."""

import logging
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, Optional

from ._logging import set_gui_handler
from .capabilities import quick_capabilities
from .cli import _auto_output
from .config import EncodeConfig, ProcessConfig
from .encoding import CODEC_CHOICES
from .pipeline import ProcessingCancelled, VideoEnhancer


class _GuiLogHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))

    def emit(self, record) -> None:
        try:
            self._callback(self.format(record))
        except Exception:
            pass


class LVEGUI(tk.Tk):
    SR_LABELS = {
        "自动选择（推荐）": "auto",
        "D3D11 视频处理 / 驱动 VSR": "dxva_vsr",
        "NVIDIA Video Effects VSR（CUDA）": "nvvfx",
        "Real-CUGAN ncnn-vulkan": "realcugan",
        "Real-ESRGAN ncnn-vulkan": "realesrgan",
        "ESRGAN ncnn-vulkan（经典模型）": "esrgan",
        "Lanczos 高质量缩放": "lanczos",
        "双三次缩放": "bicubic",
        "不改变分辨率": "none",
    }
    FI_LABELS = {
        "自动选择（推荐）": "auto",
        "RIFE AI（PyTorch，按需检测）": "rife",
        "RIFE ncnn-vulkan（便携批处理）": "rife_ncnn",
        "DIS 稠密光流": "dis",
        "Farneback 光流": "optical_flow",
        "CUDA 块匹配光流": "torch_flow",
        "快速帧混合": "blend",
        "不插帧": "none",
    }
    CODECS = CODEC_CHOICES

    def __init__(self):
        super().__init__()
        self.title("Light Video Enhancer 0.5.2")
        self.geometry("900x760")
        self.minsize(780, 650)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._style = ttk.Style(self)
        if os.name == "nt" and "vista" in self._style.theme_names():
            self._style.theme_use("vista")

        self._caps = quick_capabilities()
        self._running = False
        self._enhancer: Optional[VideoEnhancer] = None
        self._torch_python: Optional[str] = None
        self._probe_generation = 0
        self._last_output = ""

        self._build_ui()
        self._refresh_engine_lists()
        self._refresh_capability_view()
        self._set_status("就绪；依赖检测已使用快速模式", 0)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 4))
        process_tab = ttk.Frame(notebook, padding=10)
        environment_tab = ttk.Frame(notebook, padding=10)
        notebook.add(process_tab, text="处理")
        notebook.add(environment_tab, text="环境与后端")
        self._build_process_tab(process_tab)
        self._build_environment_tab(environment_tab)

        status_frame = ttk.Frame(self, padding=(10, 4, 10, 10))
        status_frame.grid(row=1, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        self._progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
        self._progress.grid(row=0, column=0, sticky="ew")
        self._status_var = tk.StringVar()
        ttk.Label(status_frame, textvariable=self._status_var).grid(row=1, column=0, sticky="w", pady=(3, 0))

    def _build_process_tab(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(4, weight=1)

        files = ttk.LabelFrame(parent, text="输入与输出", padding=8)
        files.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        files.columnconfigure(1, weight=1)
        self._input_var = tk.StringVar()
        self._output_var = tk.StringVar()
        ttk.Label(files, text="输入").grid(row=0, column=0, sticky="w")
        ttk.Entry(files, textvariable=self._input_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(files, text="浏览...", command=self._browse_input).grid(row=0, column=2)
        ttk.Label(files, text="输出").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(files, textvariable=self._output_var).grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(files, text="浏览...", command=self._browse_output).grid(row=1, column=2, pady=(6, 0))
        self._media_var = tk.StringVar(value="请选择视频")
        ttk.Label(files, textvariable=self._media_var, foreground="#555555").grid(
            row=2, column=1, columnspan=2, sticky="w", padx=6, pady=(5, 0))

        pipeline = ttk.LabelFrame(parent, text="增强管线", padding=8)
        pipeline.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for column in (1, 3, 5):
            pipeline.columnconfigure(column, weight=1)
        self._sr_var = tk.StringVar(value="自动选择（推荐）")
        self._fi_var = tk.StringVar(value="自动选择（推荐）")
        self._scale_var = tk.DoubleVar(value=2.0)
        self._multiplier_var = tk.IntVar(value=2)
        self._sr_quality_var = tk.StringVar(value="quality")
        self._fi_quality_var = tk.StringVar(value="balanced")
        self._sr_first_var = tk.BooleanVar(value=False)
        ttk.Label(pipeline, text="超分").grid(row=0, column=0, sticky="w")
        self._sr_combo = ttk.Combobox(pipeline, textvariable=self._sr_var, state="readonly", width=28)
        self._sr_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(5, 16))
        ttk.Label(pipeline, text="倍率").grid(row=0, column=3, sticky="e")
        ttk.Spinbox(pipeline, textvariable=self._scale_var, from_=1.0, to=4.0,
                    increment=0.25, width=7).grid(row=0, column=4, sticky="w", padx=5)
        self._sr_quality_label = ttk.Label(pipeline, text="超分质量")
        self._sr_quality_label.grid(row=0, column=5, sticky="e")
        self._sr_quality_combo = ttk.Combobox(
            pipeline, textvariable=self._sr_quality_var, state="readonly", width=10,
            values=["fast", "balanced", "quality", "ultra"])
        self._sr_quality_combo.grid(row=0, column=6, sticky="w", padx=5)
        ttk.Label(pipeline, text="插帧").grid(row=1, column=0, sticky="w", pady=(7, 0))
        self._fi_combo = ttk.Combobox(pipeline, textvariable=self._fi_var, state="readonly", width=28)
        self._fi_combo.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(5, 16), pady=(7, 0))
        ttk.Label(pipeline, text="倍率").grid(row=1, column=3, sticky="e", pady=(7, 0))
        ttk.Spinbox(pipeline, textvariable=self._multiplier_var, from_=2, to=4,
                    width=7).grid(row=1, column=4, sticky="w", padx=5, pady=(7, 0))
        self._fi_quality_label = ttk.Label(pipeline, text="插帧质量")
        self._fi_quality_label.grid(row=1, column=5, sticky="e", pady=(7, 0))
        self._fi_quality_combo = ttk.Combobox(
            pipeline, textvariable=self._fi_quality_var, state="readonly", width=10,
            values=["ultra", "fast", "balanced", "quality"])
        self._fi_quality_combo.grid(row=1, column=6, sticky="w", padx=5, pady=(7, 0))
        self._sr_combo.bind("<<ComboboxSelected>>", self._quality_controls_changed)
        self._fi_combo.bind("<<ComboboxSelected>>", self._quality_controls_changed)
        ttk.Checkbutton(pipeline, text="先超分再插帧（通常更慢）",
                        variable=self._sr_first_var).grid(row=2, column=1, columnspan=3,
                                                         sticky="w", pady=(7, 0))

        options = ttk.LabelFrame(parent, text="编码、设备与片段", padding=8)
        options.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self._codec_var = tk.StringVar(value="auto")
        self._preset_var = tk.StringVar(value="balanced")
        self._crf_var = tk.IntVar(value=23)
        self._container_var = tk.StringVar(value="mp4")
        self._audio_var = tk.BooleanVar(value=True)
        self._overwrite_var = tk.BooleanVar(value=False)
        self._start_var = tk.StringVar()
        self._duration_var = tk.StringVar()
        self._ncnn_var = tk.StringVar(value="自动")
        ttk.Label(options, text="编码器").grid(row=0, column=0, sticky="w")
        codec_combo = ttk.Combobox(options, textvariable=self._codec_var, state="readonly",
                                   values=self.CODECS, width=14)
        codec_combo.grid(row=0, column=1, sticky="w", padx=5)
        codec_combo.bind("<<ComboboxSelected>>", self._codec_changed)
        ttk.Label(options, text="Preset").grid(row=0, column=2, sticky="e", padx=(14, 0))
        ttk.Entry(options, textvariable=self._preset_var, width=10).grid(row=0, column=3, sticky="w", padx=5)
        ttk.Label(options, text="CQ/CRF").grid(row=0, column=4, sticky="e", padx=(14, 0))
        ttk.Spinbox(options, textvariable=self._crf_var, from_=0, to=63, width=6).grid(
            row=0, column=5, sticky="w", padx=5)
        ttk.Label(options, text="容器").grid(row=0, column=6, sticky="e", padx=(14, 0))
        ttk.Combobox(options, textvariable=self._container_var, state="readonly", width=6,
                     values=["mp4", "mkv", "mov"]).grid(row=0, column=7, sticky="w", padx=5)
        ttk.Label(options, text="NCNN 设备").grid(row=1, column=0, sticky="w", pady=(7, 0))
        values = ["自动", "CPU"] + ["GPU %d" % index for index, _ in enumerate(self._caps.get("gpus", []))]
        ttk.Combobox(options, textvariable=self._ncnn_var, state="readonly", width=14,
                     values=values).grid(row=1, column=1, sticky="w", padx=5, pady=(7, 0))
        ttk.Label(options, text="开始（秒）").grid(row=1, column=2, sticky="e", padx=(14, 0), pady=(7, 0))
        ttk.Entry(options, textvariable=self._start_var, width=10).grid(row=1, column=3, sticky="w", padx=5, pady=(7, 0))
        ttk.Label(options, text="时长（秒）").grid(row=1, column=4, sticky="e", padx=(14, 0), pady=(7, 0))
        ttk.Entry(options, textvariable=self._duration_var, width=10).grid(row=1, column=5, sticky="w", padx=5, pady=(7, 0))
        ttk.Checkbutton(options, text="复制音频", variable=self._audio_var).grid(
            row=1, column=6, sticky="w", pady=(7, 0))
        ttk.Checkbutton(options, text="覆盖输出", variable=self._overwrite_var).grid(
            row=1, column=7, sticky="w", pady=(7, 0))

        actions = ttk.Frame(parent)
        actions.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self._open_button = ttk.Button(actions, text="打开输出文件夹", command=self._open_output,
                                       state="disabled")
        self._open_button.pack(side="left")
        self._cancel_button = ttk.Button(actions, text="取消", command=self._cancel, state="disabled")
        self._cancel_button.pack(side="right", padx=(6, 0))
        self._start_button = ttk.Button(actions, text="开始处理", command=self._start)
        self._start_button.pack(side="right")

        logs = ttk.LabelFrame(parent, text="日志", padding=5)
        logs.grid(row=4, column=0, sticky="nsew")
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(0, weight=1)
        self._log = scrolledtext.ScrolledText(logs, height=10, wrap="word", state="normal")
        self._log.grid(row=0, column=0, sticky="nsew")

    def _build_environment_tab(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        controls = ttk.Frame(parent)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(controls, text="刷新快速检测", command=self._refresh_caps).pack(side="left")
        self._scan_button = ttk.Button(controls, text="扫描 PyTorch / CUDA 环境",
                                       command=self._scan_environments)
        self._scan_button.pack(side="left", padx=6)
        ttk.Label(controls, text="扫描只在手动触发或使用 RIFE 时执行。",
                  foreground="#555555").pack(side="left", padx=8)
        paned = ttk.Panedwindow(parent, orient="vertical")
        paned.grid(row=1, column=0, sticky="nsew")
        capability_frame = ttk.LabelFrame(paned, text="硬件与内置后端", padding=6)
        env_frame = ttk.LabelFrame(paned, text="Python 环境", padding=6)
        paned.add(capability_frame, weight=1)
        paned.add(env_frame, weight=1)
        self._cap_tree = ttk.Treeview(capability_frame, columns=("status", "detail"), show="tree headings")
        self._cap_tree.heading("#0", text="项目")
        self._cap_tree.heading("status", text="状态")
        self._cap_tree.heading("detail", text="说明")
        self._cap_tree.column("#0", width=230)
        self._cap_tree.column("status", width=80, anchor="center")
        self._cap_tree.column("detail", width=450)
        self._cap_tree.pack(fill="both", expand=True)
        self._env_tree = ttk.Treeview(env_frame, columns=("version", "torch", "cuda", "nvvfx"),
                                      show="tree headings")
        self._env_tree.heading("#0", text="python.exe")
        for key, title in (("version", "Python"), ("torch", "PyTorch"),
                           ("cuda", "CUDA / GPU"), ("nvvfx", "NV-VFX")):
            self._env_tree.heading(key, text=title)
        self._env_tree.column("#0", width=360)
        self._env_tree.column("version", width=80)
        self._env_tree.column("torch", width=110)
        self._env_tree.column("cuda", width=220)
        self._env_tree.column("nvvfx", width=70, anchor="center")
        self._env_tree.pack(fill="both", expand=True)

    def _refresh_engine_lists(self) -> None:
        caps = self._caps
        sr = ["自动选择（推荐）"]
        if caps.get("vsr_dll"):
            sr.append("D3D11 视频处理 / 驱动 VSR")
        if caps.get("nvvfx_current") or self._torch_python:
            sr.append("NVIDIA Video Effects VSR（CUDA）")
        if caps.get("ncnn_cugan"):
            sr.append("Real-CUGAN ncnn-vulkan")
        if caps.get("ncnn_esrgan"):
            sr.append("Real-ESRGAN ncnn-vulkan")
        if caps.get("ncnn_classic_esrgan"):
            sr.append("ESRGAN ncnn-vulkan（经典模型）")
        sr.extend(["Lanczos 高质量缩放", "双三次缩放", "不改变分辨率"])
        fi = ["自动选择（推荐）"]
        if caps.get("rife_model"):
            fi.append("RIFE AI（PyTorch，按需检测）")
        if caps.get("ncnn_rife"):
            fi.append("RIFE ncnn-vulkan（便携批处理）")
        try:
            import cv2
            if hasattr(cv2, "DISOpticalFlow_create"):
                fi.append("DIS 稠密光流")
        except ImportError:
            pass
        fi.extend(["Farneback 光流"])
        if caps.get("torch_cuda") and not self._torch_python:
            fi.append("CUDA 块匹配光流")
        fi.extend(["快速帧混合", "不插帧"])
        old_sr, old_fi = self._sr_var.get(), self._fi_var.get()
        self._sr_combo["values"] = sr
        self._fi_combo["values"] = fi
        self._sr_var.set(old_sr if old_sr in sr else sr[0])
        self._fi_var.set(old_fi if old_fi in fi else fi[0])
        self._quality_controls_changed()

    def _refresh_capability_view(self) -> None:
        tree = self._cap_tree
        for item in tree.get_children():
            tree.delete(item)
        for gpu in self._caps.get("gpus", []):
            tree.insert("", "end", text="GPU", values=("OK", "%s [%s]" % (gpu.name, gpu.vendor.upper())))
        rows = [
            ("FFmpeg Worker", "worker", "内嵌解码、编码与音频复制"),
            ("D3D11 VSR Bridge", "vsr_dll", "NVIDIA / Intel 驱动增强；AMD 为视频处理缩放"),
            ("RIFE 模型", "rife_model", "PyTorch 插帧权重"),
            ("RIFE ncnn", "ncnn_rife", "跨厂商 Vulkan，分块批处理"),
            ("Real-CUGAN ncnn", "ncnn_cugan", "跨厂商 Vulkan 超分"),
            ("Real-ESRGAN ncnn", "ncnn_esrgan", "AnimeVideo-v3 / x4plus"),
            ("ESRGAN classic ncnn", "ncnn_classic_esrgan", "原始 ESRGAN x4 感知模型"),
        ]
        for label, key, detail in rows:
            tree.insert("", "end", text=label,
                        values=("可用" if self._caps.get(key) else "不可用", detail))
        encoders = self._caps.get("encoders", ())
        tree.insert("", "end", text="内置编码器",
                    values=("可用" if encoders else "不可用",
                            ", ".join(encoders) if encoders else "Worker 未加载"))

    def _refresh_caps(self) -> None:
        self._caps = quick_capabilities()
        self._refresh_engine_lists()
        self._refresh_capability_view()
        self._set_status("快速检测已刷新", 0)

    def _scan_environments(self) -> None:
        if self._running:
            return
        self._scan_button.configure(state="disabled")
        self._set_status("正在并行检测 Python / PyTorch 环境...", 0)

        def work():
            from ._env import get_all_python_envs
            result = get_all_python_envs(force_rescan=True)
            self.after(0, self._scan_done, result)

        threading.Thread(target=work, name="lve-env-scan", daemon=True).start()

    def _scan_done(self, environments) -> None:
        for item in self._env_tree.get_children():
            self._env_tree.delete(item)
        self._torch_python = None
        current = os.path.normcase(os.path.abspath(os.sys.executable))
        for info in environments:
            exe = info.get("exe", "")
            if info.get("cuda") and self._torch_python is None and os.path.normcase(exe) != current:
                self._torch_python = exe
            self._env_tree.insert("", "end", text=exe, values=(
                info.get("version", "?"), info.get("torch_version") or "--",
                info.get("gpu_name") if info.get("cuda") else "--",
                "是" if info.get("nvvfx") else "否"))
        current_cuda = any(info.get("cuda") and os.path.normcase(info.get("exe", "")) == current
                           for info in environments)
        self._caps["torch_cuda"] = current_cuda or bool(self._torch_python)
        self._refresh_engine_lists()
        self._scan_button.configure(state="normal")
        self._set_status("环境扫描完成：%d 个 Python，%d 个 CUDA PyTorch" % (
            len(environments), sum(bool(item.get("cuda")) for item in environments)), 0)

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="选择输入视频", filetypes=[
                ("视频文件", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts"), ("所有文件", "*.*")])
        if path:
            self._input_var.set(path)
            self._output_var.set("")
            self._probe_input(path)

    def _probe_input(self, path: str) -> None:
        self._probe_generation += 1
        generation = self._probe_generation
        self._media_var.set("正在读取视频信息...")

        def work():
            try:
                from .ffmpeg_bridge import FFmpegVideoDecoder
                info = FFmpegVideoDecoder(path, use_nvdec=False).probe()
                text = "%dx%d  |  %.3f fps  |  %s 帧" % (
                    info["width"], info["height"], info["fps"], info["total_frames"] or "未知")
            except Exception as exc:
                text = "无法读取信息：%s" % exc
            self.after(0, lambda: self._media_var.set(text) if generation == self._probe_generation else None)

        threading.Thread(target=work, name="lve-probe", daemon=True).start()

    def _browse_output(self) -> None:
        ext = "." + self._container_var.get()
        path = filedialog.asksaveasfilename(
            title="选择输出路径", defaultextension=ext,
            filetypes=[("MP4", "*.mp4"), ("MKV", "*.mkv"), ("MOV", "*.mov"), ("所有文件", "*.*")])
        if path:
            self._output_var.set(path)

    def _quality_controls_changed(self, _event=None) -> None:
        sr = self.SR_LABELS.get(self._sr_var.get(), "auto")
        fi = self.FI_LABELS.get(self._fi_var.get(), "auto")
        sr_values = {
            "nvvfx": ("fast", "balanced", "quality", "ultra"),
            "realcugan": ("fast", "balanced", "quality", "ultra"),
            "realesrgan": ("fast", "balanced", "quality", "ultra"),
            "esrgan": ("quality", "ultra"),
        }.get(sr)
        if sr_values:
            self._sr_quality_combo.configure(state="readonly", values=sr_values)
            self._sr_quality_label.configure(state="normal")
            if self._sr_quality_var.get() not in sr_values:
                self._sr_quality_var.set("quality" if "quality" in sr_values else sr_values[0])
        else:
            self._sr_quality_combo.configure(state="disabled")
            self._sr_quality_label.configure(state="disabled")
        fi_values = ("ultra", "fast", "balanced", "quality") if fi in {
            "dis", "optical_flow", "torch_flow"} else None
        if fi_values:
            self._fi_quality_combo.configure(state="readonly", values=fi_values)
            self._fi_quality_label.configure(state="normal")
        else:
            self._fi_quality_combo.configure(state="disabled")
            self._fi_quality_label.configure(state="disabled")

    def _codec_changed(self, _event=None) -> None:
        codec = self._codec_var.get()
        if "nvenc" in codec:
            self._preset_var.set("p5")
        elif codec == "auto":
            self._preset_var.set("balanced")
        else:
            self._preset_var.set("medium")

    @staticmethod
    def _optional_float(value: str, label: str) -> Optional[float]:
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("%s 必须是数字" % label) from exc

    def _build_config(self) -> ProcessConfig:
        input_path = self._input_var.get().strip()
        if not os.path.isfile(input_path):
            raise FileNotFoundError("请选择有效的输入视频")
        sr = self.SR_LABELS[self._sr_var.get()]
        fi = self.FI_LABELS[self._fi_var.get()]
        output = self._output_var.get().strip()
        if not output:
            output = _auto_output(input_path, self._scale_var.get(), fi,
                                  self._multiplier_var.get(), self._container_var.get(), sr)
            self._output_var.set(output)
        ncnn_text = self._ncnn_var.get()
        ncnn_gpu = None if ncnn_text == "自动" else (-1 if ncnn_text == "CPU" else int(ncnn_text.split()[-1]))
        encode = EncodeConfig(
            codec=self._codec_var.get(), preset=self._preset_var.get().strip(),
            crf=self._crf_var.get(), container=self._container_var.get(),
            copy_audio=self._audio_var.get(), overwrite=self._overwrite_var.get())
        config = ProcessConfig(
            input_path=input_path, output_path=output, scale=self._scale_var.get(),
            sr_engine=sr, fi_engine=fi, fi_multiplier=self._multiplier_var.get(),
            sr_quality=self._sr_quality_var.get(),
            fi_quality=self._fi_quality_var.get(), encode=encode,
            start_time=self._optional_float(self._start_var.get(), "开始时间"),
            duration=self._optional_float(self._duration_var.get(), "时长"),
            device="auto", torch_python=self._torch_python,
            sr_first=self._sr_first_var.get(), ncnn_gpu=ncnn_gpu)
        config.validate()
        return config

    def _start(self) -> None:
        if self._running:
            return
        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("无法开始", str(exc), parent=self)
            return
        self._running = True
        self._log.delete("1.0", "end")
        self._start_button.configure(state="disabled")
        self._cancel_button.configure(state="normal")
        self._open_button.configure(state="disabled")
        self._progress.configure(value=0, maximum=100)
        self._set_status("正在初始化处理管线...", 0)
        handler = _GuiLogHandler(lambda message: self.after(0, self._log_message, message))
        set_gui_handler(handler)
        self._enhancer = VideoEnhancer(config, progress_callback=self._worker_progress)

        def work():
            try:
                output = self._enhancer.run()
                self.after(0, self._done, True, output)
            except ProcessingCancelled:
                self.after(0, self._done, False, "已取消")
            except Exception as exc:
                self.after(0, self._done, False, str(exc))
            finally:
                set_gui_handler(None)

        threading.Thread(target=work, name="lve-pipeline", daemon=True).start()

    def _worker_progress(self, stage: str, current: int, total: int) -> None:
        self.after(0, self._set_progress, stage, current, total)

    def _set_progress(self, stage: str, current: int, total: int) -> None:
        if total > 0:
            self._progress.configure(maximum=total, value=min(current, total), mode="determinate")
            self._status_var.set("%s：%d / %d（%.1f%%）" % (stage, current, total, current * 100.0 / total))
        else:
            self._status_var.set("%s：已处理 %d 帧" % (stage, current))

    def _cancel(self) -> None:
        if self._enhancer is not None:
            self._enhancer.cancel()
            self._cancel_button.configure(state="disabled")
            self._set_status("正在安全停止并封装已编码数据...", self._progress["value"])

    def _done(self, success: bool, value: str) -> None:
        self._running = False
        self._enhancer = None
        self._start_button.configure(state="normal")
        self._cancel_button.configure(state="disabled")
        if success:
            self._last_output = value
            self._open_button.configure(state="normal")
            self._progress.configure(value=self._progress["maximum"])
            self._set_status("处理完成：%s" % value, self._progress["value"])
            self._log_message("\n处理完成：%s" % value)
        else:
            self._set_status(value, self._progress["value"])
            self._log_message("\n处理未完成：%s" % value)
            if value != "已取消":
                messagebox.showerror("处理失败", value, parent=self)

    def _open_output(self) -> None:
        if self._last_output and os.path.isfile(self._last_output):
            try:
                os.startfile(os.path.dirname(self._last_output))
            except OSError as exc:
                messagebox.showerror("无法打开", str(exc), parent=self)

    def _log_message(self, message: str) -> None:
        self._log.insert("end", message + "\n")
        self._log.see("end")

    def _set_status(self, text: str, progress) -> None:
        self._status_var.set(text)
        if progress is not None:
            self._progress.configure(value=progress)

    def _close(self) -> None:
        if self._running:
            if not messagebox.askyesno("退出", "处理仍在进行。取消任务并退出？", parent=self):
                return
            self._cancel()
        self.destroy()


def main() -> None:
    LVEGUI().mainloop()


if __name__ == "__main__":
    main()

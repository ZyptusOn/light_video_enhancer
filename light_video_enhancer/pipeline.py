"""Chunked video enhancement pipeline with atomic output and cancellation."""

import copy
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from ._image_batch import read_frames, write_frames
from ._ncnn_directory_stream import NcnnDirectoryStream
from ._logging import get_logger
from .capabilities import choose_codec, quick_capabilities, select_engines
from .config import ProcessConfig
from .executor import FrameBatchExecutor
from .fi import FrameInterpolationEngine, create_fi_engine
from .i18n import is_chinese, tr
from .sr import SuperResolutionEngine, create_sr_engine

_log = get_logger(__name__)


class ProcessingCancelled(RuntimeError):
    pass


ProgressCallback = Callable[[str, int, int], None]


def _find_program(name: str) -> Optional[str]:
    package_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(package_dir), "ffmpeg", "build", "bin", name + ".exe"),
        os.path.join(os.path.dirname(os.path.dirname(package_dir)), "ffmpeg", "build", "bin", name + ".exe"),
        shutil.which(name),
    ]
    return next((os.path.normpath(path) for path in candidates if path and os.path.isfile(path)), None)


def _parse_rate(value: str) -> float:
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den) if float(den) else 0.0
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _normalise_preset(codec: str, preset: str) -> str:
    value = (preset or "balanced").lower()
    if "nvenc" in codec:
        return value if value in {"p1", "p2", "p3", "p4", "p5", "p6", "p7"} else "p5"
    if "amf" in codec:
        return value if value in {"speed", "balanced", "quality"} else "balanced"
    if "qsv" in codec:
        return value if value in {"veryfast", "faster", "fast", "medium", "slow"} else "medium"
    if codec in {"libaom-av1", "libsvtav1"}:
        if value.isdigit():
            return value
        return value if value in {
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "balanced", "medium", "slow", "slower", "veryslow", "quality",
        } else "balanced"
    if codec in {"libx264", "libx265"}:
        value = {"balanced": "medium", "quality": "slow"}.get(value, value)
    return value if value in {
        "ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
        "slow", "slower", "veryslow",
    } else "medium"


class _AsyncEncoder:
    def __init__(self, encoder, queue_size: int = 4):
        self._encoder = encoder
        self._queue = queue.Queue(maxsize=max(4, queue_size))
        self._error = None
        self.count = 0
        self._thread = threading.Thread(target=self._run, name="lve-encoder", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                if getattr(frame, "is_yuv420", False):
                    self._encoder.encode_yuv(frame.data, frame.width, frame.height)
                else:
                    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).ravel()
                    self._encoder.encode_yuv(yuv, frame.shape[1], frame.shape[0])
                self.count += 1
        except BaseException as exc:
            self._error = exc

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("编码线程失败: %s" % self._error) from self._error

    def put(self, frame: np.ndarray) -> None:
        while True:
            self._raise_if_failed()
            try:
                queued = frame if getattr(frame, "is_yuv420", False) else np.ascontiguousarray(frame)
                self._queue.put(queued, timeout=0.25)
                return
            except queue.Full:
                continue

    def finish(self) -> int:
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(None, timeout=0.25)
                break
            except queue.Full:
                continue
        self._thread.join()
        self._raise_if_failed()
        return self.count


class _AsyncDirectoryCleaner:
    def __init__(self):
        self._queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(
            target=self._run, name="lve-temp-cleaner", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            path = self._queue.get()
            if path is None:
                break
            try:
                shutil.rmtree(path)
            except OSError:
                _log.warning("无法清理 NCNN 临时目录: %s", path, exc_info=True)

    def submit(self, path: str) -> None:
        self._queue.put(path)

    def finish(self) -> None:
        self._queue.put(None)
        self._thread.join()


class _FrameRateResampler:
    def __init__(self, natural_fps: float, target_fps: float):
        self._natural_fps = natural_fps
        self._target_fps = target_fps
        self._input_index = 0
        self._next_output_time = 0.0

    def feed(self, frame: np.ndarray) -> Iterator[np.ndarray]:
        if abs(self._natural_fps - self._target_fps) < 1e-6:
            self._input_index += 1
            yield frame
            return
        current_time = self._input_index / self._natural_fps
        self._input_index += 1
        epsilon = 0.5 / max(self._natural_fps, self._target_fps)
        while self._next_output_time <= current_time + epsilon:
            yield frame
            self._next_output_time += 1.0 / self._target_fps


class VideoEnhancer:
    def __init__(self, config: ProcessConfig,
                 progress_callback: Optional[ProgressCallback] = None,
                 cancel_event: Optional[threading.Event] = None):
        self._config = copy.deepcopy(config)
        self._progress_callback = progress_callback
        self._cancel_event = cancel_event or threading.Event()
        self._sr_engine: Optional[SuperResolutionEngine] = None
        self._fi_engine: Optional[FrameInterpolationEngine] = None
        self._batch_executor: Optional[FrameBatchExecutor] = None
        self._src_width = self._src_height = 0
        self._src_fps = 0.0
        self._total_frames = 0
        self._selected_frames = 0
        self._dst_width = self._dst_height = 0
        self._partial_path = ""
        self._temp_cleaner = None

    def cancel(self) -> None:
        self._cancel_event.set()

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise ProcessingCancelled("用户已取消处理")

    def _progress(self, stage: str, current: int, total: int) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(stage, current, total)
            except Exception:
                _log.debug("进度回调失败", exc_info=True)

    def run(self) -> str:
        cfg = self._config
        cfg.validate()
        cfg.input_path = os.path.abspath(cfg.input_path)
        cfg.output_path = os.path.abspath(cfg.output_path)
        if os.path.normcase(cfg.input_path) == os.path.normcase(cfg.output_path):
            raise ValueError("输出文件不能覆盖输入文件")
        if os.path.exists(cfg.output_path) and not cfg.encode.overwrite:
            raise FileExistsError("输出文件已存在；请更换路径或启用覆盖: %s" % cfg.output_path)
        output_dir = os.path.dirname(cfg.output_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        requested_sr = cfg.sr_engine
        requested_fi = cfg.fi_engine
        requested_codec = cfg.encode.codec
        if requested_sr == "osdenhancer":
            if requested_fi not in {"auto", "none"}:
                raise ValueError("OSDEnhancer already performs 2x interpolation; select fi-engine none")
            cfg.fi_engine = "none"
        if requested_sr == "sparkvsr":
            # Reference indices describe original source frames.  Keeping SR
            # first preserves that mapping when a separate FI stage is used.
            cfg.sr_first = True
        caps = quick_capabilities()
        cfg.encode.codec = choose_codec(
            cfg.encode.codec, list(caps.get("gpus", ())),
            caps.get("encoders", ()))
        cfg.encode.preset = _normalise_preset(cfg.encode.codec, cfg.encode.preset)
        if requested_codec == "auto":
            _log.info(tr(
                "自动选择编码器: %s（已通过后端可用性检测）",
                "Auto-selected encoder: %s (verified by the backend)"),
                cfg.encode.codec)
        self._partial_path = self._make_partial_path(cfg.output_path)
        self._probe_input()
        target_width, target_height = self._requested_output_geometry()
        selection = select_engines(
            cfg.sr_engine, cfg.fi_engine,
            source_width=self._src_width, source_height=self._src_height,
            target_width=target_width, target_height=target_height,
            source_fps=self._src_fps, target_fps=cfg.fps,
            sr_quality=cfg.sr_quality, fi_quality=cfg.fi_quality,
            fi_multiplier=cfg.fi_multiplier, sr_first=cfg.sr_first,
            device=cfg.device, ncnn_gpu=cfg.ncnn_gpu,
            torch_python=cfg.torch_python,
            capabilities=caps)
        cfg.sr_engine, cfg.fi_engine = (
            selection.sr_engine, selection.fi_engine)
        if requested_sr == "auto":
            _log.info(tr(
                "自动选择超分: %s（评分 %d；%s）",
                "Auto-selected SR: %s (score %d; %s)"),
                cfg.sr_engine, selection.sr_score,
                selection.sr_reason_zh if is_chinese()
                else selection.sr_reason_en)
        if requested_fi == "auto":
            _log.info(tr(
                "自动选择插帧: %s（评分 %d；%s）",
                "Auto-selected interpolation: %s (score %d; %s)"),
                cfg.fi_engine, selection.fi_score,
                selection.fi_reason_zh if is_chinese()
                else selection.fi_reason_en)
        self._calculate_geometry()
        self._init_engines()
        success = False
        try:
            self._process_embedded()
            self._check_cancelled()
            if not os.path.isfile(self._partial_path) or os.path.getsize(self._partial_path) == 0:
                raise RuntimeError("编码器未生成有效输出")
            os.replace(self._partial_path, cfg.output_path)
            success = True
            _log.info("完成: %s (%.1f MB)", cfg.output_path,
                      os.path.getsize(cfg.output_path) / (1024 * 1024))
            return cfg.output_path
        finally:
            self._release_engines()
            if not success and self._partial_path and os.path.exists(self._partial_path):
                if cfg.keep_partial:
                    _log.warning("保留未完成输出: %s", self._partial_path)
                else:
                    try:
                        os.remove(self._partial_path)
                    except OSError:
                        pass

    @staticmethod
    def _make_partial_path(output_path: str) -> str:
        directory, filename = os.path.split(output_path)
        stem, ext = os.path.splitext(filename)
        return os.path.join(directory, ".%s.lve-%d.partial%s" % (stem, os.getpid(), ext or ".mp4"))

    def _probe_input(self) -> None:
        cfg = self._config
        if not os.path.isfile(cfg.input_path):
            raise FileNotFoundError("输入文件不存在: %s" % cfg.input_path)
        try:
            from .ffmpeg_bridge import FFmpegVideoDecoder
            info = FFmpegVideoDecoder(cfg.input_path, use_nvdec=False).probe()
        except Exception as worker_error:
            probe = _find_program("ffprobe")
            if not probe:
                raise RuntimeError("内嵌 FFmpeg 无法探测视频，且未找到 ffprobe: %s" % worker_error) from worker_error
            command = [probe, "-v", "error", "-print_format", "json",
                       "-show_format", "-show_streams", cfg.input_path]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
            if result.returncode:
                raise RuntimeError("ffprobe 失败: %s" % result.stderr.strip())
            data = json.loads(result.stdout)
            stream = next((item for item in data.get("streams", [])
                           if item.get("codec_type") == "video"), None)
            if stream is None:
                raise RuntimeError("输入文件中没有视频流")
            fps = _parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0")
            frames = int(stream.get("nb_frames") or 0)
            if not frames:
                frames = int(float(data.get("format", {}).get("duration") or 0) * fps + 0.5)
            info = {"width": int(stream["width"]), "height": int(stream["height"]),
                    "fps": fps, "total_frames": frames}

        self._src_width = int(info["width"])
        self._src_height = int(info["height"])
        self._src_fps = float(info["fps"])
        self._total_frames = int(info["total_frames"])
        if self._src_width <= 0 or self._src_height <= 0 or self._src_fps <= 0:
            raise RuntimeError("无效的视频参数: %s" % info)
        start = int(round((cfg.start_time or 0.0) * self._src_fps))
        available = max(0, self._total_frames - start) if self._total_frames else 0
        requested = int(round(cfg.duration * self._src_fps)) if cfg.duration else available
        self._selected_frames = min(available, requested) if available else requested
        _log.info("输入: %dx%d @ %.3f fps, %s 帧", self._src_width, self._src_height,
                  self._src_fps, self._selected_frames or "未知")

    def _requested_output_geometry(self) -> Tuple[int, int]:
        cfg = self._config
        if cfg.width and cfg.height:
            width, height = cfg.width, cfg.height
        elif cfg.width:
            width = cfg.width
            height = int(round(self._src_height * width / self._src_width))
        elif cfg.height:
            height = cfg.height
            width = int(round(self._src_width * height / self._src_height))
        else:
            width = int(round(self._src_width * cfg.scale))
            height = int(round(self._src_height * cfg.scale))
        return max(2, width - width % 2), max(2, height - height % 2)

    def _calculate_geometry(self) -> None:
        cfg = self._config
        if cfg.sr_engine == "none":
            self._dst_width, self._dst_height = self._src_width, self._src_height
        else:
            self._dst_width, self._dst_height = self._requested_output_geometry()
        if cfg.sr_engine == "dxva_vsr" and (self._dst_width > 4096 or self._dst_height > 2160):
            raise ValueError("DXVA VSR 输出上限为 4096x2160；请降低倍率或改用其他超分引擎")
        if cfg.sr_engine == "osdenhancer":
            if cfg.fi_engine != "none":
                raise ValueError("OSDEnhancer already performs 2x interpolation")
            if (self._dst_width, self._dst_height) != (self._src_width * 4, self._src_height * 4):
                raise ValueError("OSDEnhancer is a native joint 4x/2x model; select exactly 4x")
        if cfg.sr_engine == "sparkvsr":
            if (self._dst_width, self._dst_height) != (self._src_width * 4, self._src_height * 4):
                raise ValueError("SparkVSR is a native 4x model; select exactly 4x")
        _log.info("输出: %dx%d @ %.3f fps", self._dst_width, self._dst_height,
                  self._output_fps())
        _log.info("管线: %s -> %s -> %s", cfg.sr_engine if cfg.sr_first else cfg.fi_engine,
                  cfg.fi_engine if cfg.sr_first else cfg.sr_engine, cfg.encode.codec)

    def _natural_fps(self) -> float:
        if self._config.sr_engine == "osdenhancer":
            return self._src_fps * 2
        return self._src_fps * (self._config.fi_multiplier if self._config.fi_engine != "none" else 1)

    def _output_fps(self) -> float:
        return self._config.fps or self._natural_fps()

    def _fusion_python(self) -> str:
        """Prefer a CUDA Python containing both PyTorch and NV-VFX."""
        from ._env import get_cached_python_envs

        preferred = self._config.torch_python
        preferred_path = os.path.abspath(preferred) if preferred else None
        # Consume an explicit scan cache first; an on-demand scan is only used
        # below when a frozen CLI explicitly needs this external runtime.
        environments = get_cached_python_envs()
        usable = [item for item in environments
                  if item.get("torch") and item.get("cuda") and item.get("nvvfx")]
        if preferred:
            wanted = os.path.normcase(os.path.abspath(preferred))
            for item in usable:
                if os.path.normcase(os.path.abspath(item.get("exe", ""))) == wanted:
                    return item["exe"]
        if usable:
            return usable[0]["exe"]
        # An explicit CLI choice is authoritative.  The fused worker will
        # validate its imports and cleanly fall back if the environment is not
        # suitable, so no prior GUI scan should be required.
        if preferred_path and os.path.isfile(preferred_path):
            return preferred_path
        if getattr(sys, "frozen", False):
            from ._env import get_python_for_feature
            _log.info("正在按需扫描 CUDA PyTorch / NV-VFX Python 环境...")
            detected = get_python_for_feature("nvvfx")
            if detected:
                _log.info("自动选择 Python 环境: %s", detected)
                return detected
            raise RuntimeError("没有找到同时支持 CUDA PyTorch 与 nvvfx 的 Python 环境")
        return preferred or sys.executable

    def _try_init_cuda_executor(self) -> bool:
        cfg = self._config
        if not (cfg.fi_engine == "rife" and cfg.sr_engine == "nvvfx" and
                not cfg.sr_first and cfg.device != "cpu"):
            return False
        try:
            from .fused_rife_nvvfx import (
                FusedRifeNvvfxEngine, modern_windows_available)
            if not modern_windows_available():
                return False
            engine = FusedRifeNvvfxEngine(
                self._fusion_python(), quality=cfg.sr_quality)
            engine.initialize(
                self._src_width, self._src_height,
                self._dst_width, self._dst_height, cfg.fi_multiplier)
            self._batch_executor = engine
            _log.info("融合快速路径就绪: %s", engine.name)
            return True
        except Exception as exc:
            try:
                if "engine" in locals():
                    engine.release()
            except Exception:
                pass
            _log.warning("融合 CUDA 快速路径不可用，回退到独立后端: %s", exc)
            return False

    def _try_init_native_ncnn(self) -> bool:
        """Replace compatible NCNN CLI engines with one persistent worker."""
        cfg = self._config
        if ((cfg.sr_first and self._sr_engine is not None and
             self._fi_engine is not None) or cfg.device == "cpu" or
                cfg.ncnn_gpu == -1):
            return False
        engine = None
        try:
            from .native_ncnn import (
                NativeNcnnEngine, native_worker_available, spec_from_engines)
            if not native_worker_available():
                return False
            spec = spec_from_engines(
                self._sr_engine, self._fi_engine, cfg.ncnn_gpu)
            if spec is None:
                return False
            engine = NativeNcnnEngine(spec)
            engine.initialize(
                self._src_width, self._src_height,
                self._dst_width, self._dst_height,
                cfg.fi_multiplier if self._fi_engine is not None else 1)
            legacy_engines = (self._fi_engine, self._sr_engine)
            self._fi_engine = None
            self._sr_engine = None
            self._batch_executor = engine
            for legacy in legacy_engines:
                if legacy is not None:
                    try:
                        legacy.release()
                    except Exception:
                        _log.warning(
                            "旧 NCNN 引擎释放失败，常驻 Worker 仍可继续",
                            exc_info=True)
            _log.info("NCNN 常驻快速路径就绪: %s", engine.name)
            return True
        except Exception as exc:
            if engine is not None:
                try:
                    engine.release()
                except Exception:
                    pass
            _log.warning("NCNN 常驻快速路径不可用，回退到 CLI 流水: %s", exc)
            return False

    def _init_engines(self) -> None:
        cfg = self._config
        if self._try_init_cuda_executor():
            return
        ncnn_gpu = cfg.ncnn_gpu
        sr_torch_python = cfg.torch_python
        external_sr_runtime = (
            cfg.sr_engine in {"flashvsr", "seedvr2", "dloral", "osdenhancer", "sparkvsr"}
            or (cfg.sr_engine == "nvvfx" and getattr(sys, "frozen", False)))
        if external_sr_runtime and sr_torch_python is None:
            from ._env import get_python_for_feature
            sr_torch_python = get_python_for_feature(cfg.sr_engine)
        if cfg.sr_engine != "none":
            self._sr_engine = create_sr_engine(
                cfg.sr_engine, device=cfg.device, torch_python=sr_torch_python,
                ncnn_gpu=ncnn_gpu, quality=cfg.sr_quality,
                spark_reference_path=cfg.spark_reference_path,
                spark_reference_indices=cfg.spark_reference_indices,
                spark_reference_guidance=cfg.spark_reference_guidance)
            self._sr_engine.initialize(self._src_width, self._src_height,
                                       self._dst_width, self._dst_height)
            _log.info("超分就绪: %s", self._sr_engine.name)
        if cfg.fi_engine != "none":
            torch_python = cfg.torch_python
            if cfg.fi_engine == "vfimamba" and torch_python is None:
                from ._env import get_python_for_feature
                torch_python = get_python_for_feature("vfimamba")
            elif cfg.fi_engine in {"rife", "ema_vfi"} and torch_python is None:
                from ._env import get_torch_python
                torch_python = get_torch_python()
            if cfg.fi_engine == "vfimamba" and torch_python is None:
                raise RuntimeError("未扫描到满足 VFIMamba 依赖的 CUDA Python 环境")
            self._fi_engine = create_fi_engine(
                cfg.fi_engine, device=cfg.device, quality=cfg.fi_quality,
                torch_python=torch_python, ncnn_gpu=ncnn_gpu)
            width = self._dst_width if cfg.sr_first else self._src_width
            height = self._dst_height if cfg.sr_first else self._src_height
            self._fi_engine.initialize(width, height, cfg.fi_multiplier)
            _log.info("插帧就绪: %s", self._fi_engine.name)
        self._try_init_native_ncnn()

    def _selected_input(self, decoder) -> Iterator[np.ndarray]:
        cfg = self._config
        skip = int(round((cfg.start_time or 0.0) * self._src_fps))
        limit = int(round(cfg.duration * self._src_fps)) if cfg.duration else None
        emitted = 0
        for index, frame in enumerate(decoder):
            self._check_cancelled()
            if index < skip:
                continue
            if limit is not None and emitted >= limit:
                break
            emitted += 1
            yield frame

    @staticmethod
    def _chunks(frames: Iterable[np.ndarray], size: int) -> Iterator[List[np.ndarray]]:
        previous = None
        chunk: List[np.ndarray] = []
        for frame in frames:
            if previous is not None and not chunk:
                chunk.append(previous)
            chunk.append(frame)
            previous = frame
            if len(chunk) >= size:
                yield chunk
                chunk = []
        if chunk and (len(chunk) > 1 or previous is None):
            yield chunk
        elif chunk and previous is not None and len(chunk) == 1:
            yield chunk

    def _sr_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if self._sr_engine is None:
            return frames
        return self._sr_engine.process_batch(frames)

    def _fi_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if self._fi_engine is None or len(frames) < 2:
            return frames
        return self._fi_engine.interpolate_batch(frames)

    def _directory_chain_available(self) -> bool:
        available = (
            self._sr_engine is not None and self._fi_engine is not None
            and self._sr_engine.supports_directory_batch
            and self._fi_engine.supports_directory_batch)
        if not available:
            return False
        if self._config.sr_first:
            return self._sr_engine.batch_output_size == (
                self._dst_width, self._dst_height)
        return True

    def _run_directory_chain(self, work: str, input_dir: str,
                             input_count: int) -> Tuple[str, int]:
        if self._config.sr_first:
            sr_dir = os.path.join(work, "upscaled")
            output_dir = os.path.join(work, "interpolated")
            count = self._sr_engine.process_directory(
                input_dir, sr_dir, input_count)
            count = self._fi_engine.process_directory(sr_dir, output_dir, count)
        else:
            rife_dir = os.path.join(work, "interpolated")
            output_dir = os.path.join(work, "upscaled")
            count = self._fi_engine.process_directory(
                input_dir, rife_dir, input_count)
            count = self._sr_engine.process_directory(rife_dir, output_dir, count)
        return output_dir, count

    def _submit_directory_cleanup(self, work: str) -> None:
        if self._temp_cleaner is None:
            self._temp_cleaner = _AsyncDirectoryCleaner()
        self._temp_cleaner.submit(work)

    def _transform_directory_chain(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        work = tempfile.mkdtemp(prefix="lve_ncnn_chain_")
        try:
            input_dir = os.path.join(work, "input")
            write_frames(frames, input_dir, "NCNN 管线")
            output_dir, count = self._run_directory_chain(
                work, input_dir, len(frames))
            result = read_frames(
                output_dir, count, (self._dst_width, self._dst_height), "NCNN 管线")
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
        self._submit_directory_cleanup(work)
        return result

    def _transform(self, frames: List[np.ndarray],
                   skip_first: bool = False) -> List[np.ndarray]:
        if self._batch_executor is not None:
            return self._batch_executor.process(
                frames, skip_first=skip_first)
        if self._directory_chain_available() and len(frames) >= 2:
            return self._transform_directory_chain(frames)
        if self._config.sr_first:
            return self._fi_sequence(self._sr_sequence(frames))
        return self._sr_sequence(self._fi_sequence(frames))

    def _batch_size(self) -> int:
        if self._batch_executor is not None:
            return self._batch_executor.batch_size
        preferred = int(getattr(
            self._sr_engine, "preferred_batch_size", 0) or 0)
        if preferred:
            if self._fi_engine is not None and not self._config.sr_first:
                preferred = (
                    (preferred - 1) // max(1, self._config.fi_multiplier)
                ) + 1
            return max(3, preferred)
        uses_batch = any(
            engine is not None and engine.supports_batch
            for engine in (self._sr_engine, self._fi_engine))
        if not uses_batch:
            return 3
        largest_pixels = max(
            self._src_width * self._src_height,
            self._dst_width * self._dst_height,
            self._sr_engine.batch_output_pixels
            if self._sr_engine is not None else 0)
        output_multiplier = (2 if self._config.sr_engine == "osdenhancer"
                             else self._config.fi_multiplier
                             if self._config.fi_engine != "none" else 1)
        budget = (384 if self._directory_chain_available() else 192) * 1024 * 1024
        max_output_frames = max(3, budget // max(1, largest_pixels * 3))
        input_frames = ((max_output_frames - 1) // output_multiplier) + 1
        return max(3, min(32, int(input_frames)))

    def _process_embedded(self) -> None:
        try:
            from .ffmpeg_bridge import FFmpegVideoDecoder, FFmpegVideoEncoder, worker_is_loadable
        except Exception as exc:
            raise RuntimeError("无法导入内嵌 FFmpeg: %s" % exc) from exc
        if not worker_is_loadable():
            raise RuntimeError("内嵌 FFmpeg Worker 或其 DLL 依赖不可用")

        cfg = self._config
        decoder = FFmpegVideoDecoder(cfg.input_path, hardware="auto")
        source_audio = cfg.input_path if cfg.encode.copy_audio else None
        encoder = FFmpegVideoEncoder(
            self._partial_path, self._dst_width, self._dst_height, self._output_fps(),
            codec=cfg.encode.codec, crf=cfg.encode.crf, preset=cfg.encode.preset,
            source_path=source_audio, audio_start=cfg.start_time or 0.0,
            audio_duration=cfg.duration)
        async_encoder = None
        input_count = 0
        first_chunk = True
        resampler = _FrameRateResampler(self._natural_fps(), self._output_fps())
        try:
            decoder.open()
            encoder.open()
            batch_size = self._batch_size()
            temporal_multiplier = (2 if cfg.sr_engine == "osdenhancer" else
                                   (cfg.fi_multiplier if cfg.fi_engine != "none" else 1))
            output_per_batch = ((batch_size - 1) * temporal_multiplier + 1)
            encoder_queue = max(4, min(64, output_per_batch))
            async_encoder = _AsyncEncoder(encoder, queue_size=encoder_queue)
            _log.info("批处理: 输入=%d, 编码队列=%d%s", batch_size, encoder_queue,
                      ", NCNN 三级流水" if self._directory_chain_available() else "")

            def consume(transformed, chunk_length):
                nonlocal first_chunk, input_count
                self._check_cancelled()
                if (not first_chunk and transformed and
                        self._batch_executor is None):
                    transformed = transformed[1:]
                unique_input = (chunk_length if first_chunk
                                else max(0, chunk_length - 1))
                input_count += unique_input
                first_chunk = False
                for frame in transformed:
                    for output_frame in resampler.feed(frame):
                        async_encoder.put(output_frame)
                self._progress("处理", input_count, self._selected_frames)

            chunks = self._chunks(self._selected_input(decoder), batch_size)
            if self._directory_chain_available():
                with NcnnDirectoryStream(
                        chunks, self._run_directory_chain) as stream:
                    for job in stream:
                        try:
                            transformed = read_frames(
                                job.output_dir, job.output_count,
                                (self._dst_width, self._dst_height),
                                "NCNN 管线")
                        except BaseException:
                            shutil.rmtree(job.work, ignore_errors=True)
                            raise
                        self._submit_directory_cleanup(job.work)
                        consume(transformed, job.input_count)
            else:
                for chunk in chunks:
                    transformed = self._transform(
                        chunk, skip_first=not first_chunk)
                    consume(transformed, len(chunk))

            if input_count == 0:
                raise RuntimeError("选定时间范围内没有可处理的视频帧")
            output_count = async_encoder.finish()
            async_encoder = None
            _log.info("输入 %d 帧，输出 %d 帧", input_count, output_count)
        finally:
            if async_encoder is not None:
                try:
                    async_encoder.finish()
                except Exception:
                    pass
            try:
                encoder.close()
            finally:
                decoder.close()

    def _release_engines(self) -> None:
        engines = (
            self._batch_executor, self._fi_engine, self._sr_engine)
        self._batch_executor = None
        self._fi_engine = None
        self._sr_engine = None
        cleaner = self._temp_cleaner
        self._temp_cleaner = None
        if cleaner is not None:
            cleaner.finish()
        for engine in engines:
            if engine is not None:
                try:
                    engine.release()
                except Exception:
                    _log.warning("引擎释放失败", exc_info=True)

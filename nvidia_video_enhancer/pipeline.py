"""
视频增强流水线。

解码/编码方式自动选择:
  优先 → 内嵌 FFmpeg C API (ffmpeg_bridge)，零额外依赖，NVDEC/NVENC 直通
  回退 → subprocess 调用系统 ffmpeg/ffprobe
"""

import os
import subprocess
import json
from typing import Optional
import cv2
from tqdm import tqdm

from .config import ProcessConfig
from .sr import create_sr_engine, SuperResolutionEngine
from .fi import create_fi_engine, FrameInterpolationEngine
from ._logging import get_logger

_log = get_logger(__name__)

_ffmpeg_cache = None
_ffprobe_cache = None


def _find_ffmpeg() -> str:
    global _ffmpeg_cache
    if _ffmpeg_cache is not None:
        return _ffmpeg_cache
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(pkg_dir, "..", "ffmpeg", "build", "bin", "ffmpeg.exe")),
        os.path.normpath(os.path.join(pkg_dir, "..", "..", "ffmpeg", "build", "bin", "ffmpeg.exe")),
        "ffmpeg",
    ]
    for c in candidates:
        if c == "ffmpeg":
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
                _ffmpeg_cache = "ffmpeg"
                return _ffmpeg_cache
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
            except Exception:
                _log.debug("ffmpeg -version 意外失败，跳过", exc_info=True)
                continue
        if os.path.isfile(c):
            _ffmpeg_cache = c
            return _ffmpeg_cache
    _ffmpeg_cache = "ffmpeg"
    return _ffmpeg_cache


def _find_ffprobe() -> str:
    global _ffprobe_cache
    if _ffprobe_cache is not None:
        return _ffprobe_cache
    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path != "ffmpeg":
        probe_path = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe")
        if os.path.isfile(probe_path):
            _ffprobe_cache = probe_path
            return _ffprobe_cache
    _ffprobe_cache = "ffprobe"
    return _ffprobe_cache


_FFMPEG_BRIDGE_AVAILABLE = False
try:
    from .ffmpeg_bridge import FFmpegVideoDecoder, FFmpegVideoEncoder
    _FFMPEG_BRIDGE_AVAILABLE = True
except (FileNotFoundError, OSError, ImportError):
    pass


class VideoEnhancer:

    def __init__(self, config: ProcessConfig):
        self._config = config
        self._sr_engine: Optional[SuperResolutionEngine] = None
        self._fi_engine: Optional[FrameInterpolationEngine] = None
        self._src_width = 0
        self._src_height = 0
        self._src_fps = 0.0
        self._total_frames = 0
        self._dst_width = 0
        self._dst_height = 0

    def run(self) -> None:
        self._probe_input()
        self._calc_output_size()
        self._init_engines()
        try:
            if _FFMPEG_BRIDGE_AVAILABLE:
                _log.info("使用内嵌 FFmpeg C API (NVDEC 解码 + NVENC 编码)")
                self._process_embedded()
            else:
                _log.info("使用外部 ffmpeg/ffprobe (subprocess)")
                self._process_subprocess()
        finally:
            self._release_engines()

    def _probe_input(self):
        path = self._config.input_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"输入文件不存在: {path}")

        if _FFMPEG_BRIDGE_AVAILABLE:
            try:
                dec = FFmpegVideoDecoder(path, use_nvdec=True)
                info = dec.probe()
                self._src_width = info["width"]
                self._src_height = info["height"]
                self._src_fps = info["fps"]
                self._total_frames = info["total_frames"]
                _log.info("输入: %dx%d @ %.2ffps", self._src_width, self._src_height, self._src_fps)
                _log.info("帧数: %s", self._total_frames or '未知')
                return
            except Exception as e:
                _log.warning("内嵌探针失败: %s，回退到 ffprobe", e)

        probe = _find_ffprobe()
        if probe == "ffprobe":
            try:
                subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                raise RuntimeError(
                    "找不到 ffprobe。请安装 FFmpeg 并确保 ffprobe 在 PATH 中。"
                )

        probe_cmd = [
            probe, "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", path
        ]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffprobe 探测失败 (返回码 {e.returncode}): {path}\n"
                f"stderr: {e.stderr.strip()}"
            ) from e
        except FileNotFoundError:
            raise RuntimeError(f"找不到 ffprobe 可执行文件: {probe}")

        try:
            info = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"ffprobe 输出解析失败: {e}\n"
                f"原始输出前200字符: {result.stdout[:200]}"
            ) from e

        video_stream = None
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                video_stream = s
                break

        if video_stream is None:
            raise RuntimeError("输入文件中未找到视频流")

        self._src_width = int(video_stream["width"])
        self._src_height = int(video_stream["height"])

        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            den_f = float(den)
            self._src_fps = float(num) / den_f if den_f > 0 else 30.0
        else:
            self._src_fps = float(fps_str) if fps_str else 30.0

        nb = video_stream.get("nb_frames", "0")
        self._total_frames = int(nb)

        _log.info("输入: %dx%d @ %.2ffps", self._src_width, self._src_height, self._src_fps)
        _log.info("帧数: %s", self._total_frames or '未知')

        if self._total_frames == 0:
            dur_str = info.get("format", {}).get("duration", "0")
            duration = float(dur_str)
            if duration > 0:
                self._total_frames = int(duration * self._src_fps)

    def _calc_output_size(self):
        cfg = self._config
        if cfg.sr_engine == "none":
            self._dst_width = self._src_width
            self._dst_height = self._src_height
        elif cfg.width > 0 and cfg.height > 0:
            self._dst_width = cfg.width
            self._dst_height = cfg.height
        elif cfg.width > 0:
            ratio = cfg.width / self._src_width
            self._dst_width = cfg.width
            self._dst_height = int(self._src_height * ratio)
        elif cfg.height > 0:
            ratio = cfg.height / self._src_height
            self._dst_width = int(self._src_width * ratio)
            self._dst_height = cfg.height
        else:
            self._dst_width = int(self._src_width * cfg.scale)
            self._dst_height = int(self._src_height * cfg.scale)

        self._dst_width = (self._dst_width // 2) * 2
        self._dst_height = (self._dst_height // 2) * 2

        out_fps = self._calc_output_fps()

        _log.info("输出: %dx%d @ %.2ffps", self._dst_width, self._dst_height, out_fps)
        _log.info("超分引擎: %s", cfg.sr_engine)
        _log.info("插帧引擎: %s (x%d)", cfg.fi_engine, cfg.fi_multiplier)

    def _init_engines(self):
        cfg = self._config

        if cfg.sr_engine == "none":
            self._sr_engine = None
            _log.info("超分: 无")
        else:
            _log.info("初始化超分引擎 (%s) ...", cfg.sr_engine)
            try:
                torch_python = getattr(cfg, "torch_python", None)
                self._sr_engine = create_sr_engine(cfg.sr_engine, device=cfg.device,
                                                    torch_python=torch_python)
                self._sr_engine.initialize(
                    self._src_width, self._src_height,
                    self._dst_width, self._dst_height
                )
            except Exception as e:
                _log.warning("%s 初始化失败: %s", cfg.sr_engine, e)
                _log.info("自动回退到 bicubic 超分引擎 ...")
                from .sr.fallback import BicubicEngine
                self._sr_engine = BicubicEngine()
                self._sr_engine.initialize(
                    self._src_width, self._src_height,
                    self._dst_width, self._dst_height
                )
            _log.info("超分引擎就绪: %s", self._sr_engine.name)

        if cfg.fi_engine != "none":
            _log.info("初始化插帧引擎 (%s) ...", cfg.fi_engine)
            torch_python = getattr(cfg, "torch_python", None)
            if torch_python is None and cfg.fi_engine in ("rife", "torch_flow"):
                from ._env import get_torch_python
                torch_python = get_torch_python()

            self._fi_engine = create_fi_engine(cfg.fi_engine, device=cfg.device,
                                               quality=getattr(cfg, "fi_quality", "balanced"),
                                               torch_python=torch_python)
            fi_init_w = self._dst_width if cfg.sr_first else self._src_width
            fi_init_h = self._dst_height if cfg.sr_first else self._src_height
            self._fi_engine.initialize(
                fi_init_w, fi_init_h,
                multiplier=cfg.fi_multiplier
            )
            _log.info("插帧引擎就绪: %s", self._fi_engine.name)
        else:
            self._fi_engine = None

    def _calc_output_fps(self) -> float:
        cfg = self._config
        out_fps = self._src_fps
        if cfg.fi_engine != "none":
            out_fps = self._src_fps * cfg.fi_multiplier
        if cfg.fps:
            out_fps = cfg.fps
        return out_fps

    # ====== 内嵌 FFmpeg 模式 ======

    def _process_embedded(self):
        cfg = self._config
        out_fps = self._calc_output_fps()

        decoder = None
        encoder = None
        close_errors = []

        try:
            decoder = FFmpegVideoDecoder(cfg.input_path, use_nvdec=True)
            decoder.open()
            _log.info("解码: %dx%d @ %.2ffps (NVDEC)", decoder.width, decoder.height, decoder.fps)

            encoder = FFmpegVideoEncoder(
                cfg.output_path,
                self._dst_width, self._dst_height,
                out_fps,
                codec=cfg.encode.codec,
                crf=cfg.encode.crf,
                preset=cfg.encode.preset,
                source_path=cfg.input_path,
            )
            encoder.open()

            _log.info("编码: %s (NVENC)", cfg.encode.codec)
            _log.info("%d 帧待处理", self._total_frames)

            frame_count = 0
            encoded_count = 0
            do_sr = self._sr_engine.process if self._sr_engine else (lambda x: x)
            do_fi = self._fi_engine.interpolate if self._fi_engine else None

            if do_fi is not None and self._total_frames >= 2:
                if cfg.sr_first:
                    frame_count, encoded_count = self._process_sr_then_fi(
                        decoder, encoder, do_sr, do_fi)
                else:
                    frame_count, encoded_count = self._process_with_fi(
                        decoder, encoder, do_sr, do_fi)
            else:
                frame_count, encoded_count = self._process_sr_only(
                    decoder, encoder, do_sr)

            _log.info("输入帧: %d, 输出帧: %d", frame_count, encoded_count)

            try:
                encoder.close()
            except Exception as e:
                close_errors.append(f"编码器关闭: {e}")
                _log.warning("编码器关闭失败: %s", e)
            try:
                decoder.close()
            except Exception as e:
                close_errors.append(f"解码器关闭: {e}")
                _log.warning("解码器关闭失败: %s", e)

            if close_errors:
                _log.warning("关闭阶段出现 %d 个警告，输出文件可能不完整", len(close_errors))

            if os.path.exists(cfg.output_path):
                size_mb = os.path.getsize(cfg.output_path) / (1024 * 1024)
                _log.info("输出文件: %s", cfg.output_path)
                _log.info("大小: %.1f MB", size_mb)
            else:
                _log.error("输出文件未生成: %s", cfg.output_path)

        except Exception as e:
            if encoder is not None:
                try:
                    encoder.close()
                except Exception as ce:
                    _log.warning("编码器关闭失败: %s", ce)
            if decoder is not None:
                try:
                    decoder.close()
                except Exception as ce:
                    _log.warning("解码器关闭失败: %s", ce)
            raise RuntimeError(
                f"处理失败: {e}\n"
                "请尝试 --sr-engine bicubic 或 --fi-engine none"
            ) from e

    def _process_with_fi(self, decoder, encoder, do_sr, do_fi):
        import threading
        import queue
        import cv2

        frame_count = 0
        encoded_count = [0]
        enc_w = self._dst_width
        enc_h = self._dst_height
        enc_error = [None]
        enc_lock = threading.Lock()

        enc_q = queue.Queue(maxsize=4)
        enc_done = threading.Event()

        def _enc_worker():
            try:
                while True:
                    item = enc_q.get()
                    if item is None:
                        break
                    bgr, w, h = item
                    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).ravel()
                    encoder.encode_yuv(yuv, w, h)
                    with enc_lock:
                        encoded_count[0] += 1
            except Exception as e:
                with enc_lock:
                    enc_error[0] = e
            enc_done.set()

        enc_th = threading.Thread(target=_enc_worker, daemon=True)
        enc_th.start()

        fi_q = queue.Queue(maxsize=1)
        fi_out_q = queue.Queue(maxsize=1)
        fi_running = threading.Event()
        fi_running.set()

        def _fi_worker():
            try:
                while fi_running.is_set():
                    try:
                        f0, f1 = fi_q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if f0 is None:
                        break
                    fi_out_q.put(do_fi(f0, f1))
            except Exception as e:
                fi_out_q.put(e)

        fi_th = threading.Thread(target=_fi_worker, daemon=True)
        fi_th.start()

        def _check_enc():
            with enc_lock:
                err = enc_error[0]
            if err:
                raise RuntimeError(f"编码线程错误: {err}")

        try:
            raw0 = None
            previous_fi_result = None
            first_pair = True
            for raw1 in tqdm(decoder, total=self._total_frames, desc="FI+SR", unit="帧"):

                if raw0 is not None:
                    if first_pair:
                        first_pair = False
                    else:
                        fi_result = previous_fi_result
                        if isinstance(fi_result, Exception):
                            raise RuntimeError(f"FI 线程错误: {fi_result}")
                        for im in fi_result:
                            _check_enc()
                            enc_q.put((do_sr(im).copy(), enc_w, enc_h))

                    fi_q.put((raw0, raw1))

                    _check_enc()
                    enc_q.put((do_sr(raw0).copy(), enc_w, enc_h))
                    frame_count += 1

                    if not first_pair:
                        try:
                            previous_fi_result = fi_out_q.get(timeout=60)
                        except queue.Empty:
                            raise RuntimeError("FI 线程超时（60秒无响应）")

                raw0 = raw1

            if not first_pair:
                fi_result = previous_fi_result
                if isinstance(fi_result, Exception):
                    raise RuntimeError(f"FI 线程错误: {fi_result}")
                for im in fi_result:
                    _check_enc()
                    enc_q.put((do_sr(im).copy(), enc_w, enc_h))

            if raw0 is not None:
                _check_enc()
                enc_q.put((do_sr(raw0).copy(), enc_w, enc_h))
                frame_count += 1

        finally:
            fi_q.put((None, None))
            fi_running.clear()
            fi_th.join(timeout=5.0)
            enc_q.put(None)
            enc_done.wait(timeout=120)
            enc_th.join(timeout=5.0)

        with enc_lock:
            err = enc_error[0]
            final_count = encoded_count[0]
        if err:
            raise RuntimeError(f"编码线程错误: {err}")
        return frame_count, final_count

    def _process_sr_then_fi(self, decoder, encoder, do_sr, do_fi):
        """先超分，再用高分帧插帧。FI 输出已是目标分辨率，无需再 SR。"""
        import threading
        import queue
        import cv2

        frame_count = 0
        encoded_count = [0]
        enc_w = self._dst_width
        enc_h = self._dst_height
        enc_error = [None]
        enc_lock = threading.Lock()

        enc_q = queue.Queue(maxsize=4)
        enc_done = threading.Event()

        def _enc_worker():
            try:
                while True:
                    item = enc_q.get()
                    if item is None:
                        break
                    bgr, w, h = item
                    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).ravel()
                    encoder.encode_yuv(yuv, w, h)
                    with enc_lock:
                        encoded_count[0] += 1
            except Exception as e:
                with enc_lock:
                    enc_error[0] = e
            enc_done.set()

        enc_th = threading.Thread(target=_enc_worker, daemon=True)
        enc_th.start()

        fi_q = queue.Queue(maxsize=1)
        fi_out_q = queue.Queue(maxsize=1)
        fi_running = threading.Event()
        fi_running.set()

        def _fi_worker():
            try:
                while fi_running.is_set():
                    try:
                        f0, f1 = fi_q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if f0 is None:
                        break
                    fi_out_q.put(do_fi(f0, f1))
            except Exception as e:
                fi_out_q.put(e)

        fi_th = threading.Thread(target=_fi_worker, daemon=True)
        fi_th.start()

        def _check_enc():
            with enc_lock:
                err = enc_error[0]
            if err:
                raise RuntimeError(f"编码线程错误: {err}")

        try:
            raw0 = None
            previous_fi_result = None
            first_pair = True
            for raw1 in tqdm(decoder, total=self._total_frames, desc="SR+FI", unit="帧"):
                sr1 = do_sr(raw1)

                if raw0 is not None:
                    if first_pair:
                        first_pair = False
                    else:
                        fi_result = previous_fi_result
                        if isinstance(fi_result, Exception):
                            raise RuntimeError(f"FI 线程错误: {fi_result}")
                        for im in fi_result:
                            _check_enc()
                            enc_q.put((im, enc_w, enc_h))

                    fi_q.put((sr0, sr1))

                    _check_enc()
                    enc_q.put((sr0, enc_w, enc_h))
                    frame_count += 1

                    if not first_pair:
                        try:
                            previous_fi_result = fi_out_q.get(timeout=60)
                        except queue.Empty:
                            raise RuntimeError("FI 线程超时（60秒无响应）")

                raw0 = raw1
                sr0 = sr1

            if not first_pair:
                fi_result = previous_fi_result
                if isinstance(fi_result, Exception):
                    raise RuntimeError(f"FI 线程错误: {fi_result}")
                for im in fi_result:
                    _check_enc()
                    enc_q.put((im, enc_w, enc_h))

            if raw0 is not None:
                _check_enc()
                enc_q.put((sr0, enc_w, enc_h))
                frame_count += 1

        finally:
            fi_q.put((None, None))
            fi_running.clear()
            fi_th.join(timeout=5.0)
            enc_q.put(None)
            enc_done.wait(timeout=120)
            enc_th.join(timeout=5.0)

        with enc_lock:
            err = enc_error[0]
            final_count = encoded_count[0]
        if err:
            raise RuntimeError(f"编码线程错误: {err}")
        return frame_count, final_count

    def _process_sr_only(self, decoder, encoder, do_sr):
        import threading
        import queue
        import time
        import cv2

        frame_count = 0
        encoded_count = [0]
        enc_w = self._dst_width
        enc_h = self._dst_height
        enc_error = [None]
        enc_lock = threading.Lock()

        enc_q = queue.Queue(maxsize=2)

        def _enc_worker():
            try:
                while True:
                    item = enc_q.get()
                    if item is None:
                        break
                    bgr = item
                    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).ravel()
                    encoder.encode_yuv(yuv, enc_w, enc_h)
                    with enc_lock:
                        encoded_count[0] += 1
            except Exception as e:
                with enc_lock:
                    enc_error[0] = e

        enc_th = threading.Thread(target=_enc_worker, daemon=True)
        enc_th.start()

        def _check_enc():
            with enc_lock:
                err = enc_error[0]
            if err:
                raise RuntimeError(f"编码线程错误: {err}")

        t_start = time.perf_counter()
        try:
            for raw_frame in tqdm(decoder, total=self._total_frames,
                                  desc="超分→YUV", unit="帧"):
                sr_bgr = do_sr(raw_frame)
                _check_enc()
                while True:
                    try:
                        enc_q.put(sr_bgr.copy(), timeout=1.0)
                        break
                    except queue.Full:
                        _check_enc()
                frame_count += 1
        finally:
            enc_q.put(None)
            enc_th.join(timeout=300)
            with enc_lock:
                err = enc_error[0]
            if err:
                raise RuntimeError(f"编码线程错误: {err}")

        elapsed = time.perf_counter() - t_start
        if frame_count > 0:
            _log.info("纯超分: %d帧 / %.1fs = %.2f帧/s", frame_count, elapsed, frame_count / elapsed)

        return frame_count, encoded_count[0]

    # ====== Subprocess 回退模式 ======

    def _process_subprocess(self):
        import tempfile
        import shutil

        cfg = self._config
        out_fps = self._calc_output_fps()
        tmp_dir = tempfile.mkdtemp(prefix="nve_")

        try:
            raw_path = os.path.join(tmp_dir, "raw_frames")
            os.makedirs(raw_path, exist_ok=True)

            _log.info("解码视频帧 ...")
            self._decode_frames_subprocess(raw_path)

            enhanced_dir = os.path.join(tmp_dir, "enhanced")
            os.makedirs(enhanced_dir, exist_ok=True)

            _log.info("超分处理 ...")
            frame_count = self._process_frames_files(raw_path, enhanced_dir)

            _log.info("编码输出视频 ...")
            self._encode_output_subprocess(enhanced_dir, out_fps)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _decode_frames_subprocess(self, output_dir: str):
        cfg = self._config
        ffmpeg = _find_ffmpeg()
        args = [
            ffmpeg, "-y",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", cfg.input_path,
            "-vsync", "0",
            "-pix_fmt", "bgr24",
        ]
        if cfg.start_time is not None:
            args += ["-ss", str(cfg.start_time)]
        if cfg.duration is not None:
            args += ["-t", str(cfg.duration)]
        args.append(os.path.join(output_dir, "frame_%08d.png"))
        try:
            subprocess.run(args, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr.decode(errors="replace") if e.stderr else "(无)"
            raise RuntimeError(
                f"ffmpeg 解码失败 (返回码 {e.returncode}):\n{stderr_text}"
            ) from e

    def _process_frames_files(self, raw_dir: str, out_dir: str) -> int:
        frame_files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".png"))
        if not frame_files:
            raise RuntimeError("解码后没有找到帧文件")

        _log.info("共 %d 帧待处理", len(frame_files))

        prev_frame = None
        frame_idx = 0

        for fname in tqdm(frame_files, desc="超分+插帧", unit="帧"):
            fpath = os.path.join(raw_dir, fname)
            frame = cv2.imread(fpath)
            if frame is None:
                continue

            enhanced = self._sr_engine.process(frame)

            if self._fi_engine is not None and prev_frame is not None:
                inter_frames = self._fi_engine.interpolate(prev_frame, enhanced)
                for inter in inter_frames:
                    out_path = os.path.join(out_dir, f"frame_{frame_idx:08d}.png")
                    cv2.imwrite(out_path, inter)
                    frame_idx += 1

            out_path = os.path.join(out_dir, f"frame_{frame_idx:08d}.png")
            cv2.imwrite(out_path, enhanced)
            frame_idx += 1
            prev_frame = enhanced

        _log.info("输出帧总数: %d", frame_idx)
        return frame_idx

    def _encode_output_subprocess(self, frames_dir: str, fps: float):
        cfg = self._config
        ffmpeg = _find_ffmpeg()
        args = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%08d.png"),
            "-c:v", cfg.encode.codec,
            "-preset", cfg.encode.preset,
            "-cq", str(cfg.encode.crf),
            "-pix_fmt", cfg.encode.pixel_format,
        ]
        if cfg.encode.codec in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
            args += ["-rc", "vbr", "-tune", "hq"]
        args.append(cfg.output_path)
        try:
            subprocess.run(args, check=True)
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr.decode(errors="replace") if e.stderr else "(无)"
            raise RuntimeError(
                f"ffmpeg 编码失败 (返回码 {e.returncode}):\n{stderr_text}"
            ) from e

        if os.path.exists(cfg.output_path):
            size_mb = os.path.getsize(cfg.output_path) / (1024 * 1024)
            _log.info("输出文件: %s", cfg.output_path)
            _log.info("大小: %.1f MB", size_mb)
        else:
            _log.warning("编码后未找到输出文件: %s", cfg.output_path)

    def _release_engines(self):
        if self._sr_engine:
            self._sr_engine.release()
        if self._fi_engine:
            self._fi_engine.release()

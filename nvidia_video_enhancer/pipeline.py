"""
视频增强流水线。

解码/编码方式自动选择:
  优先 → 内嵌 FFmpeg C API (ffmpeg_bridge)，零额外依赖，NVDEC/NVENC 直通
  回退 → subprocess 调用系统 ffmpeg/ffprobe
"""

import os
import sys
import subprocess
import json
from typing import Optional
import numpy as np
import cv2
from tqdm import tqdm

from .config import ProcessConfig
from .sr import create_sr_engine, SuperResolutionEngine
from .fi import create_fi_engine, FrameInterpolationEngine


def _find_ffmpeg() -> str:
    """查找 ffmpeg.exe。优先用编译的，回退系统 PATH。"""
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
                return "ffmpeg"
            except Exception:
                continue
        if os.path.isfile(c):
            return c
    return "ffmpeg"


def _find_ffprobe() -> str:
    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path != "ffmpeg":
        probe_path = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe")
        if os.path.isfile(probe_path):
            return probe_path
    return "ffprobe"


_FFMPEG_BRIDGE_AVAILABLE = False
try:
    from .ffmpeg_bridge import FFmpegVideoDecoder, FFmpegVideoEncoder
    _FFMPEG_BRIDGE_AVAILABLE = True
except (FileNotFoundError, OSError, ImportError):
    pass


class VideoEnhancer:
    """
    视频增强流水线：解码 → 超分 → 插帧 → 编码

    流程：
    1. 使用内嵌 FFmpeg C API (或 subprocess ffmpeg) 解码视频帧
    2. 每帧送入超分引擎（NVIDIA VSR / Real-ESRGAN / Bicubic）
    3. 相邻帧之间送入插帧引擎（RIFE / Blend）生成中间帧
    4. 使用内嵌 FFmpeg C API (或 subprocess ffmpeg) 编码输出视频
    """

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
                print("[信息] 使用内嵌 FFmpeg C API (NVDEC 解码 + NVENC 编码)")
                self._process_embedded()
            else:
                print("[信息] 使用外部 ffmpeg/ffprobe (subprocess)")
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
                print(f"[信息] 输入: {self._src_width}x{self._src_height} @ {self._src_fps:.2f}fps")
                print(f"[信息] 帧数: {self._total_frames or '未知'}")
                return
            except Exception as e:
                print(f"[警告] 内嵌探针失败: {e}，回退到 ffprobe")

        probe_cmd = [
            _find_ffprobe(), "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)

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
            self._src_fps = float(num) / float(den)
        else:
            self._src_fps = float(fps_str)

        nb = video_stream.get("nb_frames", "0")
        self._total_frames = int(nb)

        print(f"[信息] 输入: {self._src_width}x{self._src_height} @ {self._src_fps:.2f}fps")
        print(f"[信息] 帧数: {self._total_frames or '未知'}")

        if self._total_frames == 0:
            dur_str = info.get("format", {}).get("duration", "0")
            duration = float(dur_str)
            if duration > 0:
                self._total_frames = int(duration * self._src_fps)

    def _calc_output_size(self):
        cfg = self._config
        if cfg.width > 0 and cfg.height > 0:
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

        out_fps = self._src_fps
        if cfg.fi_engine != "none":
            out_fps = self._src_fps * cfg.fi_multiplier
        if cfg.fps:
            out_fps = cfg.fps

        print(f"[信息] 输出: {self._dst_width}x{self._dst_height} @ {out_fps:.2f}fps")
        print(f"[信息] 超分引擎: {cfg.sr_engine}")
        print(f"[信息] 插帧引擎: {cfg.fi_engine} (x{cfg.fi_multiplier})")

    def _init_engines(self):
        cfg = self._config
        print(f"[信息] 初始化超分引擎 ({cfg.sr_engine}) ...")
        try:
            self._sr_engine = create_sr_engine(cfg.sr_engine, device=cfg.device)
            self._sr_engine.initialize(
                self._src_width, self._src_height,
                self._dst_width, self._dst_height
            )
        except Exception as e:
            print(f"[警告] {cfg.sr_engine} 初始化失败: {e}")
            print(f"[信息] 自动回退到 bicubic 超分引擎 ...")
            from .sr.fallback import BicubicEngine
            self._sr_engine = BicubicEngine()
            self._sr_engine.initialize(
                self._src_width, self._src_height,
                self._dst_width, self._dst_height
            )
        print(f"[信息] 超分引擎就绪: {self._sr_engine.name}")

        if cfg.fi_engine != "none":
            print(f"[信息] 初始化插帧引擎 ({cfg.fi_engine}) ...")
            self._fi_engine = create_fi_engine(cfg.fi_engine, device=cfg.device,
                                               quality=getattr(cfg, "fi_quality", "balanced"))
            self._fi_engine.initialize(
                self._src_width, self._src_height,
                multiplier=cfg.fi_multiplier
            )
            print(f"[信息] 插帧引擎就绪: {self._fi_engine.name}")
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

        try:
            decoder = FFmpegVideoDecoder(cfg.input_path, use_nvdec=True)
            decoder.open()
            print(f"[信息] 解码: {decoder.width}x{decoder.height} @ {decoder.fps:.2f}fps (NVDEC)")

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

            print(f"[信息] 编码: {cfg.encode.codec} (NVENC)")
            print(f"[信息] {self._total_frames} 帧待处理")

            frame_count = 0
            encoded_count = 0
            do_sr = self._sr_engine.process
            do_fi = self._fi_engine.interpolate if self._fi_engine else None

            if do_fi is not None and self._total_frames >= 2:
                frame_count, encoded_count = self._process_with_fi(
                    decoder, encoder, do_sr, do_fi)
            else:
                frame_count, encoded_count = self._process_sr_only(
                    decoder, encoder, do_sr)

            encoder.close()
            decoder.close()

            print(f"[信息] 输入帧: {frame_count}, 输出帧: {encoded_count}")

            size_mb = os.path.getsize(cfg.output_path) / (1024 * 1024)
            print(f"\n[完成] 输出文件: {cfg.output_path}")
            print(f"[完成] 大小: {size_mb:.1f} MB")

        except Exception as e:
            if encoder is not None:
                try: encoder.close()
                except Exception: pass
            if decoder is not None:
                try: decoder.close()
                except Exception: pass
            raise RuntimeError(
                f"处理失败: {e}\n"
                "请尝试 --sr-engine bicubic 或 --fi-engine none"
            ) from e

    def _process_with_fi(self, decoder, encoder, do_sr, do_fi):
        """FI(CPU) + SR(GPU) + Enc(线程) — GPU 永不等待 FI。"""
        import threading
        import queue
        import cv2

        frame_count = 0
        encoded_count = [0]
        enc_w = self._dst_width
        enc_h = self._dst_height

        # ====== 编码线程 ======
        enc_q = queue.Queue(maxsize=4)
        enc_done = threading.Event()

        def _enc_worker():
            while True:
                item = enc_q.get()
                if item is None:
                    break
                yuv, w, h = item
                encoder.encode_yuv(yuv, w, h)
                encoded_count[0] += 1
            enc_done.set()

        enc_th = threading.Thread(target=_enc_worker, daemon=True)
        enc_th.start()

        # ====== FI 线程 ======
        fi_q = queue.Queue(maxsize=1)
        fi_ready = threading.Event()
        fi_running = [True]
        fi_result = [None]

        def _fi_worker():
            while fi_running[0]:
                try:
                    f0, f1 = fi_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if f0 is None:
                    break
                fi_result[0] = do_fi(f0, f1)
                fi_ready.set()

        fi_th = threading.Thread(target=_fi_worker, daemon=True)
        fi_th.start()

        def _to_yuv(bgr):
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).ravel()

        # ====== 主线程: 永不等 FI，GPU 持续忙碌 ======
        try:
            raw0 = None
            fi_pending = False
            for raw1 in tqdm(decoder, total=self._total_frames, desc="FI+SR", unit="帧"):

                if raw0 is not None:
                    if fi_pending:
                        fi_ready.wait()
                        fi_ready.clear()
                        for im in fi_result[0]:
                            enc_q.put((_to_yuv(do_sr(im)), enc_w, enc_h))
                        fi_pending = False

                    fi_q.put((raw0, raw1))
                    fi_pending = True

                    enc_q.put((_to_yuv(do_sr(raw0)), enc_w, enc_h))
                    frame_count += 1

                raw0 = raw1

            if fi_pending:
                fi_ready.wait()
                for im in fi_result[0]:
                    enc_q.put((_to_yuv(do_sr(im)), enc_w, enc_h))

            if raw0 is not None:
                enc_q.put((_to_yuv(do_sr(raw0)), enc_w, enc_h))
                frame_count += 1

        finally:
            fi_running[0] = False
            fi_q.put((None, None))
            fi_th.join(timeout=2.0)
            enc_q.put(None)
            enc_done.wait(timeout=120)
            enc_th.join(timeout=2.0)

        return frame_count, encoded_count[0]

    def _process_sr_only(self, decoder, encoder, do_sr):
        """纯超分 — 编码线程并行，GPU 不等 NVENC + BGR→YUV。"""
        import threading
        import queue
        import time
        import cv2

        frame_count = 0
        encoded_count = [0]
        enc_w = self._dst_width
        enc_h = self._dst_height
        enc_error = [None]

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
                    encoded_count[0] += 1
            except Exception as e:
                enc_error[0] = e

        enc_th = threading.Thread(target=_enc_worker, daemon=True)
        enc_th.start()

        t_start = time.perf_counter()
        try:
            for raw_frame in tqdm(decoder, total=self._total_frames,
                                  desc="超分→YUV", unit="帧"):
                sr_bgr = do_sr(raw_frame)
                enc_q.put(sr_bgr)
                if enc_error[0]:
                    raise RuntimeError(f"编码线程错误: {enc_error[0]}")
                frame_count += 1
        finally:
            enc_q.put(None)
            enc_th.join(timeout=300)
            if enc_error[0]:
                raise RuntimeError(f"编码线程错误: {enc_error[0]}")

        elapsed = time.perf_counter() - t_start
        if frame_count > 0:
            print(f"[性能] 纯超分: {frame_count}帧 / {elapsed:.1f}s = {frame_count/elapsed:.2f}帧/s")

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

            print("[信息] 解码视频帧 ...")
            self._decode_frames_subprocess(raw_path)

            enhanced_dir = os.path.join(tmp_dir, "enhanced")
            os.makedirs(enhanced_dir, exist_ok=True)

            print("[信息] 超分处理 ...")
            frame_count = self._process_frames_files(raw_path, enhanced_dir)

            print(f"[信息] 编码输出视频 ...")
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
        subprocess.run(args, check=True, capture_output=True)

    def _process_frames_files(self, raw_dir: str, out_dir: str) -> int:
        frame_files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".png"))
        if not frame_files:
            raise RuntimeError("解码后没有找到帧文件")

        print(f"[信息] 共 {len(frame_files)} 帧待处理")

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

        print(f"[信息] 输出帧总数: {frame_idx}")
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
        subprocess.run(args, check=True)

        size_mb = os.path.getsize(cfg.output_path) / (1024 * 1024)
        print(f"\n[完成] 输出文件: {cfg.output_path}")
        print(f"[完成] 大小: {size_mb:.1f} MB")

    def _release_engines(self):
        if self._sr_engine:
            self._sr_engine.release()
        if self._fi_engine:
            self._fi_engine.release()

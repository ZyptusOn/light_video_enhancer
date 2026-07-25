"""Repeatable end-to-end benchmark for Light Video Enhancer pipelines.

The benchmark launches the same frozen backend used by the WinUI frontend,
samples system CPU and NVIDIA GPU counters, probes every output with the
embedded FFmpeg worker, and writes both JSON and Markdown reports.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PIPELINES: Dict[str, Dict[str, str]] = {
    "rife_nvvfx": {
        "label": "RIFE PyTorch + NVIDIA Video Effects VSR",
        "fi": "rife", "sr": "nvvfx", "sr_quality": "quality",
    },
    "rife_dxva_vsr": {
        "label": "RIFE PyTorch + D3D11 driver VSR",
        "fi": "rife", "sr": "dxva_vsr", "sr_quality": "quality",
    },
    "rife_realcugan": {
        "label": "RIFE PyTorch + Real-CUGAN",
        "fi": "rife", "sr": "realcugan", "sr_quality": "quality",
    },
    "rife_realesrgan_fast": {
        "label": "RIFE PyTorch + Real-ESRGAN AnimeVideo-v3",
        "fi": "rife", "sr": "realesrgan", "sr_quality": "fast",
    },
    "rife_ncnn_realcugan_native": {
        "label": "RIFE NCNN + Real-CUGAN (persistent native worker)",
        "fi": "rife_ncnn", "sr": "realcugan", "sr_quality": "quality",
    },
    "rife_ncnn_realcugan_cli": {
        "label": "RIFE NCNN + Real-CUGAN (legacy CLI/PNG)",
        "fi": "rife_ncnn", "sr": "realcugan", "sr_quality": "quality",
        "disable_native_ncnn": "1",
    },
    "rife_ncnn_realesrgan_native": {
        "label": "RIFE NCNN + Real-ESRGAN AnimeVideo-v3 (native)",
        "fi": "rife_ncnn", "sr": "realesrgan", "sr_quality": "fast",
    },
    "rife_ncnn_realesrgan_cli": {
        "label": "RIFE NCNN + Real-ESRGAN AnimeVideo-v3 (CLI/PNG)",
        "fi": "rife_ncnn", "sr": "realesrgan", "sr_quality": "fast",
        "disable_native_ncnn": "1",
    },
    "rife_ncnn_esrgan_native": {
        "label": "RIFE NCNN + ESRGAN classic (persistent native worker)",
        "fi": "rife_ncnn", "sr": "esrgan", "sr_quality": "quality",
    },
    "rife_ncnn_esrgan_cli": {
        "label": "RIFE NCNN + ESRGAN classic (legacy CLI/PNG)",
        "fi": "rife_ncnn", "sr": "esrgan", "sr_quality": "quality",
        "disable_native_ncnn": "1",
    },
    "rife_only": {
        "label": "RIFE PyTorch only",
        "fi": "rife", "sr": "none", "sr_quality": "quality",
    },
    "nvvfx_only": {
        "label": "NVIDIA Video Effects VSR only",
        "fi": "none", "sr": "nvvfx", "sr_quality": "quality",
    },
    "dxva_vsr_only": {
        "label": "D3D11 driver VSR only",
        "fi": "none", "sr": "dxva_vsr", "sr_quality": "quality",
    },
}


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    @property
    def value(self) -> int:
        return (int(self.high) << 32) | int(self.low)


class SystemCpuSampler:
    """Sample total CPU usage without psutil or WMI."""

    def __init__(self) -> None:
        self._last = self._read()

    @staticmethod
    def _read():
        idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
        if not ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            raise ctypes.WinError()
        return idle.value, kernel.value, user.value

    def sample(self) -> float:
        current = self._read()
        idle = current[0] - self._last[0]
        total = ((current[1] - self._last[1]) +
                 (current[2] - self._last[2]))
        self._last = current
        return 0.0 if total <= 0 else max(0.0, min(100.0, 100.0 * (1.0 - idle / total)))


def _mean(values: Iterable[float]) -> float:
    materialised = list(values)
    return statistics.fmean(materialised) if materialised else 0.0


def _maximum(values: Iterable[float]) -> float:
    materialised = list(values)
    return max(materialised) if materialised else 0.0


def _parse_number(value: str) -> Optional[float]:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


def sample_nvidia() -> Dict[str, float]:
    fields = [
        "utilization.gpu", "utilization.encoder", "utilization.decoder",
        "memory.used", "power.draw",
    ]
    command = [
        "nvidia-smi", "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=3, creationflags=flags)
        if result.returncode:
            return {}
        values = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
        parsed = [_parse_number(value) for value in values]
        return {
            name: value for name, value in zip(
                ("gpu", "encoder", "decoder", "memory_mib", "power_w"), parsed)
            if value is not None
        }
    except (OSError, subprocess.SubprocessError, IndexError):
        return {}


def probe_video(path: Path) -> Dict[str, float]:
    from light_video_enhancer.ffmpeg_bridge import FFmpegVideoDecoder

    decoder = FFmpegVideoDecoder(str(path), use_nvdec=False)
    try:
        return decoder.probe()
    finally:
        decoder.close()


@dataclass
class BenchmarkResult:
    name: str
    label: str
    return_code: int
    wall_seconds: float
    init_seconds: float
    processing_seconds: float
    input_frames: int
    output_frames: int
    output_width: int
    output_height: int
    output_fps: float
    output_mib: float
    input_fps_effective: float
    output_fps_effective: float
    realtime_factor: float
    cpu_avg: float
    cpu_peak: float
    gpu_avg: float
    gpu_peak: float
    encoder_avg: float
    encoder_peak: float
    decoder_avg: float
    memory_peak_mib: float
    power_avg_w: float
    log_path: str
    output_path: str
    error: str = ""


def run_pipeline(args, name: str, source_info: Dict[str, float]) -> BenchmarkResult:
    spec = PIPELINES[name]
    output = args.output_dir / (name + ".mp4")
    log_path = args.output_dir / (name + ".log")
    command = [
        str(args.backend), str(args.input), "-o", str(output),
        "--sr-engine", spec["sr"], "--fi-engine", spec["fi"],
        "--sr-quality", spec["sr_quality"], "--fi-quality", "balanced",
        "--scale", "2", "--fi-multiplier", "2",
        "--codec", args.codec, "--preset", args.preset, "--crf", str(args.crf),
        "--start", str(args.start), "--duration", str(args.duration),
        "--no-audio", "--overwrite", "--progress-json",
    ]
    if args.torch_python:
        command.extend(["--torch-python", str(args.torch_python)])
    if args.ncnn_gpu is not None:
        command.extend(["--ncnn-gpu", str(args.ncnn_gpu)])

    output.unlink(missing_ok=True)
    started = time.perf_counter()
    events: Dict[str, float] = {}
    lines: List[str] = []
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    environment = dict(os.environ)
    environment["LVE_DISABLE_FUSED_NCNN"] = spec.get(
        "disable_native_ncnn", "0")
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        creationflags=flags, env=environment)

    def read_output() -> None:
        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            for line in process.stdout:
                elapsed = time.perf_counter() - started
                lines.append(line.rstrip())
                log.write("[%.3f] %s" % (elapsed, line))
                if "批处理:" in line or "Batch:" in line:
                    events.setdefault("ready", elapsed)

    reader = threading.Thread(target=read_output, name="benchmark-log", daemon=True)
    reader.start()
    cpu_sampler = SystemCpuSampler()
    samples: List[Dict[str, float]] = []
    last_status = started
    while process.poll() is None:
        time.sleep(args.sample_interval)
        now = time.perf_counter()
        sample = {"time": now - started, "cpu": cpu_sampler.sample()}
        sample.update(sample_nvidia())
        samples.append(sample)
        if now - last_status >= 5:
            print("  %-28s %6.1fs GPU %4.0f%% CPU %4.0f%%" % (
                name, now - started, sample.get("gpu", 0.0), sample["cpu"]),
                flush=True)
            last_status = now
    return_code = process.wait()
    reader.join(timeout=5)
    wall_seconds = time.perf_counter() - started
    ready = events.get("ready", 0.0)
    active_samples = [sample for sample in samples
                      if sample["time"] >= ready] if ready else samples
    expected_input = int(round(args.duration * float(source_info["fps"])))

    output_info: Dict[str, float] = {}
    error = ""
    if return_code == 0 and output.is_file():
        try:
            output_info = probe_video(output)
        except Exception as exc:  # pragma: no cover - diagnostic path
            error = "output probe failed: %s" % exc
    else:
        error = "\n".join(lines[-12:])

    processing_seconds = max(0.001, wall_seconds - ready)
    output_frames = int(output_info.get("total_frames", 0))
    result = BenchmarkResult(
        name=name,
        label=spec["label"],
        return_code=return_code,
        wall_seconds=wall_seconds,
        init_seconds=ready,
        processing_seconds=processing_seconds,
        input_frames=expected_input,
        output_frames=output_frames,
        output_width=int(output_info.get("width", 0)),
        output_height=int(output_info.get("height", 0)),
        output_fps=float(output_info.get("fps", 0.0)),
        output_mib=(output.stat().st_size / (1024 * 1024)
                    if output.is_file() else 0.0),
        input_fps_effective=expected_input / processing_seconds,
        output_fps_effective=output_frames / processing_seconds,
        realtime_factor=args.duration / wall_seconds,
        cpu_avg=_mean(sample.get("cpu", 0.0) for sample in active_samples),
        cpu_peak=_maximum(sample.get("cpu", 0.0) for sample in active_samples),
        gpu_avg=_mean(sample.get("gpu", 0.0) for sample in active_samples),
        gpu_peak=_maximum(sample.get("gpu", 0.0) for sample in active_samples),
        encoder_avg=_mean(sample.get("encoder", 0.0) for sample in active_samples),
        encoder_peak=_maximum(sample.get("encoder", 0.0) for sample in active_samples),
        decoder_avg=_mean(sample.get("decoder", 0.0) for sample in active_samples),
        memory_peak_mib=_maximum(
            sample.get("memory_mib", 0.0) for sample in active_samples),
        power_avg_w=_mean(sample.get("power_w", 0.0) for sample in active_samples),
        log_path=str(log_path),
        output_path=str(output),
        error=error,
    )
    print("  %-28s done: %.2f input fps, %.1f%% GPU, %.1f%% CPU" % (
        name, result.input_fps_effective, result.gpu_avg, result.cpu_avg),
        flush=True)
    return result


def write_reports(args, source_info: Dict[str, float],
                  results: List[BenchmarkResult]) -> None:
    payload = {
        "input": str(args.input),
        "source": source_info,
        "start": args.start,
        "duration": args.duration,
        "codec": args.codec,
        "preset": args.preset,
        "crf": args.crf,
        "sample_interval": args.sample_interval,
        "results": [asdict(result) for result in results],
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline = next((item for item in results if item.name == args.baseline), None)
    lines = [
        "# Video pipeline benchmark",
        "",
        "- Input: `%s`" % args.input,
        "- Segment: %.3f–%.3f s" % (args.start, args.start + args.duration),
        "- Source: %dx%d @ %.3f fps, %d frames total" % (
            source_info["width"], source_info["height"], source_info["fps"],
            source_info["total_frames"]),
        "- Output target: 2× resolution, 2× frame rate, `%s` / `%s` / CQ %d" % (
            args.codec, args.preset, args.crf),
        "",
        "| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | "
        "CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        relative = (
            baseline.wall_seconds / item.wall_seconds
            if baseline and item.wall_seconds else 0.0)
        lines.append(
            "| %s | %.2f | %.2f | %.2f | %.2f | %.1f/%.1f%% | %.1f/%.1f%% | "
            "%.0f | %.1f | %.2fx | %s |" % (
                item.label, item.wall_seconds, item.init_seconds,
                item.input_fps_effective, item.output_fps_effective,
                item.gpu_avg, item.gpu_peak, item.cpu_avg, item.cpu_peak,
                item.memory_peak_mib, item.power_avg_w, relative,
                "OK" if item.return_code == 0 and not item.error else "FAILED"))
    lines.extend(["", "Raw samples and paths are stored in `results.json`.", ""])
    (args.output_dir / "RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--torch-python", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=float, default=4.0)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--codec", default="hevc_nvenc")
    parser.add_argument("--preset", default="p5")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--ncnn-gpu", type=int)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--baseline", default="rife_nvvfx")
    parser.add_argument(
        "--pipelines", nargs="+", choices=sorted(PIPELINES),
        default=["rife_nvvfx", "rife_ncnn_realcugan_native",
                 "rife_ncnn_realcugan_cli",
                 "rife_ncnn_realesrgan_native",
                 "rife_ncnn_realesrgan_cli"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.backend = args.backend.resolve()
    args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.torch_python:
        args.torch_python = args.torch_python.resolve()
    if not args.backend.is_file():
        raise FileNotFoundError(args.backend)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_info = probe_video(args.input)
    results: List[BenchmarkResult] = []
    for name in args.pipelines:
        print("Running %s: %s" % (name, PIPELINES[name]["label"]), flush=True)
        results.append(run_pipeline(args, name, source_info))
        write_reports(args, source_info, results)
    print("Report: %s" % (args.output_dir / "RESULTS.md"))


if __name__ == "__main__":
    main()

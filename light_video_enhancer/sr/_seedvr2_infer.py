"""Persistent SeedVR2 low-VRAM inference worker."""

import argparse
import contextlib
import os
import shutil
import sys
import tempfile
import traceback
import zipfile

if os.name == "nt":
    # A native CUDA/extension failure must close the worker pipe and return an
    # error to the GUI, not leave a modal python.exe crash dialog behind.
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from _shared_frames import read_framed, write_framed


def _read():
    return read_framed(sys.stdin.buffer)


def _write(value) -> None:
    write_framed(sys.stdout.buffer, value)


def _arguments(model_dir, dit_model, model_family, quality, count, total_vram):
    is_7b = model_family == "7b"
    batch = 5 if is_7b else max(
        5, ((int(count) - 1) // 4) * 4 + 1)
    tile = 384 if is_7b and total_vram <= 14 * 1024 ** 3 else (
        512 if total_vram <= 14 * 1024 ** 3 else 768)
    if is_7b:
        swap = 36 if total_vram <= 16 * 1024 ** 3 else (
            30 if total_vram <= 24 * 1024 ** 3 else 16)
    else:
        swap = 32 if total_vram <= 12 * 1024 ** 3 else (
            28 if total_vram <= 16 * 1024 ** 3 else 16)
    color = {
        "fast": "lab",
        "balanced": "lab",
        "quality": "wavelet_adaptive",
        "ultra": "wavelet_adaptive",
    }.get(quality, "lab")
    return argparse.Namespace(
        cache_dit=True, cache_vae=True,
        dit_offload_device="cpu", vae_offload_device="cpu",
        tensor_offload_device="cpu",
        compile_dit=False, compile_vae=False,
        compile_backend="inductor", compile_mode="default",
        compile_fullgraph=False, compile_dynamic=False,
        compile_dynamo_cache_size_limit=64,
        compile_dynamo_recompile_limit=128,
        model_dir=model_dir,
        dit_model=dit_model,
        blocks_to_swap=swap, swap_io_components=True,
        vae_encode_tiled=True, vae_encode_tile_size=tile,
        vae_encode_tile_overlap=64,
        vae_decode_tiled=True, vae_decode_tile_size=tile,
        vae_decode_tile_overlap=64, tile_debug="false",
        attention_mode="sdpa",
        resolution=1080, max_resolution=0,
        batch_size=batch, uniform_batch_size=True,
        seed=42, prepend_frames=0, temporal_overlap=0,
        input_noise_scale=0.0, latent_noise_scale=0.0,
        color_correction=color,
    )


def _load_runtime(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    required = (
        "torchvision", "safetensors", "psutil", "einops", "omegaconf",
        "diffusers", "peft", "rotary_embedding_torch", "gguf")
    missing = []
    import importlib.util
    for name in required:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "SeedVR2 Python dependencies are missing: " + ", ".join(missing))
    runtime_dir = tempfile.mkdtemp(prefix="lve_seedvr2_runtime_")
    with zipfile.ZipFile(args["runtime"], "r") as bundle:
        bundle.extractall(runtime_dir)
    sys.path.insert(0, runtime_dir)
    old_cwd = os.getcwd()
    os.chdir(runtime_dir)
    try:
        # The upstream runtime prints status lines during import. stdout is
        # reserved for framed IPC, so route third-party text to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            import inference_cli
    finally:
        os.chdir(old_cwd)
    inference_cli.debug.enabled = False
    return inference_cli, runtime_dir


def _process(module, cache, args, request, total_vram):
    paths = [
        os.path.join(request["input_dir"], name)
        for name in sorted(os.listdir(request["input_dir"]))
        if name.lower().endswith(".png")
    ]
    count = int(request["count"])
    if len(paths) != count:
        raise RuntimeError(
            "SeedVR2 input count mismatch: %d/%d" % (len(paths), count))
    frames = []
    for path in paths:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Cannot read SeedVR2 input: " + path)
        frames.append(frame)
    tensor = torch.from_numpy(
        np.ascontiguousarray(np.stack(frames))).to(torch.float16).div_(255)
    options = _arguments(
        args["model_dir"], args["dit_model"], args.get("model_family", "3b"),
        args.get("quality"), count, total_vram)
    options.resolution = min(
        int(args["dst_width"]), int(args["dst_height"]))
    options.max_resolution = max(
        int(args["dst_width"]), int(args["dst_height"]))
    with torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
        result = module._single_gpu_direct_processing(
            tensor, options, "0", cache)
    if result.shape[0] < count:
        raise RuntimeError(
            "SeedVR2 output count mismatch: %d/%d" %
            (result.shape[0], count))
    output_dir = request["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    width, height = int(args["dst_width"]), int(args["dst_height"])
    result = result[:count].float().clamp_(0, 1).mul_(255).byte().numpy()
    for index, frame in enumerate(result):
        if frame.shape[1::-1] != (width, height):
            frame = cv2.resize(
                frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        if not cv2.imwrite(
                os.path.join(output_dir, "%08d.png" % index), frame,
                [cv2.IMWRITE_PNG_COMPRESSION, 0]):
            raise RuntimeError("Cannot write SeedVR2 output")
    del tensor, result, frames
    return {"count": count}


def main() -> None:
    runtime_dir = None
    try:
        args = _read()
        module, runtime_dir = _load_runtime(args)
        cache = {}
        properties = torch.cuda.get_device_properties(0)
        total_vram = int(properties.total_memory)
        required_vram = int(float(args.get("min_vram_gib", 0)) * 1024 ** 3)
        if total_vram < required_vram:
            raise RuntimeError(
                "SeedVR2 %s requires at least %.1f GiB VRAM; detected %.1f GiB" %
                (args.get("model_family", "model"),
                 required_vram / 1024 ** 3, total_vram / 1024 ** 3))
        _write({
            "ready": True,
            "gpu_name": properties.name,
            "total_vram": total_vram,
        })
        while True:
            try:
                request = _read()
            except EOFError:
                break
            try:
                _write(_process(
                    module, cache, args, request, total_vram))
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                _write({"error": str(exc)})
                break
    except Exception as exc:
        _write({"error": str(exc)})
    finally:
        if runtime_dir:
            shutil.rmtree(runtime_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

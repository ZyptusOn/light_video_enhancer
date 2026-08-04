"""Persistent official SparkVSR Stage-2 worker."""

import contextlib
import os
import sys
import traceback

if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from _shared_frames import read_framed, write_framed


def _read():
    return read_framed(sys.stdin.buffer)


def _write(value) -> None:
    write_framed(sys.stdout.buffer, value)


def _physical_memory() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        if os.name != "nt":
            return 0
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        value = MEMORYSTATUSEX()
        value.dwLength = ctypes.sizeof(value)
        return int(value.ullTotalPhys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) else 0


def _image_paths(path):
    extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    if not path:
        return []
    if os.path.isdir(path):
        return [os.path.join(path, name) for name in sorted(os.listdir(path))
                if name.lower().endswith(extensions)]
    if path.lower().endswith(extensions):
        return [path]
    raise ValueError(
        "SparkVSR reference path must be an image or a directory of images")


def _load_references(args):
    paths = _image_paths(args.get("reference_path", ""))
    indices = [int(item) for item in args.get("reference_indices", [])]
    if len(paths) != len(indices):
        raise ValueError(
            "SparkVSR reference image/index count mismatch: %d/%d" %
            (len(paths), len(indices)))
    result = []
    expected = (int(args["dst_width"]), int(args["dst_height"]))
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if rgb.size != expected:
                raise ValueError(
                    "SparkVSR reference must match the 4x output size %dx%d: %s is %dx%d" %
                    (expected[0], expected[1], path, rgb.width, rgb.height))
            array = np.asarray(rgb, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        result.append(tensor)
    return indices, result


def _load_runtime(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    total_ram = _physical_memory()
    if total_vram < 11 * 1024 ** 3:
        raise RuntimeError(
            "SparkVSR needs at least 11 GiB VRAM even with sequential CPU offload; detected %.1f GiB" %
            (total_vram / 1024 ** 3))
    cpu_offload = total_vram < 40 * 1024 ** 3
    if (cpu_offload and total_ram and total_ram < 56 * 1024 ** 3 and
            os.environ.get("LVE_ALLOW_HEAVY_OFFLOAD") != "1"):
        raise RuntimeError(
            "SparkVSR Stage-2 contains 42.2 GB of weights. This safety gate requires "
            "at least 56 GiB system RAM when GPU VRAM is below 40 GiB; detected "
            "%.1f GiB RAM / %.1f GiB VRAM. Set LVE_ALLOW_HEAVY_OFFLOAD=1 only if "
            "you accept paging and system-instability risk." %
            (total_ram / 1024 ** 3, total_vram / 1024 ** 3))
    runtime = args["runtime"]
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    from sparkvsr_wrapper.model_loader import load_sparkvsr_pipeline
    pipe, empty_prompt = load_sparkvsr_pipeline(
        args["model_path"], dtype_str="bfloat16", cpu_offload=cpu_offload,
        vae_slicing=True, vae_tiling=True, device="cuda")
    ref_indices, ref_frames = _load_references(args)
    return pipe, empty_prompt, ref_indices, ref_frames, cpu_offload


def _load_video(paths):
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())
    return torch.from_numpy(np.stack(frames)).float().div_(255)


def _process(state, args, request):
    from sparkvsr_wrapper.infer import run_sparkvsr
    from sparkvsr_wrapper.preprocess import (
        preprocess_frames, remove_padding_and_extra_frames)
    paths = _image_paths(request["input_dir"])
    expected = int(request["input_count"])
    if len(paths) != expected:
        raise RuntimeError("SparkVSR input count mismatch: %d/%d" % (len(paths), expected))
    if not paths:
        return {"count": 0}
    pipe, empty_prompt, global_indices, global_refs, _ = state
    batch_start = int(request.get("batch_start", 0))
    batch_end = batch_start + expected
    pairs = [(idx - batch_start, ref) for idx, ref in zip(global_indices, global_refs)
             if batch_start <= idx < batch_end]
    local_indices = [item[0] for item in pairs]
    local_refs = [item[1] for item in pairs]
    video, _, pad_f, pad_h, pad_w, _ = preprocess_frames(
        _load_video(paths), upscale=4)
    quality = args.get("quality", "quality")
    tile = {"fast": 256, "balanced": 384, "quality": 512, "ultra": 0}.get(quality, 512)
    torch.manual_seed(0)
    with torch.no_grad():
        output = run_sparkvsr(
            pipe=pipe, video=video,
            ref_frames_list=local_refs, ref_indices=local_indices,
            chunk_len=0, overlap_t=0,
            tile_size_hw=(tile, tile) if tile else (0, 0),
            overlap_hw=(64, 64),
            ref_guidance_scale=float(args.get("reference_guidance", 1.0)),
            noise_step=0, sr_noise_step=399, prompt="",
            empty_prompt_embedding=empty_prompt)
    output = remove_padding_and_extra_frames(output, pad_f, pad_h, pad_w)
    if output.shape[2] != expected:
        raise RuntimeError("SparkVSR output count mismatch: %d/%d" % (output.shape[2], expected))
    frames = output[0].detach().cpu().clamp_(0, 1)
    frames = frames.permute(1, 2, 3, 0).mul_(255).round_().byte().numpy()
    os.makedirs(request["output_dir"], exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame, mode="RGB").save(
            os.path.join(request["output_dir"], "%08d.png" % index), compress_level=0)
    return {"count": expected, "references": local_indices}


def main() -> None:
    try:
        args = _read()
        with contextlib.redirect_stdout(sys.stderr):
            state = _load_runtime(args)
        _write({
            "ready": True, "gpu_name": torch.cuda.get_device_name(0),
            "cpu_offload": state[-1], "reference_count": len(state[2]),
        })
        while True:
            try:
                request = _read()
            except EOFError:
                break
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = _process(state, args, request)
                _write(result)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                _write({"error": str(exc)})
                break
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _write({"error": str(exc)})


if __name__ == "__main__":
    main()

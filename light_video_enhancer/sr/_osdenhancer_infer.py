"""Persistent OSDEnhancer joint 4x spatial / 2x temporal worker."""

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


def _load_runtime(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    if total_vram < 79 * 1024 ** 3:
        raise RuntimeError(
            "OSDEnhancer follows the author's >=80 GB VRAM requirement; "
            "detected %.1f GiB" % (total_vram / 1024 ** 3))
    runtime = args["runtime"]
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    from pipeline.OSDEnhancer_pipeline import OSDEnhancerPipeline
    pipe = OSDEnhancerPipeline.from_pretrained(
        args["checkpoint_path"], torch_dtype=torch.bfloat16,
        device="cuda", local_files_only=True)
    return pipe


def _image_paths(directory):
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def _load_video(paths):
    frames = []
    for path in paths:
        with Image.open(path) as image:
            value = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        frames.append(torch.from_numpy(value).permute(2, 0, 1))
    return torch.stack(frames).float().div_(255).unsqueeze(0)


def _process(pipe, args, request):
    paths = _image_paths(request["input_dir"])
    if len(paths) != int(request["input_count"]):
        raise RuntimeError(
            "OSDEnhancer input count mismatch: %d/%d" %
            (len(paths), int(request["input_count"])))
    if not paths:
        return {"count": 0}
    requested_count = (len(paths) - 1) * 2 + 1
    repeated_single = len(paths) == 1
    if repeated_single:
        paths.append(paths[0])
    video = _load_video(paths).to(device="cuda", dtype=torch.float32)
    with torch.no_grad():
        output = pipe(
            input=video, spatial_scale=4, temporal_scale=2,
            chunk_length=None, overlap=None)
    if repeated_single:
        output = output[:, :1]
    if output.shape[1] != requested_count:
        raise RuntimeError(
            "OSDEnhancer output count mismatch: %d/%d" %
            (output.shape[1], requested_count))
    output = output[0].detach().cpu().clamp_(0, 1)
    output = output.permute(0, 2, 3, 1).mul_(255).round_().byte().numpy()
    os.makedirs(request["output_dir"], exist_ok=True)
    for index, frame in enumerate(output):
        Image.fromarray(frame, mode="RGB").save(
            os.path.join(request["output_dir"], "%08d.png" % index),
            compress_level=0)
    return {"count": requested_count}


def main() -> None:
    try:
        args = _read()
        with contextlib.redirect_stdout(sys.stderr):
            pipe = _load_runtime(args)
        _write({
            "ready": True,
            "gpu_name": torch.cuda.get_device_name(0),
        })
        while True:
            try:
                request = _read()
            except EOFError:
                break
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = _process(pipe, args, request)
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

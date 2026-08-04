"""Persistent optional DLoRAL one-step video super-resolution worker."""

import contextlib
import os
import sys
import traceback
from types import SimpleNamespace

if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)

import numpy as np
import torch
import torch.nn.functional as F
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
    if total_vram < 11 * 1024 ** 3:
        raise RuntimeError(
            "DLoRAL requires at least 11 GiB VRAM; detected %.1f GiB" %
            (total_vram / 1024 ** 3))

    runtime = args["runtime"]
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    os.environ["LVE_DLORAL_SPYNET"] = args["spynet_path"]
    from src.DLoRAL_model import Generator_eval

    options = SimpleNamespace(
        pretrained_path=args["checkpoint_path"],
        pretrained_model_path=args["sd_path"],
        pretrained_model_name_or_path=args["sd_path"],
        vae_decoder_tiled_size=192,
        vae_encoder_tiled_size=768,
        latent_tiled_size=64 if total_vram <= 14 * 1024 ** 3 else 96,
        latent_tiled_overlap=24 if total_vram <= 14 * 1024 ** 3 else 32,
        load_cfr=True,
        merge_and_unload_lora=False,
    )
    model = Generator_eval(options)
    model.set_eval()
    quality = args.get("quality", "quality")
    if quality in {"fast", "balanced"}:
        stage = 0
        adapters = [
            "default_encoder_consistency",
            "default_decoder_consistency",
            "default_others_consistency",
        ]
    else:
        stage = 1
        adapters = [
            "default_encoder_quality", "default_decoder_quality",
            "default_others_quality", "default_encoder_consistency",
            "default_decoder_consistency", "default_others_consistency",
        ]
    model.unet.set_adapter(adapters)
    return model, stage


def _image_paths(directory):
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def _load_rgb(path, width, height):
    with Image.open(path) as image:
        value = image.convert("RGB").resize(
            (width, height), Image.Resampling.LANCZOS)
        return np.asarray(value, dtype=np.uint8).copy()


def _uncertainty_mask(previous, current):
    pair = torch.stack((previous, current), dim=0)
    gray = pair.mul(torch.tensor(
        [0.299, 0.587, 0.114], dtype=pair.dtype,
        device=pair.device).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    gray = F.interpolate(gray, scale_factor=0.125, mode="bilinear",
                         align_corners=False)
    variance = gray.var(dim=0)
    threshold = variance.mean()
    return (variance >= threshold).to(dtype=pair.dtype).unsqueeze(0).unsqueeze(0)


def _run_pair(model, stage, previous, current, prompt):
    tensors = []
    for frame in (previous, current):
        tensor = torch.from_numpy(np.ascontiguousarray(frame)).to(
            device="cuda", dtype=torch.float16)
        tensors.append(tensor.permute(2, 0, 1).div_(127.5).sub_(1))
    pair = torch.stack(tensors, dim=0)
    mask = _uncertainty_mask(
        pair[0].add(1).mul(0.5), pair[1].add(1).mul(0.5))
    with torch.inference_mode():
        output, _, _, _, _ = model(
            stages=stage, c_t=pair.unsqueeze(0),
            uncertainty_map=mask, prompt=prompt,
            weight_dtype=torch.float16)
    frame = output[0].float().add_(1).mul_(127.5).clamp_(0, 255)
    return frame.byte().permute(1, 2, 0).cpu().numpy()


def _process(model, stage, args, request):
    paths = _image_paths(request["input_dir"])
    count = int(request["count"])
    has_history = bool(request.get("has_history"))
    expected = count + (1 if has_history else 0)
    if len(paths) != expected:
        raise RuntimeError(
            "DLoRAL input count mismatch: %d/%d" % (len(paths), expected))
    width, height = int(args["dst_width"]), int(args["dst_height"])
    frames = [_load_rgb(path, width, height) for path in paths]
    if has_history:
        pairs = [(frames[index], frames[index + 1])
                 for index in range(count)]
    else:
        pairs = [(frames[0], frames[0])]
        pairs.extend((frames[index - 1], frames[index])
                     for index in range(1, count))
    prompt = (
        "high quality, detailed, clean, natural, temporally consistent video,")
    os.makedirs(request["output_dir"], exist_ok=True)
    for index, (previous, current) in enumerate(pairs):
        result = _run_pair(model, stage, previous, current, prompt)
        Image.fromarray(result, mode="RGB").save(
            os.path.join(request["output_dir"], "%08d.png" % index),
            compress_level=0)
    return {"count": count}


def main() -> None:
    try:
        args = _read()
        with contextlib.redirect_stdout(sys.stderr):
            model, stage = _load_runtime(args)
        _write({
            "ready": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "stage": stage,
        })
        while True:
            try:
                request = _read()
            except EOFError:
                break
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = _process(model, stage, args, request)
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

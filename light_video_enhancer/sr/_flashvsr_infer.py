"""Persistent optional FlashVSR v1.1 Tiny Long worker."""

import contextlib
import io
import os
import sys
import traceback
import zipfile

if os.name == "nt":
    # Suppress Windows Error Reporting UI for isolated native extensions; the
    # parent process reports the closed worker pipe in its normal error path.
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
    runtime = args["runtime"]
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    try:
        import block_sparse_attn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "block_sparse_attn is not installed in this Python environment") from exc

    from diffsynth.models import ModelManager
    from diffsynth.pipelines.flashvsr_tiny_long import FlashVSRTinyLongPipeline
    from flashvsr_utils.TCDecoder import build_tcdecoder
    from flashvsr_utils.utils import Causal_LQ4x_Proj

    model_dir = args["model_dir"]
    model_path = os.path.join(
        model_dir, "diffusion_pytorch_model_streaming_dmd.safetensors")
    manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    manager.load_models([model_path])
    pipe = FlashVSRTinyLongPipeline.from_model_manager(
        manager, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
        in_dim=3, out_dim=1536, layer_num=1).to(
            "cuda", dtype=torch.bfloat16)
    pipe.denoising_model().LQ_proj_in.load_state_dict(
        torch.load(
            os.path.join(model_dir, "LQ_proj_in.ckpt"),
            map_location="cpu"),
        strict=True)
    pipe.denoising_model().LQ_proj_in.to("cuda")
    pipe.TCDecoder = build_tcdecoder(
        new_channels=[512, 256, 128, 128],
        new_latent_channels=16 + 768)
    pipe.TCDecoder.load_state_dict(
        torch.load(
            os.path.join(model_dir, "TCDecoder.ckpt"),
            map_location="cpu"),
        strict=False)
    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    with zipfile.ZipFile(runtime, "r") as bundle:
        prompt = torch.load(io.BytesIO(
            bundle.read("flashvsr_assets/posi_prompt.pth")),
            map_location="cpu")
    pipe.init_cross_kv(context_tensor=prompt)
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def _quality_settings(quality):
    return {
        "fast": (1.5, 9),
        "balanced": (2.0, 9),
        "quality": (2.0, 11),
        "ultra": (2.5, 11),
    }.get(quality, (2.0, 9))


def _image_paths(directory):
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def _prepare_video(paths, width, height):
    padded_width = ((width + 127) // 128) * 128
    padded_height = ((height + 127) // 128) * 128
    frames = []
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(
                image.convert("RGB").resize(
                    (width, height), Image.Resampling.BICUBIC),
                dtype=np.uint8)
        if padded_width != width or padded_height != height:
            rgb = np.pad(
                rgb,
                ((0, padded_height - height),
                 (0, padded_width - width), (0, 0)),
                mode="edge")
        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb)).permute(2, 0, 1)
        tensor = tensor.to(dtype=torch.bfloat16).div_(127.5).sub_(1)
        frames.append(tensor)
    frames.extend([frames[-1]] * 4)
    video = torch.stack(frames, dim=0).permute(
        1, 0, 2, 3).unsqueeze(0)
    return video, padded_width, padded_height


def _process(pipe, args, request):
    paths = _image_paths(request["input_dir"])
    count = int(request["count"])
    if len(paths) != count:
        raise RuntimeError(
            "FlashVSR input count mismatch: %d/%d" % (len(paths), count))
    width, height = int(args["dst_width"]), int(args["dst_height"])
    video, padded_width, padded_height = _prepare_video(
        paths, width, height)
    sparse_ratio, local_range = _quality_settings(args.get("quality"))
    with torch.inference_mode():
        result = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0,
            num_inference_steps=1, seed=0,
            LQ_video=video, num_frames=count + 4,
            height=padded_height, width=padded_width,
            is_full_block=False, if_buffer=True,
            topk_ratio=(
                sparse_ratio * 768 * 1280 /
                (padded_height * padded_width)),
            kv_ratio=3.0, local_range=local_range,
            color_fix=True)
    if result.shape[1] != count:
        raise RuntimeError(
            "FlashVSR output count mismatch: %d/%d" %
            (result.shape[1], count))
    os.makedirs(request["output_dir"], exist_ok=True)
    frames = result[:, :, :height, :width].permute(
        1, 2, 3, 0).float().add_(1).mul_(127.5)
    frames = frames.clamp_(0, 255).byte().cpu().numpy()
    for index, frame in enumerate(frames):
        Image.fromarray(frame, mode="RGB").save(
            os.path.join(request["output_dir"], "%08d.png" % index),
            compress_level=0)
    del video, result, frames
    torch.cuda.empty_cache()
    return {"count": count}


def main() -> None:
    try:
        args = _read()
        # DiffSynth writes model-loading and pipeline status to stdout. Keep
        # stdout exclusively for framed IPC or text corrupts binary replies.
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
        _write({"error": str(exc)})


if __name__ == "__main__":
    main()

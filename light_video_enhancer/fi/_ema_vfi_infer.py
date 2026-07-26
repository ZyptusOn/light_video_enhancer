"""Persistent external-Python worker for EMA-VFI Small."""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
for directory in (HERE, PACKAGE_ROOT):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from _ema_vfi_vendor import EMAVFISmall
from _shared_frames import SharedNDArray, read_framed, write_framed


def _read():
    return read_framed(sys.stdin.buffer)


def _write(value) -> None:
    write_framed(sys.stdout.buffer, value)


def _to_tensor(frame, device, fp16):
    tensor = torch.from_numpy(frame).to(device, non_blocking=True)
    tensor = tensor.permute(2, 0, 1).flip(0).unsqueeze(0).contiguous()
    return tensor.half().div_(255.0) if fp16 else tensor.float().div_(255.0)


def _load(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected PyTorch environment")
    path = args.get("model_path", "")
    if not os.path.isfile(path):
        raise FileNotFoundError("EMA-VFI model does not exist: %s" % path)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    model = EMAVFISmall()
    model.load_official_checkpoint(path)
    model = model.to(device).eval()
    fp16 = bool(args.get("fp16"))
    if fp16:
        model.half()
    return model, device, fp16


def _process(model, device, fp16, args, request,
             shared_input, shared_output):
    frame0, frame1 = shared_input.array[0], shared_input.array[1]
    input0 = _to_tensor(frame0, device, fp16)
    input1 = _to_tensor(frame1, device, fp16)
    pad_w, pad_h = int(request["pad_w"]), int(request["pad_h"])
    if pad_w or pad_h:
        input0 = F.pad(input0, (0, pad_w, 0, pad_h))
        input1 = F.pad(input1, (0, pad_w, 0, pad_h))
    with torch.inference_mode():
        predictions = model(
            input0, input1, request["timesteps"],
            down_scale=float(args.get("down_scale", 1.0)),
            tta=bool(args.get("tta", False)))
    height, width = frame0.shape[:2]
    for index, prediction in enumerate(predictions):
        bgr = (prediction[0, :, :height, :width].float().flip(0)
               .permute(1, 2, 0).clamp_(0, 1).mul_(255)
               .byte().contiguous())
        torch.from_numpy(shared_output.array[index]).copy_(bgr)
    return {"count": len(predictions)}


def main() -> None:
    shared_input = None
    shared_output = None
    try:
        args = _read()
        shared_input = SharedNDArray.attach(args["shared_input"])
        shared_output = SharedNDArray.attach(args["shared_output"])
        model, device, fp16 = _load(args)
        _write({"ready": True})
        while True:
            try:
                request = _read()
            except EOFError:
                break
            try:
                _write(_process(
                    model, device, fp16, args, request,
                    shared_input, shared_output))
            except Exception as exc:
                _write({"error": str(exc)})
                break
    except Exception as exc:
        _write({"error": str(exc)})
    finally:
        if shared_input is not None:
            shared_input.close()
        if shared_output is not None:
            shared_output.close()


if __name__ == "__main__":
    main()

"""Persistent external-Python RIFE worker with shared-memory fast path."""

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

from _rife_model import FlownetCas
from _shared_frames import SharedNDArray, read_framed, write_framed


def _read():
    return read_framed(sys.stdin.buffer)


def _write(value) -> None:
    write_framed(sys.stdout.buffer, value)


def _pack_array(value: np.ndarray) -> dict:
    array = np.ascontiguousarray(value)
    return {
        "__lve_array__": 1,
        "shape": tuple(int(part) for part in array.shape),
        "dtype": array.dtype.str,
        "data": array.tobytes(order="C"),
    }


def _unpack_array(value) -> np.ndarray:
    if not isinstance(value, dict) or value.get("__lve_array__") != 1:
        raise TypeError("invalid array message")
    shape = tuple(int(part) for part in value["shape"])
    result = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
    expected = int(np.prod(shape, dtype=np.int64))
    if result.size != expected:
        raise ValueError("array message length mismatch")
    return result.reshape(shape)


def _load_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("外部 Python 的 PyTorch CUDA 不可用")
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    # RIFE uses fixed shapes, but cuDNN search costs several seconds per job.
    torch.backends.cudnn.benchmark = False
    model = FlownetCas().to(device).eval()
    path = args.get("model_path", "")
    if not os.path.isfile(path):
        raise FileNotFoundError("模型权重不存在: %s" % path)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    fp16 = bool(args.get("fp16", True))
    if fp16:
        model.half()
    return device, model, fp16


def _to_tensor(device, frame, fp16):
    # Upload BGR once, then reorder channels on the GPU.
    tensor = torch.from_numpy(frame).to(device, non_blocking=True)
    tensor = tensor.permute(2, 0, 1).flip(0).unsqueeze(0).contiguous()
    return tensor.half().div_(255.0) if fp16 else tensor.float().div_(255.0)


def _process(device, model, fp16, frame0, frame1,
             timestep, pad_w, pad_h, scale, output=None):
    i0 = _to_tensor(device, frame0, fp16)
    i1 = _to_tensor(device, frame1, fp16)
    if pad_w or pad_h:
        i0 = F.pad(i0, (0, pad_w, 0, pad_h))
        i1 = F.pad(i1, (0, pad_w, 0, pad_h))
    with torch.inference_mode():
        pred = model.inference(i0, i1, timestep, scale)
    height, width = frame0.shape[:2]
    bgr_gpu = (pred[0, :, :height, :width].float().flip(0).permute(1, 2, 0)
               .clamp_(0, 1).mul_(255).byte().contiguous())
    if output is not None:
        torch.from_numpy(output).copy_(bgr_gpu)
        return None
    return bgr_gpu.cpu().numpy()


def _process_request(device, model, fp16, request, shared_input, shared_output):
    if not request:
        return []
    if isinstance(request, dict) and request.get("protocol") == 3:
        if shared_input is None or shared_output is None:
            raise RuntimeError("shared-memory buffers were not initialised")
        timesteps = request["timesteps"]
        for index, timestep in enumerate(timesteps):
            _process(device, model, fp16, shared_input.array[0], shared_input.array[1],
                     timestep, request["pad_w"], request["pad_h"], request["scale"],
                     output=shared_output.array[index])
        return {"shared": True, "count": len(timesteps)}
    if isinstance(request, dict) and request.get("protocol") == 2:
        frame0 = _unpack_array(request["frame0"])
        frame1 = _unpack_array(request["frame1"])
        return [_pack_array(_process(
            device, model, fp16, frame0, frame1, timestep,
            request["pad_w"], request["pad_h"], request["scale"]))
                for timestep in request["timesteps"]]
    return [_pack_array(_process(device, model, fp16, *task)) for task in request]


def main() -> None:
    shared_input = None
    shared_output = None
    try:
        args = _read()
        if args.get("ipc") == "shared_v1":
            shared_input = SharedNDArray.attach(args["shared_input"])
            shared_output = SharedNDArray.attach(args["shared_output"])
        first_request = _read()
        device, model, fp16 = _load_model(args)
        _write(_process_request(
            device, model, fp16, first_request, shared_input, shared_output))
    except Exception as exc:
        _write({"error": str(exc)})
        if shared_input is not None:
            shared_input.close()
        if shared_output is not None:
            shared_output.close()
        return

    try:
        while True:
            try:
                request = _read()
            except EOFError:
                break
            try:
                _write(_process_request(
                    device, model, fp16, request, shared_input, shared_output))
            except Exception as exc:
                _write({"error": str(exc)})
                break
    finally:
        if shared_input is not None:
            shared_input.close()
        if shared_output is not None:
            shared_output.close()


if __name__ == "__main__":
    main()

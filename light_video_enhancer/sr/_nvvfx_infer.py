"""Persistent NV-VFX inference worker with shared-memory fast path."""

import os
import sys

import numpy as np
import torch
from nvvfx import VideoSuperRes

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

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


def _run_effect(effect, bgr, output=None):
    # Upload BGR once; channel reorder and conversion happen on the GPU.
    tensor = torch.from_numpy(bgr).to(device="cuda", non_blocking=True)
    tensor = (tensor.permute(2, 0, 1).flip(0).contiguous()
              .to(torch.float32).div_(255.0))
    capsule = effect.run(tensor).image
    enhanced = torch.from_dlpack(capsule).clone()
    bgr_gpu = (enhanced.flip(0).permute(1, 2, 0).mul_(255.0)
               .clamp_(0.0, 255.0).to(torch.uint8).contiguous())
    if output is not None:
        torch.from_numpy(output).copy_(bgr_gpu)
        return None
    return bgr_gpu.cpu().numpy()


def main() -> None:
    effect = None
    shared_input = None
    shared_output = None
    try:
        args = _read()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        if args.get("ipc") == "shared_v1":
            shared_input = SharedNDArray.attach(args["shared_input"])
            shared_output = SharedNDArray.attach(args["shared_output"])
        effect = VideoSuperRes(quality=args["quality"])
        effect.output_width = args["dst_w"]
        effect.output_height = args["dst_h"]
        effect.load()
        _write({"ready": True, "shared": shared_input is not None})
        while True:
            try:
                request = _read()
            except EOFError:
                break
            try:
                if isinstance(request, dict) and request.get("protocol") == 2:
                    if shared_input is None or shared_output is None:
                        raise RuntimeError("shared-memory buffers were not initialised")
                    _run_effect(effect, shared_input.array, output=shared_output.array)
                    _write({"shared": True})
                else:
                    bgr = _unpack_array(request)
                    _write(_pack_array(_run_effect(effect, bgr)))
            except Exception as exc:
                _write({"error": str(exc)})
    except Exception as exc:
        _write({"error": str(exc)})
    finally:
        if effect is not None:
            effect.close()
        if shared_input is not None:
            shared_input.close()
        if shared_output is not None:
            shared_output.close()


if __name__ == "__main__":
    main()

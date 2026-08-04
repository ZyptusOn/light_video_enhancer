"""Persistent external-Python worker for VFIMamba."""
import contextlib
import os
import sys

if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
for directory in (HERE, PACKAGE_ROOT):
    if directory not in sys.path:
        sys.path.insert(0, directory)
from _shared_frames import SharedNDArray, read_framed, write_framed


def _read():
    return read_framed(sys.stdin.buffer)


def _write(value) -> None:
    write_framed(sys.stdout.buffer, value)


def _load(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected PyTorch environment")
    runtime, model_path = args.get("runtime", ""), args.get("model_path", "")
    if not os.path.isfile(runtime):
        raise FileNotFoundError("VFIMamba runtime does not exist: %s" % runtime)
    if not os.path.isfile(model_path):
        raise FileNotFoundError("VFIMamba model does not exist: %s" % model_path)
    sys.path.insert(0, runtime)
    with contextlib.redirect_stdout(sys.stderr):
        import config
        if args.get("variant") == "full":
            config.MODEL_CONFIG["LOGNAME"] = "VFIMamba"
            config.MODEL_CONFIG["MODEL_ARCH"] = config.init_model_config(F=32, depth=[2, 2, 2, 3, 3], M=False)
        else:
            config.MODEL_CONFIG["LOGNAME"] = "VFIMamba_S"
            config.MODEL_CONFIG["MODEL_ARCH"] = config.init_model_config(F=16, depth=[2, 2, 2, 3, 3], M=False)
        import Trainer_finetune
        from model.feature_extractor import VFIMAMBA_SCAN_BACKEND
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        model = Trainer_finetune.Model(-1)
        checkpoint = torch.load(model_path, map_location="cpu")
        state = Trainer_finetune.convert(checkpoint)
        if not state and isinstance(checkpoint, dict):
            state = checkpoint.get("state_dict", checkpoint)
        model.net.load_state_dict(state, strict=True)
        model.eval()
    return model, torch.device("cuda"), VFIMAMBA_SCAN_BACKEND


def _tensor(frame, device):
    return (torch.from_numpy(frame).to(device, non_blocking=True)
            .permute(2, 0, 1).flip(0).unsqueeze(0).float().div_(255.0))


def _process(model, device, args, request, shared_input, shared_output):
    frame0, frame1 = shared_input.array[0], shared_input.array[1]
    input0, input1 = _tensor(frame0, device), _tensor(frame1, device)
    height, width = frame0.shape[:2]
    pad_h, pad_w = ((height + 63) // 64) * 64 - height, ((width + 63) // 64) * 64 - width
    if pad_h or pad_w:
        input0 = F.pad(input0, (0, pad_w, 0, pad_h))
        input1 = F.pad(input1, (0, pad_w, 0, pad_h))
    scale, tta = float(args.get("flow_scale", 0.0)), bool(args.get("tta", False))
    with torch.inference_mode():
        predictions = [model.inference(input0, input1, local=True, TTA=tta,
                                       timestep=float(timestep), scale=scale)
                       for timestep in request["timesteps"]]
    for index, prediction in enumerate(predictions):
        bgr = (prediction[0, :, :height, :width].float().flip(0)
               .permute(1, 2, 0).clamp_(0, 1).mul_(255).byte().contiguous())
        torch.from_numpy(shared_output.array[index]).copy_(bgr)
    return {"count": len(predictions)}


def main() -> None:
    shared_input = shared_output = None
    try:
        args = _read()
        shared_input = SharedNDArray.attach(args["shared_input"])
        shared_output = SharedNDArray.attach(args["shared_output"])
        model, device, scan_backend = _load(args)
        _write({"ready": True, "scan_backend": scan_backend})
        while True:
            try:
                request = _read()
            except EOFError:
                break
            try:
                _write(_process(model, device, args, request, shared_input, shared_output))
            except Exception as exc:
                _write({"error": str(exc)})
                break
    except Exception as exc:
        try:
            _write({"error": str(exc)})
        except Exception:
            pass
    finally:
        if shared_input is not None:
            shared_input.close()
        if shared_output is not None:
            shared_output.close()


if __name__ == "__main__":
    main()

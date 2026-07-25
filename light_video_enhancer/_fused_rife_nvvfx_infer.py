"""CUDA Worker that keeps RIFE intermediates on GPU for NV-VFX."""

import os
import sys

import torch
import torch.nn.functional as F
from nvvfx import VideoSuperRes

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _shared_frames import SharedNDArray, read_framed, write_framed
from fi._rife_model import FlownetCas


def _read():
    return read_framed(sys.stdin.buffer)


def _write(value) -> None:
    write_framed(sys.stdout.buffer, value)


def _load_rife(path, device):
    model = FlownetCas().to(device).eval()
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    return model.half()


def _source_tensor(frame, device):
    tensor = torch.from_numpy(frame).to(device, non_blocking=True)
    return (tensor.permute(2, 0, 1).flip(0).unsqueeze(0).contiguous()
            .half().div_(255.0))


def _enhance_to_shared(effect, rgb_tensor, output) -> None:
    source = rgb_tensor[0].float().contiguous()
    capsule = effect.run(source).image
    enhanced = torch.from_dlpack(capsule).clone().clamp_(0.0, 1.0)
    red, green, blue = enhanced[0], enhanced[1], enhanced[2]
    # OpenCV/FFmpeg-compatible limited-range BT.601 I420. Producing YUV on
    # CUDA halves the download size and removes the CPU BGR->I420 pass.
    y_plane = 16.0 + 255.0 * (0.257 * red + 0.504 * green + 0.098 * blue)
    u_full = 128.0 + 255.0 * (-0.148 * red - 0.291 * green + 0.439 * blue)
    v_full = 128.0 + 255.0 * (0.439 * red - 0.368 * green - 0.071 * blue)
    u_plane = F.avg_pool2d(u_full[None, None], 2, 2)[0, 0]
    v_plane = F.avg_pool2d(v_full[None, None], 2, 2)[0, 0]
    i420 = torch.cat((y_plane.reshape(-1), u_plane.reshape(-1),
                      v_plane.reshape(-1)))
    i420 = i420.round_().clamp_(0.0, 255.0).to(torch.uint8).contiguous()
    torch.from_numpy(output).copy_(i420)


def _process(args, model, effect, device, inputs, outputs, count, skip_first,
             pair_modes):
    if count < 1 or count > inputs.shape[0]:
        raise ValueError("invalid fused input count")
    if len(pair_modes) != max(0, count - 1):
        raise ValueError("invalid fused pair-mode count")
    sources = [_source_tensor(inputs.array[index], device) for index in range(count)]
    padded = [F.pad(value, (0, args["pad_w"], 0, args["pad_h"]))
              if args["pad_w"] or args["pad_h"] else value for value in sources]
    output_index = 0
    if not skip_first:
        _enhance_to_shared(effect, sources[0], outputs.array[output_index])
        output_index += 1
    height, width = args["src_h"], args["src_w"]
    with torch.inference_mode():
        for pair in range(count - 1):
            for step in range(1, args["multiplier"]):
                timestep = step / args["multiplier"]
                pair_mode = int(pair_modes[pair])
                if pair_mode == 1:
                    prediction = sources[pair]
                elif pair_mode == 2:
                    prediction = (sources[pair] if step * 2 <= args["multiplier"]
                                  else sources[pair + 1])
                else:
                    prediction = model.inference(
                        padded[pair], padded[pair + 1], timestep, args["scale"])
                    prediction = prediction[:, :, :height, :width]
                _enhance_to_shared(effect, prediction, outputs.array[output_index])
                output_index += 1
            _enhance_to_shared(effect, sources[pair + 1], outputs.array[output_index])
            output_index += 1
    return output_index

def main() -> None:
    inputs = None
    outputs = None
    effect = None
    try:
        args = _read()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        if not os.path.isfile(args["model_path"]):
            raise FileNotFoundError("RIFE 模型不存在: %s" % args["model_path"])
        inputs = SharedNDArray.attach(args["shared_input"])
        outputs = SharedNDArray.attach(args["shared_output"])
        device = torch.device("cuda")
        torch.set_grad_enabled(False)
        # Algorithm search adds multi-second stalls for each batch shape.
        torch.backends.cudnn.benchmark = False
        model = _load_rife(args["model_path"], device)
        effect = VideoSuperRes(quality=args["quality"])
        effect.output_width = args["dst_w"]
        effect.output_height = args["dst_h"]
        effect.load()
        _write({"ready": True})
        while True:
            try:
                request = _read()
            except EOFError:
                break
            try:
                if request.get("command") != "process":
                    raise ValueError("unknown fused command")
                count = _process(
                    args, model, effect, device, inputs, outputs,
                    int(request["count"]), bool(request.get("skip_first", False)),
                    list(request.get("pair_modes", [])))
                _write({"count": count, "shared": True})
            except Exception as exc:
                _write({"error": str(exc)})
                break
    except Exception as exc:
        _write({"error": str(exc)})
    finally:
        if effect is not None:
            effect.close()
        if inputs is not None:
            inputs.close()
        if outputs is not None:
            outputs.close()


if __name__ == "__main__":
    main()

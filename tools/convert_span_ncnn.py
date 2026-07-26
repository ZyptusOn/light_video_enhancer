"""Convert official SPAN PyTorch checkpoints into dynamic-shape NCNN graphs.

This is a development/release tool, not a runtime dependency. It deliberately
loads the official ``span_arch.py`` from a caller-provided checkout so the
vendored application does not need to carry BasicSR or PyTorch.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import struct
import sys
import types
from pathlib import Path

import torch


class _Registry:
    def register(self):
        return lambda value: value


def _load_span_class(source: Path):
    basicsr = types.ModuleType("basicsr")
    utils = types.ModuleType("basicsr.utils")
    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = _Registry()
    sys.modules.setdefault("basicsr", basicsr)
    sys.modules.setdefault("basicsr.utils", utils)
    sys.modules["basicsr.utils.registry"] = registry
    spec = importlib.util.spec_from_file_location("lve_span_arch", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load SPAN architecture: %s" % source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SPAN, module.Conv3XC


def _externalize_preprocessing(
        param_path: Path, model_path: Path) -> None:
    """Move fixed RGB normalization out of the graph and weight stream."""
    lines = param_path.read_text(encoding="utf-8").splitlines()
    layer_count, blob_count = (int(value) for value in lines[1].split())
    body = lines[2:]
    if not (body[0].startswith("Input ") and
            body[1].startswith("MemoryData ") and
            body[2].startswith("BinaryOp ") and
            body[3].startswith("BinaryOp ")):
        raise RuntimeError("Unexpected SPAN preprocessing graph")
    del body[1:4]
    convolution_fields = body[1].split()
    convolution_fields[4] = "in0"
    body[1] = " ".join(convolution_fields)
    lines = [lines[0], "%d %d" % (
        layer_count - 3, blob_count - 3)] + body
    param_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # MemoryData is the first layer that consumes model bytes. Its RGB mean
    # is stored as three raw FP32 values, so remove those values when the
    # corresponding graph layer is removed or all following weights shift.
    weights = model_path.read_bytes()
    rgb_mean = struct.unpack("<3f", weights[:12])
    expected = (0.4488, 0.4371, 0.4040)
    if any(abs(left - right) > 1e-5
           for left, right in zip(rgb_mean, expected)):
        raise RuntimeError("Unexpected SPAN MemoryData weights")
    model_path.write_bytes(weights[12:])


def convert(source: Path, checkpoint: Path, output: Path) -> None:
    match = re.search(r"spanx([24])_ch(48|52)", checkpoint.stem, re.I)
    if not match:
        raise ValueError("Checkpoint name must contain spanx{2|4}_ch{48|52}")
    scale, channels = (int(value) for value in match.groups())
    span_class, reparameterized_conv = _load_span_class(source)
    model = span_class(
        3, 3, feature_channels=channels, upscale=scale).cpu().eval()
    payload = torch.load(
        str(checkpoint), map_location="cpu", weights_only=True)
    state = payload.get("params_ema", payload.get("params", payload))
    model.load_state_dict(state, strict=True)
    for layer in model.modules():
        if isinstance(layer, reparameterized_conv):
            layer.update_params()
    model.eval()

    output.mkdir(parents=True, exist_ok=True)
    stem = output / checkpoint.stem.lower()
    sample = torch.rand(1, 3, 64, 64)
    sample2 = torch.rand(1, 3, 80, 96)

    def portable(path: Path) -> str:
        return path.resolve().as_posix()

    import pnnx
    pnnx.export(
        model, portable(stem.with_suffix(".pt")),
        inputs=(sample,), inputs2=(sample2,),
        pnnxparam=portable(stem.with_suffix(".pnnx.param")),
        pnnxbin=portable(stem.with_suffix(".pnnx.bin")),
        pnnxpy=portable(Path(str(stem) + "_pnnx.py")),
        pnnxonnx=portable(stem.with_suffix(".pnnx.onnx")),
        ncnnparam=portable(stem.with_suffix(".param")),
        ncnnbin=portable(stem.with_suffix(".bin")),
        ncnnpy=portable(Path(str(stem) + "_ncnn.py")),
        # Repeated multiplicative attention amplifies FP16 weight error.
        # FP32 keeps the conversion faithful and the graphs remain tiny.
        fp16=False,
    )
    _externalize_preprocessing(
        stem.with_suffix(".param"), stem.with_suffix(".bin"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path,
                        help="Path to the official span_arch.py")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(args.source, args.checkpoint, args.output)


if __name__ == "__main__":
    main()

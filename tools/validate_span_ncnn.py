"""Compare converted SPAN NCNN graphs with their official PyTorch weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from convert_span_ncnn import _load_span_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--param", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()

    import ncnn

    stem = args.checkpoint.stem.lower()
    scale = int(stem.split("spanx", 1)[1][0])
    channels = int(stem.rsplit("ch", 1)[1])
    span_class, reparameterized_conv = _load_span_class(args.source)
    pytorch_model = span_class(
        3, 3, feature_channels=channels, upscale=scale).cpu().eval()
    payload = torch.load(
        str(args.checkpoint), map_location="cpu", weights_only=True)
    state = payload.get("params_ema", payload.get("params", payload))
    pytorch_model.load_state_dict(state, strict=True)
    for layer in pytorch_model.modules():
        if isinstance(layer, reparameterized_conv):
            layer.update_params()

    torch.manual_seed(20260726)
    source = torch.rand(1, 3, 48, 64, dtype=torch.float32)
    with torch.inference_mode():
        expected = pytorch_model(source).cpu().numpy()

    with ncnn.Net() as net:
        if net.load_param(str(args.param.resolve())) != 0:
            raise RuntimeError("NCNN param load failed")
        if net.load_model(str(args.model.resolve())) != 0:
            raise RuntimeError("NCNN model load failed")
        with net.create_extractor() as extractor:
            mean = np.asarray(
                [0.4488, 0.4371, 0.4040], dtype=np.float32
            )[:, None, None]
            prepared = (source.squeeze(0).numpy() - mean) * 255.0
            extractor.input(
                "in0", ncnn.Mat(prepared).clone())
            status, result = extractor.extract("out0")
            if status != 0:
                raise RuntimeError("NCNN inference failed: %s" % status)
            actual = np.asarray(result)[None, ...]

    delta = actual.astype(np.float32) - expected.astype(np.float32)
    mae = float(np.abs(delta).mean())
    maximum = float(np.abs(delta).max())
    signal = float(np.mean(expected.astype(np.float32) ** 2))
    noise = float(np.mean(delta ** 2))
    psnr = float("inf") if noise == 0 else 10.0 * np.log10(signal / noise)
    print("shape=%s mae=%.8f max=%.8f psnr=%.3f dB" %
          (actual.shape, mae, maximum, psnr))
    if mae > 0.05 or psnr < 50.0:
        print("SPAN NCNN conversion exceeds the validation threshold",
              file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

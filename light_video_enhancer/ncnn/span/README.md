# SPAN NCNN models

These graphs were converted from the official SPAN `spanx2/spanx4`
48-channel and 52-channel checkpoints with Tencent PNNX. The conversion first
reparameterizes `Conv3XC` exactly as the official evaluation path does and
keeps weights in FP32 to avoid attention-amplified quantisation error.

- Upstream: <https://github.com/hongyuanyu/SPAN>
- License: Apache-2.0 (`LICENSE.txt`)
- Converter: `tools/convert_span_ncnn.py`
- Numerical check: `tools/validate_span_ncnn.py`

At runtime the native worker uses Vulkan FP16 packing/storage with FP32
arithmetic and overlap-safe tiling on memory-constrained GPUs.

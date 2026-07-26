# 新算法集成与真实视频冒烟报告

测试日期：2026-07-26。测试设备为 NVIDIA GeForce RTX 5070 Ti Laptop GPU，输入为 `YUKI_Z.mp4` 的前 2 秒（1280×720、29.999 fps、60 帧），输出使用 H.264 NVENC `p1`，不复制音频。所有时间均为同一命令的端到端墙钟时间，包含解码、模型初始化、推理、编码和收尾，因此适合比较本项目的实际体验，不等于纯模型 FPS。

## 结果

| 管线 | 输出 | 端到端时间 | 输入吞吐 | 原生 worker 时间 |
|---|---:|---:|---:|---:|
| IFRNet S（fast），不超分 | 1280×720，119 帧 | 3.697 s | 16.23 fps | 0.879 s |
| SPAN x2 ch48（fast），不插帧 | 2560×1440，60 帧 | 12.915 s | 4.65 fps | 10.123 s |
| IFRNet S → SPAN x2 ch48 | 2560×1440，119 帧 | 19.335 s | 3.10 fps | 15.039 s |

此前同一视频条件下记录的 RIFE PyTorch + NVIDIA VFX 基线为 8.01 输入 fps；它来自上一轮吞吐基准，不与本表的端到端秒数混算。IFRNet S 本身非常轻，SPAN x2 是新组合的主要瓶颈。因此模型齐全时，“自动超分”继续优先选择实测更快的 Real-ESRGAN，SPAN 保留为用户明确选择的轻量模型。

## 正确性回归

首次真实视频冒烟发现 SPAN 输出帧数正确但画面全黑。原因有两处：转换后的 NCNN 图输出与官方 PyTorch 模型相同，是约 `[0,1]` 的 RGB 浮点值；原生 worker 却直接调用 `to_pixels`，几乎所有像素都被舍入为 0。同时，FFmpeg 管线传入 BGR，旧实现按 RGB 读取，颜色顺序也不正确。

修复后 worker 会：

1. 以 `BGR -> RGB` 读取输入；
2. 在输出端乘以 255 并饱和到字节范围；
3. 以 `RGB -> BGR` 写回共享内存。

修复后的 2 秒 SPAN 输出完整解码 60 帧，首帧均值/标准差为 `120.874 / 74.239`，不再是 `0 / 0`。将首帧缩回 720p 与源帧比较，平均绝对误差为 1.786 个 8-bit 像素值，B/G/R 通道均值顺序与源视频一致。`tests/test_native_ncnn.py` 增加了由 `LVE_NATIVE_SMOKE=1` 启用的真实 Vulkan 字节范围和 BGR 顺序测试。

## 复现命令

```powershell
# IFRNet S
python -m light_video_enhancer YUKI_Z.mp4 -o ifrnet.mp4 --duration 2 -s 2 `
  --fi-engine ifrnet_ncnn --fi-quality fast --sr-engine none `
  --codec h264_nvenc --preset p1 --no-audio -y

# SPAN x2 ch48
python -m light_video_enhancer YUKI_Z.mp4 -o span.mp4 --duration 2 -s 2 `
  --fi-engine none --sr-engine span --sr-quality fast `
  --codec h264_nvenc --preset p1 --no-audio -y

# IFRNet S -> SPAN
python -m light_video_enhancer YUKI_Z.mp4 -o ifrnet_span.mp4 --duration 2 -s 2 `
  --fi-engine ifrnet_ncnn --fi-quality fast --sr-engine span --sr-quality fast `
  --codec h264_nvenc --preset p1 --no-audio -y
```

FlashVSR 与 SeedVR2 未纳入本次本机速度表：它们分别需要约 6.5 GiB 与 3.6 GiB 的可选权重，以及匹配的独立 CUDA Python 环境。当前验证范围是运行时固定、环境门控、模型下载/哈希、批次调度和错误路径；在完成权重下载和目标显卡实测前，不宣称其实际速度或兼容性。
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

## 重型模型实测（2026-08-04）

同一台 RTX 5070 Ti Laptop GPU（12,820,480,000 字节显存）上已下载并逐文件校验两套重型模型。FlashVSR v1.1 共 6,440,783,673 字节；SeedVR2 3B FP8 共 3,892,869,510 字节。模型管理器与独立 `Get-FileHash` 结果均匹配清单中的 SHA-256。

| 引擎 | 输入 → 输出 | 请求/内部帧 | 初始化 | 推理 | 请求帧吞吐 | GPU 峰值 | 显存峰值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SeedVR2 3B FP8 fast | 320×180 → 640×360 | 5 / 5 | 10.588 s | 6.707 s | 0.746 fps | 99% | 8,479 MiB |
| SeedVR2 3B FP8 fast | 1280×720 → 2560×1440 | 5 / 5 | 12.164 s | 40.738 s | 0.123 fps | 99% | 9,953 MiB |
| FlashVSR v1.1 fast | 160×90 → 640×360 | 5 / 21（算法补帧） | 18.062 s | 2.317 s | 2.158 fps | 91% | 9,920 MiB |
| SeedVR2 7B Q4 quality | 160×90 → 320×180 | 5 / 5 | 18.348 s | 8.298 s | 0.603 fps | 75% | 5,599 MiB |
| SeedVR2 7B Q4 quality | 1280×720 → 2560×1440 | 5 / 5 | 8.382 s | 45.253 s | 0.110 fps | 100% | 8,589 MiB |
| DLoRAL fast | 128×128 → 512×512 | 2 / 2 | 15.057 s | 2.557 s | 0.782 fps | 85% | 9,390 MiB |
| DLoRAL quality | 128×128 → 512×512 | 2 / 2 | 16.285 s | 2.712 s | 0.737 fps | 51% | 9,354 MiB |
| DLoRAL fast | 320×180 → 1280×720 | 2 / 2 | 16.727 s | 7.338 s | 0.273 fps | 99% | 8,920 MiB |
| DLoRAL fast | 640×360 → 2560×1440 | 2 / 2 | 16.000 s | 85.723 s | 0.0233 fps | 100% | 11,822 MiB |
| VFIMamba Small fast（安全回退） | 64×64 → 64×64 | 2 / 1 中间帧 | 3.399 s | 23.650 s | 0.0423 fps | — | — |
| VFIMamba Full quality（安全回退） | 64×64 → 64×64 | 2 / 1 中间帧 | 3.767 s | 22.732 s | 0.0440 fps | — | — |
| FlashVSR v1.1 fast | 320×180 → 1280×720 | 5 / 21（算法补帧） | 12.272 s | 206.743 s | 0.024 fps | 100% | 11,774 MiB |

这里的“推理”包含适配器的 PNG 交换与输出读取。FlashVSR Tiny Long 必须满足内部时序长度约束，短请求会补到 21 帧，因此短片段吞吐不能直接外推到长视频。其 720p 输出已经逼近 12 GB 显存上限且非常慢，不应参与自动选择；保留为用户明确启用的实验模式。SeedVR2 的 block swap 与分块 VAE 在 720p→1440p 下没有 OOM，可作为慢速高质量修复模式。DLoRAL 的 720p 输出速度尚可用于短片修复，但 1440p 仅 0.0233 fps 且峰值 11,822 MiB，同样只能手动启用。

完整视频管线也已通过：

- `YUKI_Z.mp4` 前 5 帧经过 `none -> seedvr2 -> h264_nvenc`，生成 2560×1440 MP4；重新解码为精确 5 帧。
- 从同一视频制作的 160×90 五帧片段经过 `none -> flashvsr -> h264_nvenc`，生成 640×360 MP4；重新解码为精确 5 帧。
- 所有输出像素范围均为 0–255，均值/标准差非零，未出现黑帧或尺寸错误。

### 实测发现并修复的问题

1. 环境扫描原先只用 `find_spec`，会把 ABI 不匹配、导入即失败的 `torchvision` 判为可用；现对 `torchvision`、OpenCV、safetensors、GGUF 与 Block-Sparse Attention 执行真实导入。
2. 重型运行时会向 stdout 打印状态文本，破坏 framed IPC；现统一重定向到 stderr，并强制子进程 UTF-8。
3. SeedVR2 在 torch 已加载后才切换 CUDA 分配器会触发 PyTorch 内部断言及 Windows 访问冲突；现于创建子进程前固定 `backend:cudaMallocAsync`。
4. SeedVR2 运行时 ZIP 漏打包 VAE YAML；现从固定上游提交 `4490bd1f...` 补入并校验 SHA-256。
5. SeedVR2 外层 `inference_mode` 会让上游缓存的原地更新失败；现改为 `no_grad`。
6. 隔离 worker 现在抑制 Windows 原生崩溃弹窗，异常通过管道与完整 traceback 返回 GUI/CLI。
7. 应用可自动发现 `%LOCALAPPDATA%\\LightVideoEnhancer\\runtimes` 下的 venv `Scripts\\python.exe` 和 Conda 根目录 `python.exe`。
8. FlashVSR 能力门控现在同时要求 ModelScope、Transformers、PEFT、Accelerate 等实际启动依赖。

9. SeedVR2 质量档现支持按需 7B Q4 / 7B Sharp Q4 权重，并在 12 GB 显卡上使用 5 帧批次、384 tile 与最多 36 个 block swap。
10. 大文件下载器现严格验证 `Content-Range`，能处理忽略 Range、镜像超发字节和完整 `.part` 未原子安装三类续传边界情况。
11. DLoRAL 运行时补齐 `devices.py`，Torch 兼容 `ConvModule` 保留 `.conv.*` checkpoint 键名，并把分块不确定性图修正为 `[B,T-1,C,H,W]`。
12. DLoRAL 的 8.14 GiB 核心包已逐文件校验；Google Drive 配额超限时改用维护者提供的 Dropbox 备份，下载器会拒绝把 HTML 错误页当权重。
13. OSDEnhancer 已接入联合 4×/2× 管线、12.0 GiB 权重清单与 80 GB 显存硬门控；当前 12 GB 测试机不满足官方要求，因此没有虚构本地推理数据。
14. SparkVSR 已接入 42.2 GB 固定权重清单、关键帧输入和 40 GiB 显存/56 GiB 内存安全门，低配主机在下载或初始化前明确禁用。
15. VFIMamba 已接入 Small/Full 与官方 PyTorch selective-scan 安全回退；环境扫描不导入可能导致解释器退出的可选 `mamba_ssm` 原生扩展。

### FlashVSR Windows 说明

FlashVSR 官方依赖的 Block-Sparse Attention 只声明 Linux 支持，FlashVSR 官方也将 RTX 40/50 兼容性列为未知。本次 Windows 测试使用独立 Python 3.11.13、官方 PyTorch 2.8.0+cu128，以及 deAPI-ai 编译的第三方 Windows wheel；该 wheel 的 SHA-256 为 `5518B9A92C53FF7540B0A091F2D35E4CF717FEF4AEC319599328856E0F0F3408`。安装前已检查其 Python 源码、BSD 许可证、元数据和文件清单，并在 SM 12.0 上先运行独立 BF16 CUDA kernel。此路径不能等同于上游官方 Windows 支持，仍保持“实验”标记和手动启用。

### 复现

```powershell
python -m benchmarks.run_heavy_smoke --engine seedvr2 --input C:\\Users\\24645\\Downloads\\YUKI_Z.mp4 --torch-python "$env:LOCALAPPDATA\\LightVideoEnhancer\\runtimes\\seedvr2\\Scripts\\python.exe" --frames 5 --src-width 320 --src-height 180 --dst-width 640 --dst-height 360 --quality fast --output-dir build\\heavy-smoke\\seedvr2-320
python -m benchmarks.run_heavy_smoke --engine flashvsr --input C:\\Users\\24645\\Downloads\\YUKI_Z.mp4 --torch-python "$env:LOCALAPPDATA\\LightVideoEnhancer\\runtimes\\flashvsr\\python.exe" --frames 5 --src-width 160 --src-height 90 --dst-width 640 --dst-height 360 --quality fast --output-dir build\\heavy-smoke\\flashvsr-160
python -m benchmarks.run_heavy_smoke --engine dloral --input C:\\Users\\24645\\Downloads\\YUKI_Z.mp4 --torch-python "$env:LOCALAPPDATA\\LightVideoEnhancer\\runtimes\\seedvr2\\Scripts\\python.exe" --frames 2 --src-width 320 --src-height 180 --dst-width 1280 --dst-height 720 --quality fast --output-dir build\\heavy-smoke\\dloral-320
```

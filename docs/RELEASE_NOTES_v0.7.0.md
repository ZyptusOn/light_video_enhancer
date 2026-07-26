# Light Video Enhancer v0.7.0

本版本扩展了跨厂商 AI 插帧与超分能力，并将自动选择从固定优先级升级为可解释的
上下文评分器。它同时完成了 WinUI 模型门控、重型可选运行时、Full/Lite 打包和
NCNN 常驻 worker 的下一轮整合。

## 主要变化

- 新增 IFRNet S / Base / L NCNN/Vulkan 插帧，分别映射 `fast`、`balanced`
  与 `quality`，并接入常驻原生 worker。
- 新增 SPAN 2×/4×、48/52 通道 NCNN/Vulkan 超分；修复模型输出量程及
  BGR/RGB 顺序，真实视频不再产生黑帧或通道互换。
- 新增 EMA-VFI Small CUDA 插帧，使用持久隔离进程、共享内存和多时刻特征复用。
- 新增 Win10/11 可选 FlashVSR v1.1 与 SeedVR2 3B FP8 运行时。两者权重体积、
  显存和环境要求较高，只允许手动启用，不参与自动选择，也不内置到 Full 包。
- Real-ESRGAN `quality` 的 2×/3× 任务改用原生倍率模型，避免无意义的 4×
  推理后缩小；`ultra` 仍保留明确的 4× 超采样与 TTA。

## 智能自动选择

后端现在先探测输入视频，再依据 GPU、实际可用编码器、已安装模型、手动扫描后
缓存的 Python/CUDA/NV-VFX 能力、输入与目标像素数、质量档、插帧倍率和阶段顺序
评分。日志会输出最终引擎、评分与主要原因。

- NVIDIA 且已确认 CUDA + NV-VFX 时，质量档优先融合
  `RIFE PyTorch -> NV-VFX`。
- 极速档优先低延迟 IFRNet 与可用的 D3D11 驱动 VSR。
- 高分辨率“先超分再插帧”会降低高像素 PyTorch RIFE 的得分。
- D3D11 VSR 严格遵守 4096×2160 上限；1× 阶段会被跳过。
- 编码器 `auto` 只从 FFmpeg 后端实际报告可用的编码器中选择。
- 处理启动只读取显式扫描缓存，不再隐式遍历 Python、Conda、uv 或 pyenv。
- 用户明确选择的引擎始终保持不变。

## 模型与打包

- WinUI Full 内置标准模型；Lite 不含权重，可在“模型与下载”页按需安装。
- 新增 `ema-vfi-small`、`ifrnet-ncnn` 与 `span-ncnn` 下载包，并继续支持
  GitHub、镜像、自定义源和本地 ZIP。
- FlashVSR 与 SeedVR2 使用固定版本运行时及逐文件 SHA-256，但大型权重从
  Hugging Face 或镜像按需下载。
- WinUI 继续只发布两个根目录 EXE，并只保留中文和英文资源。
- Windows 7 SP1 继续使用 Python 3.8.10 / Tk LTS 全量包；Win10/11 专属重型
  运行时不会进入 Win7 构建。

## 性能与正确性

RTX 5070 Ti Laptop GPU 上对 `YUKI_Z.mp4` 前 2 秒的端到端测试：

| 管线 | 输出 | 时间 | 输入吞吐 |
|---|---:|---:|---:|
| IFRNet S，不超分 | 1280×720，119 帧 | 3.697 s | 16.23 fps |
| SPAN x2 ch48，不插帧 | 2560×1440，60 帧 | 12.915 s | 4.65 fps |
| IFRNet S -> SPAN x2 ch48 | 2560×1440，119 帧 | 19.335 s | 3.10 fps |

同机此前记录的 RIFE PyTorch + NV-VFX 基线为 8.01 输入 fps。IFRNet 本身很轻，
SPAN 是组合瓶颈，因此自动超分在没有 NV-VFX 时仍优先实测更快的 Real-ESRGAN。
完整方法见
[`benchmarks/NEW_ALGORITHMS_REPORT.md`](../benchmarks/NEW_ALGORITHMS_REPORT.md)。

## 下载

- `LightVideoEnhancer-WinUI3-Full-Win10-11-x64.zip`：标准模型内置。
- `LightVideoEnhancer-WinUI3-Lite-Win10-11-x64.zip`：不含权重，按需下载。
- `LightVideoEnhancer-Win7-x64.exe`：Python 3.8.10 / Tk / 标准模型 LTS 包。

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `LightVideoEnhancer-WinUI3-Full-Win10-11-x64.zip` | 412.64 MiB | `26833CC5A6280824123928AB57811861DE861478627DF609BDB17997B93835B0` |
| `LightVideoEnhancer-WinUI3-Lite-Win10-11-x64.zip` | 176.02 MiB | `45FA39ACAD453BEC27B1347EEFF9C6E858F10EEFC60DF7A38BAD2840DC1CD025` |
| `LightVideoEnhancer-Win7-x64.exe` | 318.92 MiB | `AF8D040BF899B0F14235796104187F5D707580C977E322C5D9FE9830D9CBA842` |

## 验证

- Python 单元及集成测试：44 项通过 43 项，1 项真实 Vulkan 冒烟默认按环境跳过。
- 真实自动管线成功执行 `RIFE -> NV-VFX -> H.264 NVENC`，输出 11 帧均可解码。
- WinUI Release x64 构建 0 警告、0 错误。
- SPAN 原生 worker 已通过真实 Vulkan 字节范围和 BGR 顺序冒烟。

## English summary

v0.7.0 adds IFRNet NCNN interpolation, SPAN NCNN super resolution, and
EMA-VFI Small CUDA interpolation. It also integrates optional FlashVSR and
SeedVR2 runtimes while keeping their multi-gigabyte weights out of automatic
selection and standard release bundles. The new context-aware selector scores
hardware, verified runtimes, target resolution, quality, stage order, and
measured throughput; encoder auto-selection now uses actual backend
availability. Full, Lite, and Windows 7 LTS packages share the same processing
core, while the WinUI model page supplies optional weights to Lite builds.

# Light Video Enhancer v0.6.0

本版本完成了 NCNN/Vulkan 路径从参数调优、目录流水到常驻原生 worker 的完整重构，
同时收录 WinUI 主题修复、统一后端执行器接口和 RIFE PyTorch 调度优化。

## 主要变化

- 新增常驻 `lve-ncnn-worker.exe`，在一个 Vulkan 进程内完成 RIFE NCNN 与
  Real-CUGAN、Real-ESRGAN 或 ESRGAN 超分。
- 帧数据改用 Windows 命名共享内存传输，移除逐批进程启动和 PNG
  编码、落盘、读取开销；初始化失败会自动回退到兼容目录流水。
- 新增 `FrameBatchExecutor` 与不可变 NCNN 阶段契约，处理管线不再依赖具体引擎的
  私有字段，便于 WinUI、Win7 Tk 和 CLI 共用同一后端。
- Real-ESRGAN 的 2×/3× 快速与平衡档使用原生倍率模型，避免无意义的 4×
  推理后缩小。
- 修复原生 worker 的 BGR/RGB 色彩约定，并增加批次错误隔离、资源所有权和
  Windows 7 SP1 兼容构建。
- 关闭 RIFE 多种运行方式中的重复 cuDNN 形状搜索，减少短视频启动开销。
- WinUI 明暗主题现在应用到窗口根视觉树、Mica、导航区和标题栏，修复切换到
  明亮模式后底层仍为黑色的问题。

## 性能

在 RTX 5070 Ti Laptop GPU、`YUKI_Z.mp4` 2 秒片段、720p→1440p、2× 插帧和
HEVC NVENC 条件下：

| 管线 | v0.6.0 输入 fps | 旧 CLI/PNG 输入 fps | 提升 |
|---|---:|---:|---:|
| RIFE NCNN + Real-CUGAN | 4.99 | 1.50 | 3.32× |
| RIFE NCNN + Real-ESRGAN AnimeVideo-v3 | 8.18 | 1.85 | 4.42× |
| RIFE NCNN + ESRGAN classic | 0.33 | 0.14 | 2.33× |

同机 RIFE PyTorch + NVIDIA VFX 基线为 8.01 输入 fps。完整方法和采样数据见
[`benchmarks/NCNN_REFACTOR_REPORT.md`](../benchmarks/NCNN_REFACTOR_REPORT.md)。

## 下载

- `LightVideoEnhancer-WinUI3-Full-Win10-11-x64.zip`：全部模型权重内置。
- `LightVideoEnhancer-WinUI3-Lite-Win10-11-x64.zip`：不含权重，可在 GUI 中下载。
- `LightVideoEnhancer-Win7-x64.exe`：Python 3.8.10 / Tk / Full 权重 LTS 包。

| 文件 | SHA-256 |
|---|---|
| Full WinUI | `8C67F2E62DA558F2C1093B3686106332EBC8A4AF2089EA8F466F7905AF26AAD4` |
| Lite WinUI | `B47F1E15DC8B3F01B1F5537A15300016985A43F0BE2121C32CA1E4591B5C8ED2` |
| Win7 LTS | `7CBA3E6487DF039860CF9DDC47B2F6C12AFF170FFF920301E88991722CC958ED` |

## 验证

- Python 单元及集成测试：35/35 通过。
- WinUI Release x64：0 警告、0 错误。
- Full/Lite 能力协议均返回版本 0.6.0；Full 显示全部权重已内置，Lite
  显示权重待下载。
- Win7 包由 Python 3.8.10 与 PyInstaller 5.13.2 构建，并通过 GUI 启动存活测试。
- 原生 NCNN 管线完成真实视频输出与旧 CLI/PNG 图像回归对照。

## English summary

v0.6.0 replaces the NCNN CLI/PNG hot path with a persistent native Vulkan
worker backed by Windows named shared memory. It delivers 3.32× faster
RIFE + Real-CUGAN and 4.42× faster RIFE + Real-ESRGAN throughput in the
recorded 720p→1440p benchmark. The release also introduces a backend-neutral
batch executor contract, corrects BGR/RGB handling, reduces RIFE startup
overhead, fixes incomplete WinUI light-theme switching, and ships refreshed
Full, Lite, and Windows 7 LTS builds.

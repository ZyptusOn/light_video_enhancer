# NCNN/Vulkan 分阶段优化与重构报告

日期：2026-07-26

## 测试条件

- 输入：`YUKI_Z.mp4` 的 0–2 秒，共 60 帧，1280×720、29.999 fps。
- 输出：2560×1440、59.999 fps，共 119 帧。
- 编码：HEVC NVENC、preset p5、CQ 23。
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU（NCNN Vulkan 设备 1）。
- 基准：RIFE PyTorch + NVIDIA Video Effects VSR。
- 每条结果均由冻结版 WinUI 后端执行，输出经过内嵌 FFmpeg 再次探测。

原始日志、逐次采样和输出信息位于
[`results/YUKI_Z_native_ncnn_refactor`](results/YUKI_Z_native_ncnn_refactor/RESULTS.md)。
最终 BGR/RGB 修正后的两条推荐链路复测位于
[`results/YUKI_Z_native_ncnn_final`](results/YUKI_Z_native_ncnn_final/RESULTS.md)。

## 结果

| 管线 | 处理阶段输入 fps | 相对旧 CLI | GPU 平均 | CPU 平均 | 显存峰值 |
|---|---:|---:|---:|---:|---:|
| RIFE PyTorch + NV-VFX 基准 | 8.01 | — | 12.4% | 12.4% | 1527 MiB |
| RIFE NCNN + Real-CUGAN，常驻 worker | 4.99 | **3.32×** | 60.8% | 25.3% | 1568 MiB |
| RIFE NCNN + Real-CUGAN，旧 CLI/PNG | 1.50 | 1.00× | 39.0% | 32.4% | 3492 MiB |
| RIFE NCNN + Real-ESRGAN AnimeVideo-v3，常驻 worker | **8.18** | **4.42×** | 58.6% | 24.0% | 782 MiB |
| RIFE NCNN + Real-ESRGAN AnimeVideo-v3，旧 CLI/PNG | 1.85 | 1.00× | 26.5% | 26.8% | 972 MiB |
| RIFE NCNN + ESRGAN classic，常驻 worker | 0.33 | **2.33×** | 90.7% | 18.8% | 1794 MiB |
| RIFE NCNN + ESRGAN classic，旧 CLI/PNG | 0.14 | 1.00× | 70.5% | 31.5% | 2838 MiB |

Real-ESRGAN AnimeVideo-v3 已达到基线同级吞吐量，并且其显存峰值最低。
经典 ESRGAN 的固定原生倍率是 4×；在目标仅为 2× 时仍要先生成 5120×2880
再缩小，因此即使消除架构开销也远慢于其他模型。它适合强调感知细节或真正需要
4× 输出的任务，不应作为 2× 自动选择。

## 由浅到深的改动

### 1. 参数与调度

- 按分辨率和引擎分别选择 NCNN load/process/save worker 数量。
- 为 RIFE、Real-CUGAN、Real-ESRGAN 提供经过范围校验的环境变量覆盖。
- 质量档位真正映射到模型、去噪、TTA 和倍率，而不再只是 GUI 文案。
- Real-ESRGAN 的 2×/3× 快速与平衡档直接使用 AnimeVideo-v3 原生倍率，
  避免无意义的 4× 推理。

### 2. 目录三级流水

旧兼容路径仍可用，但解码、NCNN 目录任务与编码现在通过双工作区交叠执行。
RIFE 输出目录可直接交给 SR CLI，避免两阶段之间回读到 Python 再写出。

### 3. 常驻原生 worker

新增 `lve-ncnn-worker.exe`，一次加载 Vulkan 和模型，并通过 Windows 命名共享内存
收发连续 BGR24 帧。它在一个进程中完成 RIFE → SR，消除了：

- 每批启动两个可执行文件；
- 每帧 PNG 编码、磁盘写入、读取和解码；
- 两个 NCNN 程序各自重复创建 Vulkan 实例和加载模型；
- 大量临时目录清理与 Python 大数组往返。

worker 复用源帧超分缓存、RIFE 预测帧和 SR 输出缓冲；单批推理错误不会破坏后续
协议。Vulkan 生命周期、共享映射和进程释放均使用明确的所有权边界。

### 4. 后端接口重构

- `FrameBatchExecutor` 统一 CUDA 与 Vulkan 常驻执行器。
- SR/FI 基类公开批处理、目录批处理、输出尺寸和原生 NCNN 阶段能力。
- 各 NCNN 引擎通过不可变阶段描述公开模型配置。
- 管线不再导入具体 NCNN 类或读取 `_model_dir`、`_quality` 等私有字段。
- GUI 与 CLI 继续只构造 `ProcessConfig`；新增后端不会要求复制前端判断逻辑。

## 正确性与兼容性

- 35 项自动化测试全部通过。
- 真实帧上，原生输出相对旧 PNG/CLI 输出：
  - Real-CUGAN：约 40.9 dB PSNR，平均绝对误差约 1.27/255。
  - Real-ESRGAN：约 42.2 dB PSNR，平均绝对误差约 1.13/255。
- 原生 worker 显式执行 BGR↔RGB 转换，与 FFmpeg/OpenCV 和上游 NCNN PNG 工具的
  色彩约定一致。
- worker 目标版本为 `_WIN32_WINNT=0x0601`，使用静态 MSVC 运行库；OpenMP
  `vcomp140.dll` 与 worker 同目录打包，保留 Windows 7 SP1 启动兼容性。
- `LVE_DISABLE_FUSED_NCNN=1` 可随时强制回退到旧 CLI/PNG 路径。

## 仍可继续探索的方向

当前 worker 已消除进程、PNG 和 Python 中间帧开销，但上游 RIFE API 仍把结果下载
为 CPU `ncnn::Mat`，SR 随后重新上传为 `VkMat`。真正的跨模型 Vulkan 零拷贝需要
维护上游算法分支、统一张量布局和命令提交器；它的维护风险明显高于本轮收益，
不适合在没有逐模型图像回归集的情况下直接进入默认路径。

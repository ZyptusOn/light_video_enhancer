# 更新日志

## v0.5.0 — WinUI 双包、模型下载与稳定前后端协议

### 发行与模型

- Windows 10/11 发行版拆分为 Full 和 Lite；两者共用同一套 WinUI 3 前端。
- Full 内置全部 RIFE、Real-CUGAN、Real-ESRGAN 与 ESRGAN 权重，Lite 只保留核心运行文件。
- 新增模型下载页，支持 GitHub、代理镜像、自定义 URL 模板和本地 ZIP 导入。
- 下载与导入会校验归档 SHA-256、单文件 SHA-256、精确文件清单和安全路径。
- 外置模型默认安装到 `%LOCALAPPDATA%\LightVideoEnhancer\models`，程序升级不会清除。

### 架构与界面

- WinUI 与后端通过版本为 1 的 JSON/JSONL 协议通信，模型状态、环境和能力查询不再依赖 Python 内部结构。
- 新增系统语言、中文和英文切换；CLI 增加 `--language` / `-L`。
- 新增应用 Logo、ICO 和完整 WinUI 图标资源，并升级到 Windows App SDK 1.8。
- Windows 10/11 Tk 版停止发布；Windows 7 Tk 版作为只含 Full 权重的冻结 LTS 保留。

### 验证

- Python 单元及集成测试 28 项通过。
- WinUI Release x64 构建 0 警告、0 错误。
- Full 与 Lite 中除后端文件外的 336 个前端文件逐一 SHA-256 相同。
- Full、Lite 和 Windows 7 GUI 均通过启动冒烟测试。

## v0.4.5 — 双平台发布、NCNN 快速管线与 ESRGAN

这是项目从早期 NVIDIA 专用原型走向可组合、跨厂商视频增强工具的一次完整重构。它保留了通过 D3D11 Video Processor 调用驱动视频增强能力的核心思路，同时补齐便携 NCNN、外部 PyTorch、软件回退、现代编码和 Windows 7 发布链。

### 主要更新

- 项目和 Python 包由 `nvidia_video_enhancer` 更名为 `light_video_enhancer`，CLI 入口由 `nve` 改为 `lve`。
- 提供 Windows 10/11 x64 与 Windows 7 SP1 x64 两套单文件 GUI。
- 新增 Real-ESRGAN AnimeVideo-v3 2×/3×/4×、x4plus、x4plus-anime，以及经典 ESRGAN x4 感知模型。
- Real-CUGAN、Real-ESRGAN、经典 ESRGAN 和 RIFE NCNN 全部使用目录批处理。
- RIFE NCNN 与 NCNN 超分支持安全的双向目录直连，消除 Python 中间帧的重复读写。
- 批量大小根据输入、目标和模型原生尺寸自动计算；编码队列动态扩展，临时目录后台清理。
- Windows 10/11 的 `RIFE PyTorch -> NV-VFX` 增加融合 CUDA Worker，减少 GPU/CPU 往返和重复颜色转换。
- RIFE 和 NV-VFX 独立模式使用预分配共享内存；跨环境协议不再序列化 NumPy 对象。

### 编解码

- 解码支持 CUDA、D3D11VA、dav1d 和软件回退。
- H.264：NVENC、AMF、Media Foundation、x264。
- H.265/HEVC：NVENC、AMF、Media Foundation、x265。
- AV1：NVENC、AMF、SVT-AV1、libaom。
- 指定编码器不可用时优先保留格式，再按 AV1 → HEVC → H.264 → MPEG-4 降级。
- 修复编码末帧丢失、错误帧率、解码包丢失、音频片段时间戳和编码器 flush。

### GUI 与环境

- 超分质量和插帧质量拆分，控件只在对应后端支持时启用。
- 增加 NCNN Vulkan GPU 选择、编码 preset、片段处理、音频复制和覆盖选项。
- GUI 启动只做快速能力检测；Python/PyTorch/CUDA 扫描改为手动、并行和缓存。
- 环境发现覆盖 PATH、Python Launcher、Windows 注册表、Conda、uv、pyenv、Poetry 和常见虚拟环境目录。
- NVIDIA Video Effects 在隔离子进程中初始化和推理，增加超时与自动回退。

### 性能

在 720p、33 帧合成基准中，旧的 Python 中间帧路径耗时 22.256 秒，v0.4.5 NCNN 目录直连耗时 15.425 秒：吞吐提升约 `1.443×`，总耗时降低约 `30.7%`。实际收益会随 GPU、模型、分辨率、磁盘和编码器变化。

### 修复

- 修复外部 Python NumPy 版本不同导致的 `No module named 'numpy._core'`。
- 修复环境扫描遗漏部分 Python/Conda 安装的问题。
- 修复 NCNN GPU 被硬编码、质量选项和模型实际行为不一致的问题。
- 修复模型原生 4× 输出直接进入 2× 插帧阶段造成的额外计算和语义偏差。
- 修复 RIFE 静帧、切镜和 warp 网格缓存持续占用显存的问题。
- 修复取消、失败和覆盖场景下的部分文件处理。

### 验证

- Python 3.13：24 项测试通过。
- Python 3.8.10：24 项测试通过。
- x264、x265、SVT-AV1、libaom、Media Foundation、MPEG-4 真实编码—解码往返通过。
- `RIFE NCNN -> Real-CUGAN -> AV1 NVENC` 端到端通过。
- `Real-CUGAN -> RIFE NCNN` 反向目录直连通过。
- 两套单文件程序均能创建并响应 GUI 窗口；Win7 包使用 Python 3.8.10 与 PyInstaller 5.13.2 构建。

### 升级说明

- 旧脚本中的 `nvidia_video_enhancer` 导入需要改为 `light_video_enhancer`。
- 旧命令 `nve` 需要改为 `lve`。
- 建议重新检查 GUI 中的质量档位、编码 preset 和 NCNN GPU。
- 首次启动和首次加载模型可能因单文件解包、杀毒扫描和 Vulkan 缓存建立而更慢。

### 下载

- `LightVideoEnhancer-Win10-11-x64.exe`：适用于 Windows 10/11 64 位。
- `LightVideoEnhancer-Win7-x64.exe`：适用于 Windows 7 SP1 64 位兼容环境。

完整使用说明、校验值和已知限制见 [README.md](README.md)。

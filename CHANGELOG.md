# 更新日志

## 未发布 / Unreleased

- 将便携后端改为可独立运行的完整控制台 CLI：无参数交互向导、分组双语 `--help`、系统信息、能力/环境/模型协议查询与完整视频处理均不依赖 GUI。
- 后端 PyInstaller 入口不再导入 Tkinter、Tcl/Tk、IDLE、旧 Tk GUI 或 `light_video_enhancer.gui`；Windows 7 Tk LTS 源码继续与后端共享稳定处理配置和核心模块。
- CLI 与后端 EXE 的向导、帮助、处理日志和主要错误默认按 Windows 用户界面语言选择中文或英文，支持 `--language` 覆盖；协议 JSON 保持标准输出纯净，诊断日志移至标准错误。
- WinUI 语言选项固定显示自称名称“中文”与“English”；补齐模型页、环境页、处理页和首次进入折叠页面时的英文动态本地化。
- WinUI 自包含单文件通过排除未使用的 AI/ML、ONNX、DirectML、Widgets 与 NPU 工作负载，并采用显式裁剪根的受控 partial trimming，从约 85.6 MiB 降至约 42.4 MiB。
- 实机验证中英文页面、12 个模型包、后端能力协议和独立 CLI；无裁剪根的 34.1 MiB 实验包因 WinUI 初始化崩溃被明确淘汰。

## v0.7.0 — 新一代算法、智能自动选择与模型运行时

- 重构自动选择为上下文评分器：在视频探测后根据 GPU、模型、已扫描 Python/CUDA/NV-VFX 能力、目标像素、质量档、倍率与处理顺序决策；极速档优先低延迟后端，高像素“先超分再插帧”会避开昂贵的 RIFE，D3D11 VSR 会遵守 4K 上限，无效的 1× 阶段会跳过，日志会解释选择原因。
- 修复 Real-ESRGAN `quality` 在 2×/3× 任务中仍运行 x4plus 4× 推理再缩小的问题；现在按目标倍率使用 AnimeVideo-v3 原生模型，`ultra` 继续保留明确的 4× 超采样 + TTA 路径。
- 新增 IFRNet S / Base / L 的 NCNN/Vulkan 插帧，并接入常驻原生 worker、模型下载、GUI、CLI 与中英文能力说明。
- 新增 SPAN 2×/4×、48/52 通道 NCNN/Vulkan 超分；转换工具会验证 PNNX/NCNN 数值对照，运行时使用 FP32 以避免 Vulkan 半精度偏差。
- 新增 EMA-VFI Small CUDA 插帧；采用持久隔离进程、共享内存和多时刻特征复用，支持 2×–4×。
- 新增 Win10/11 可选实验后端 FlashVSR v1.1：固定版本运行时、29 帧因果窗口、独立 Python 3.11 CUDA 能力门控和 Hugging Face / 镜像下载。
- 新增 Win10/11 可选重型修复后端 SeedVR2 3B FP8：接入低显存社区运行时、分块 VAE、CPU 卸载、block swap 与 4n+1 时间批次。
- FlashVSR 与 SeedVR2 的所有远程权重均增加单文件 SHA-256 校验；两者不参与自动选择，也不内置到 Full 包。
- 时间型超分的首选窗口会在“先插帧后超分”时换算为源视频批大小，避免 2× 插帧把 29 帧窗口错误膨胀为 57 帧。
- 环境扫描缓存升级并显式版本化，旧缓存会自动失效；只有扫描确认对应 Python/CUDA 依赖后，GUI 才启用重型后端。
- 修正 Full/Lite 打包清单：Lite 按模型目录精确排除所有权重，现代后端包含 EMA-VFI 与固定版本重型运行时，Win7 包排除 Win10/11 专属运行时。
- 修复 SPAN NCNN 将 [0,1] 浮点输出直接转为字节以及 BGR/RGB 顺序错误所造成的全黑视频；新增可选真实 Vulkan 冒烟测试。
- `YUKI_Z.mp4` 2 秒有效输出实测：IFRNet S 单独 60→119 帧为 3.70 秒，SPAN x2 单独为 12.92 秒，组合为 19.34 秒。SPAN 是组合瓶颈，因此自动超分在模型齐全时优先选择 Real-ESRGAN，再回退到 SPAN。

## v0.6.0 — 常驻 NCNN/Vulkan worker 与后端执行器重构

### 性能与架构

- 新增常驻原生 `lve-ncnn-worker.exe`，在同一个 Vulkan 进程内执行 RIFE NCNN 与 Real-CUGAN、Real-ESRGAN 或 ESRGAN。
- 使用 Windows 命名共享内存传输 BGR24 帧，消除逐批进程启动、PNG 编解码和磁盘中转；初始化失败会自动回退到兼容目录流水。
- 新增双工作区目录三级流水，让解码、NCNN 目录任务与编码能够交叠执行。
- 新增统一的 `FrameBatchExecutor` 和不可变 NCNN 阶段契约；管线不再读取具体引擎的私有字段。
- Real-ESRGAN 2×/3× 快速与平衡档改用 AnimeVideo-v3 原生倍率，避免 4× 推理后缩小。
- RIFE PyTorch、融合 RIFE + NV-VFX 和外部 RIFE worker 关闭重复 cuDNN 形状搜索，降低短视频启动开销。
- 原生 worker 复用模型、Vulkan 管线和中间缓冲，并增加批次错误隔离、BGR↔RGB 修正与明确的资源所有权。

### 界面与兼容

- WinUI 明暗主题应用到窗口根视觉树、Mica、导航区和标题栏，修复切换到明亮模式后背景底层仍为黑色。
- GUI 与 CLI 继续只依赖 `ProcessConfig` 和稳定协议；新增或替换后端不需要复制前端判断逻辑。
- 原生 worker 以 `_WIN32_WINNT=0x0601`、静态 MSVC 运行库构建，并将 `vcomp140.dll` 随包提供。
- 保留 `LVE_DISABLE_FUSED_NCNN=1` 兼容开关，可强制回退到旧 CLI/PNG 路径。

### 性能结果

- RIFE NCNN + Real-CUGAN：1.50 → 4.99 输入 fps，提升 3.32×。
- RIFE NCNN + Real-ESRGAN AnimeVideo-v3：1.85 → 8.18 输入 fps，提升 4.42×。
- RIFE NCNN + ESRGAN classic：0.14 → 0.33 输入 fps，提升 2.33×。
- 同机 RIFE PyTorch + NVIDIA VFX 基线为 8.01 输入 fps。

### 验证与发布

- Python 单元及集成测试 35 项通过；真实视频冒烟与图像回归对照通过。
- WinUI Release x64 构建 0 警告、0 错误；Full/Lite 均保持两个 EXE、零子目录。
- Full/Lite 后端均返回协议版本 1 和程序版本 0.6.0，模型状态分别符合内置/按需下载预期。
- Windows 7 Full GUI 使用 Python 3.8.10 与 PyInstaller 5.13.2 重建，并通过启动存活测试。

## v0.5.2 — WinUI 可用性与精简发布修复

### 界面

- 主窗口恢复为 1280×900 的普通窗口启动，不再强制最大化；所有页面改用居中的内容视口。
- 移除“编码、设备与片段”的 Expander 子层级，直接使用统一卡片布局。
- InfoBar 默认折叠，仅在标题或正文非空时显示，修复首页顶部空白通知条。
- 模型页 JSON 传输改为代码页无关格式，中文名称与说明不再乱码。
- Python 环境扫描完成后立即刷新“硬件与可用能力”，显示 PyTorch、CUDA 与 NVIDIA VFX 环境计数。
- 启动及扫描失败时禁用 NV-VFX、RIFE PyTorch 与 CUDA 光流；仅在手动扫描确认对应环境后解锁。
- 主题切换提升到窗口根视觉树，Mica 底层、导航区、页面和标题栏现在会同步切换明暗颜色。

### 发布

- WinUI 前端改为自包含单文件发布，Full/Lite 解压目录均从数百个运行库文件精简为两个 EXE。
- 发行资源限制为 `zh-CN` 与 `en-US`，不再生成无关语言目录。
- 后端查找同时支持宿主 EXE 目录与自解压运行目录。
- 后端不再打包 Tk/Tcl，并去除重复的 FFmpeg DLL，只保留运行时必须的动态库。
- Full 与 Lite 继续共用同一个前端 EXE，仅后端内置权重不同。

### 验证

- Python 单元及集成测试 28 项通过。
- WinUI Release x64 单文件发布通过，前端大小约 85.6 MiB。
- Full/Lite 目录均验证为 2 个文件、0 个子目录；最终大小分别约 305.0 MiB 与 168.6 MiB。
- 实机验证普通窗口启动、页面居中、模型中文、空通知条消失，以及环境扫描后的能力栏和算法选项联动。

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

# Light Video Enhancer

[![Release](https://img.shields.io/github/v/release/ZyptusOn/light_video_enhancer?display_name=tag)](https://github.com/ZyptusOn/light_video_enhancer/releases/latest)
[![Windows](https://img.shields.io/badge/Windows-7%20SP1%20%7C%2010%20%7C%2011-0078D6)](#系统与硬件)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Light Video Enhancer 是一款面向 Windows 的视频超分、插帧与转码工具。它最初源于一个直接的想法：模仿 VLC 等播放器的 D3D11 视频呈现路径，让显卡驱动把离线视频帧当作正在播放的内容，从而调用驱动提供的视频增强能力。现在它已经发展为一条可组合的处理流水线，同时支持 NVIDIA Video Effects、RIFE、ncnn-vulkan、OpenCV 光流和多种软硬件编码器。

项目不再假设计算机一定安装 NVIDIA 显卡。NVIDIA 用户可以使用驱动 VSR、NV-VFX、CUDA RIFE 和 NVENC；Intel、AMD 或较旧系统可以选择 Vulkan NCNN、OpenCV、Media Foundation 或软件编码后端。

## 下载

| 系统 | 文件 | 模型权重 | 说明 |
|---|---|---|---|
| Windows 10/11 x64 | `LightVideoEnhancer-WinUI3-Full-Win10-11-x64.zip` | 标准模型内置 | 解压即用；FlashVSR / SeedVR2 因体积和环境要求仍按需下载。 |
| Windows 10/11 x64 | `LightVideoEnhancer-WinUI3-Lite-Win10-11-x64.zip` | 按需下载 | 推荐给带宽和磁盘有限的用户；可切换 GitHub、代理镜像、自定义源或导入本地 ZIP。 |
| Windows 7 SP1 x64 | `LightVideoEnhancer-Win7-x64.exe` | 全部内置 | Python 3.8.10 / Tk 的冻结兼容版，不提供下载页。 |

两个 WinUI 包使用完全相同的 GUI 与后端协议；区别只在后端是否嵌入标准权重。下载的模型保存在 `%LOCALAPPDATA%\LightVideoEnhancer\models`，升级或替换程序目录不会删除。两版都内置 FFmpeg Worker 和 NCNN 执行器，不需要系统 FFmpeg。FlashVSR（约 6.5 GiB）和 SeedVR2（约 3.6 GiB）属于 Win10/11 可选重型模型，为避免把 Full 包扩大到十余 GiB，两种包都通过模型页按需下载并校验单文件 SHA-256。

### WinUI 3 现代前端

面向 Windows 10 1809 及以上系统，提供 Mica/Fluent 界面、拖放输入、硬件与环境页、模型下载页、中英文切换、机器可读进度和安全取消。GUI 不加载 Python、CUDA 或 Vulkan；它通过带版本号的标准输入输出协议调用独立后端。

Windows 7 继续提供包含全部权重的 Tk LTS 包，但不再发布 Windows 10/11 Tk 版本。建议只为 Win7 版提供关键兼容性修复，不再追求与 WinUI 的功能同步。

开发与便携打包见 [`windows/README.md`](windows/README.md)，前后端边界和兼容策略见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

### v0.7.0 发布包校验值

| 文件 | SHA-256 |
|---|---|
| `LightVideoEnhancer-WinUI3-Full-Win10-11-x64.zip` | `26833CC5A6280824123928AB57811861DE861478627DF609BDB17997B93835B0` |
| `LightVideoEnhancer-WinUI3-Lite-Win10-11-x64.zip` | `45FA39ACAD453BEC27B1347EEFF9C6E858F10EEFC60DF7A38BAD2840DC1CD025` |
| `LightVideoEnhancer-Win7-x64.exe` | `AF8D040BF899B0F14235796104187F5D707580C977E322C5D9FE9830D9CBA842` |

三套发行文件均使用 v0.7.0 源码重建。Full 与 Lite 内的 WinUI 前端逐字节相同。

## v0.7.0 亮点

- 新增 IFRNet S/Base/L NCNN 插帧、SPAN NCNN 超分与 EMA-VFI Small CUDA 插帧。
- 可按需安装 FlashVSR v1.1 和 SeedVR2 3B FP8；重型算法不会自动选择或进入标准发行包。
- 自动选择器会综合真实输入尺寸、目标分辨率、质量、阶段顺序、实测吞吐、已安装模型和显式扫描到的运行环境。
- 编码器自动选择只使用后端实际报告的 NVENC、AMF、Media Foundation、x264/x265 与 AV1 编码器。
- 新增三种标准模型下载包；Full 内置 10 个标准包，Lite 通过同一 WinUI 下载页按需安装。
- Real-ESRGAN 2×/3× 使用原生倍率模型；修复 SPAN 的输出量程、BGR/RGB 顺序和黑帧问题。
- Full、Lite 与 Win7 LTS 三套包全部按 v0.7.0 重建并验证。

完整变化和发布说明见 [CHANGELOG.md](CHANGELOG.md) 与
[`docs/RELEASE_NOTES_v0.7.0.md`](docs/RELEASE_NOTES_v0.7.0.md)。

## 处理流水线

```text
输入视频
   │
   ├─ 内嵌 FFmpeg 解码：CUDA / D3D11VA / dav1d / 软件回退
   │
   ├─ 插帧：RIFE / EMA-VFI / RIFE NCNN / IFRNet NCNN / DIS / Farneback / CUDA 光流 / Blend
   │
   ├─ 超分：驱动 VSR / NV-VFX / SPAN / Real-CUGAN / Real-ESRGAN / ESRGAN / FlashVSR / SeedVR2 / Lanczos
   │
   └─ 编码：NVENC / AMF / Media Foundation / x264 / x265 / SVT-AV1 / libaom
          └─ 可复制原视频音频，失败时保留同格式优先的自动回退
```

默认顺序是先插帧、后超分，通常更快；勾选“先超分再插帧”会提高插帧输入分辨率，可能改善细节，也会显著增加计算量。连续的 NCNN 后端会在安全条件下自动直连目录。

## 后端

### 超分辨率

| GUI / CLI | 设备 | 适用场景 |
|---|---|---|
| 驱动 VSR / `dxva_vsr` | D3D11，NVIDIA / Intel / AMD | 模拟播放器 Video Processor 路径；速度快，增强效果取决于显卡和驱动。 |
| NVIDIA Video Effects / `nvvfx` | NVIDIA CUDA | NVIDIA SDK AI 超分；支持隔离进程、超时保护和融合 CUDA 管线。 |
| SPAN / `span` | Vulkan | 轻量 2×/4× 超分；48/52 通道模型可按质量选择，支持 NVIDIA、AMD 与 Intel。 |
| Real-CUGAN / `realcugan` | Vulkan | 动漫、线稿和压缩视频；2×/3×/4× NCNN 批处理。 |
| Real-ESRGAN / `realesrgan` | Vulkan | 通用或动漫视频；提供轻量视频模型和 x4plus 系列。 |
| ESRGAN classic / `esrgan` | Vulkan | 原始 ESRGAN x4 感知模型，偏锐利和纹理增强。 |
| FlashVSR / `flashvsr` | PyTorch CUDA，Win10/11 | 实验性 4× 因果扩散视频超分；固定 29 帧窗口，需要 Python 3.11 和约 6.5 GiB 权重。 |
| SeedVR2 / `seedvr2` | PyTorch CUDA，Win10/11 | 重型视频修复；3B FP8 + VAE，支持分块 VAE、CPU 卸载与模型交换，不参与自动选择。 |
| Lanczos / `lanczos` | CPU | 稳定、清晰的传统缩放回退。 |
| Bicubic / `bicubic` | CPU | 平滑的传统缩放回退。 |

Real-ESRGAN 质量档位：

| 档位 | 模型 / 行为 |
|---|---|
| `fast` | AnimeVideo-v3 2×/3×/4×，按目标倍率选择原生模型。 |
| `balanced` | 2×/3× 使用 AnimeVideo-v3 原生倍率；4× 使用 x4plus-anime。 |
| `quality` | 2×/3× 使用 AnimeVideo-v3 原生倍率；4× 使用通用 x4plus。 |
| `ultra` | 通用 x4plus + TTA；2×/3× 也会先做 4× 超采样，最慢且显存占用最高。 |

### 帧插值

| GUI / CLI | 设备 | 说明 |
|---|---|---|
| RIFE / `rife` | PyTorch CUDA/CPU | 质量优先；持久子进程、共享内存和静帧/切镜检测。 |
| EMA-VFI Small / `ema_vfi` | PyTorch CUDA | 高效任意时刻插帧；2×–4× 复用相邻帧特征，持久隔离进程。 |
| RIFE NCNN / `rife_ncnn` | Vulkan | 不依赖 PyTorch，目录批处理，适合便携发布。 |
| IFRNet NCNN / `ifrnet_ncnn` | Vulkan | 轻量跨厂商插帧；`fast` / `balanced` / `quality` 对应 S / Base / L 模型。 |
| DIS / `dis` | CPU / OpenCV | 推荐的非 AI 通用方案，速度和运动补偿较均衡。 |
| Farneback / `optical_flow` | CPU / OpenCV | 经典稠密光流。 |
| CUDA 光流 / `torch_flow` | PyTorch CUDA | 轻量块匹配光流。 |
| 帧混合 / `blend` | CPU | 最快、最兼容，不进行运动估计。 |

“插帧质量”只在后端确实暴露可调策略时启用。IFRNet、EMA-VFI 与光流后端会映射到各自模型或精度策略；RIFE PyTorch、RIFE NCNN 和帧混合使用固定或自动策略时，GUI 会禁用该控件。

### 编码

| 格式 | 硬件编码 | 内置软件编码 |
|---|---|---|
| H.264/AVC | `h264_nvenc`、`h264_amf`、`h264_mf` | `libx264`，别名 `x264` / `h264` |
| H.265/HEVC | `hevc_nvenc`、`hevc_amf`、`hevc_mf` | `libx265`，别名 `x265` / `h265` |
| AV1 | `av1_nvenc`、`av1_amf` | `libsvtav1`、`libaom-av1`，别名 `av1` / `svt-av1` / `aom` |
| 兼容回退 | — | `mpeg4` |

无法打开指定编码器时，程序先在同一格式内寻找其他实现。AV1 全部不可用时再依次尝试 HEVC、H.264 和 MPEG-4。CQ/CRF 越小质量越高；H.264/H.265 可从 18–28 起步，AV1 可从 24–36 起步。

## 系统与硬件

| 环境 | 建议 |
|---|---|
| Windows 10/11 x64 | 使用现代版 EXE；推荐具备 Vulkan 1.1 的 GPU。NVIDIA CUDA 用户可选 RIFE、NV-VFX 和 NVENC。 |
| Windows 7 SP1 x64 | 使用 Win7 专用 EXE；建议安装 KB2670838 和 KB2999226。稳定基线是传统缩放、OpenCV 插帧、Media Foundation/MPEG-4。 |
| NVIDIA | 驱动 VSR、NV-VFX、CUDA RIFE、NVENC 和 Vulkan NCNN 可按驱动/SDK能力组合。 |
| Intel / AMD | 可使用 Vulkan NCNN、OpenCV 和软件编码；支持的 D3D11/AMF/Media Foundation 能力取决于驱动。 |

某个后端出现在 GUI 中表示程序资源完整，不代表当前显卡驱动一定支持它。特别是新式 RTX VSR、CUDA、NV-VFX、AV1 硬编通常不适用于 Windows 7。

## 快速开始

### GUI

1. 从 [Releases](https://github.com/ZyptusOn/light_video_enhancer/releases/latest) 下载对应系统的 EXE。
2. 选择输入视频和输出路径。
3. 选择超分、插帧、编码器及各自质量档位。
4. 可选设置开始时间、处理时长、目标帧率和 NCNN GPU。
5. 点击“开始处理”；日志会显示实际管线和自动回退结果。

推荐组合：

- NVIDIA 高质量：`RIFE PyTorch -> NVIDIA Video Effects -> HEVC/AV1 NVENC`
- 跨厂商便携：`IFRNet/RIFE NCNN -> SPAN/Real-CUGAN/Real-ESRGAN -> 软件或硬件编码`
- 低依赖稳定：`DIS -> Lanczos -> H.264 Media Foundation/x264`
- 重型修复：手动选择 `FlashVSR` 或 `SeedVR2`，不建议与插帧同时启用。

### 便携后端 CLI

WinUI 包中的 `LightVideoEnhancer-Backend.exe` 也可以作为独立 PowerShell / Windows 控制台程序使用，不依赖前端，也不包含 Tkinter 或旧 GUI。无参数启动进入交互向导；常用命令如下：

```powershell
.\LightVideoEnhancer-Backend.exe --help
.\LightVideoEnhancer-Backend.exe --system-info
.\LightVideoEnhancer-Backend.exe input.mp4 -o output.mp4 --scale 2 --fi-multiplier 2 --codec auto --overwrite
```

CLI 的向导、帮助、处理日志与错误会按 Windows 用户界面语言自动选择中文或英文，也可用 `--language zh-CN` / `--language en-US` 明确指定。`--help` 清晰分组输入与处理编码参数，并单独列出环境、模型和前端协议命令及可直接复制的例子。

完整说明见 [`CLI_GUIDE.md`](CLI_GUIDE.md)。

### 从源码运行

```powershell
git clone https://github.com/ZyptusOn/light_video_enhancer.git
cd light_video_enhancer
python -m pip install -r requirements.txt
python -m pip install -e .
python -m light_video_enhancer
```

Windows 7 请使用 CPython 3.8.10：

```powershell
py -3.8 -m pip install -r requirements-win7.txt
py -3.8 -m pip install -e . --no-deps
py -3.8 -m light_video_enhancer
```

也可以双击 `启动GUI.bat`。

## 命令行示例

安装项目后使用 `lve`；也可以将 `lve` 替换为 `python -m light_video_enhancer`。

```powershell
# 自动选择后端，2× 超分 + 2× 插帧
lve input.mp4 -o output.mp4 -s 2 --fi-multiplier 2 -y

# RIFE NCNN + Real-ESRGAN + AV1 NVENC
lve input.mp4 -o output.mp4 --fi-engine rife_ncnn --sr-engine realesrgan --sr-quality quality --codec av1_nvenc -y

# 经典 ESRGAN x4 感知模型，输出目标仍为 2×
lve input.mp4 -o output.mp4 -s 2 --fi-engine none --sr-engine esrgan --sr-quality quality --codec x265 -y

# 只处理从第 60 秒开始的 30 秒，并输出 60 fps
lve input.mkv -o clip.mp4 --start 60 --duration 30 --fps 60 --sr-engine none --fi-engine blend -y

# 指定第二个 Vulkan GPU，并使用 CPU x264 编码
lve input.mp4 -o output.mp4 --ncnn-gpu 1 --sr-engine realcugan --fi-engine rife_ncnn --codec x264 -y
```

完整参数见：

```powershell
lve --help
```

## 外部 PyTorch / NVIDIA VFX 环境

单文件 EXE 不捆绑 PyTorch。RIFE、EMA-VFI、CUDA 光流、NV-VFX、FlashVSR 和 SeedVR2 会按需使用外部 Python 环境。GUI 启动时不扫描外部 Python；在“环境与后端”页手动扫描后，程序会并行检查 PATH、Python Launcher、注册表、Conda、uv、pyenv 和 Poetry 等常见位置并缓存结果，未确认的 PyTorch / CUDA / NV-VFX 后端不会出现在可选列表中。

如果机器有多个环境，可以在 GUI 中明确选择，也可以在 CLI 中指定：

```powershell
lve input.mp4 --fi-engine rife --torch-python C:\path\to\python.exe -o output.mp4 -y
```

跨环境 IPC 不序列化 NumPy 对象，因此允许主程序和外部 PyTorch 环境分别使用 NumPy 1.x/2.x，避免 `numpy._core` 反序列化错误。

## 性能说明

兼容的 `RIFE NCNN -> NCNN 超分` 组合会自动切换到常驻原生 Vulkan worker。模型和
Vulkan 管线只加载一次，视频帧通过 Windows 命名共享内存传输，不再为每批帧启动
RIFE/SR CLI 或编解码 PNG。初始化失败会在处理开始前自动回退到目录三级流水；
`LVE_DISABLE_FUSED_NCNN=1` 可强制使用旧路径。

`YUKI_Z.mp4` 的 2 秒真实片段、720p→1440p、2× 插帧、HEVC NVENC 实测：

| 管线 | 常驻 worker 输入 fps | 旧 CLI/PNG 输入 fps | 加速 |
|---|---:|---:|---:|
| RIFE NCNN + Real-CUGAN | 4.99 | 1.50 | **3.32×** |
| RIFE NCNN + Real-ESRGAN AnimeVideo-v3 | **8.18** | 1.85 | **4.42×** |
| RIFE NCNN + ESRGAN classic | 0.33 | 0.14 | **2.33×** |

同机 RIFE PyTorch + NV-VFX 基线为 8.01 输入 fps。Real-ESRGAN 的便携 NCNN
链路已达到同级吞吐量。经典 ESRGAN 固定执行 4× 模型，在 2× 目标下仍需生成
5120×2880 中间结果，因此不适合作为自动 2× 默认项。

v0.7.0 的 2 秒片段测试中，IFRNet S 单独插帧达到 16.23 输入 fps；SPAN x2 ch48
单独超分为 4.65 输入 fps，IFRNet S + SPAN x2 ch48 为 3.10 输入 fps。自动选择器
因此会结合分辨率、质量、阶段顺序、模型可用性和实测吞吐评分，而不是固定选择最新模型。

新算法数据见 [`benchmarks/NEW_ALGORITHMS_REPORT.md`](benchmarks/NEW_ALGORITHMS_REPORT.md)；
常驻 worker 的完整方法、GPU/CPU/显存采样、正确性对照和架构说明见
[`benchmarks/NCNN_REFACTOR_REPORT.md`](benchmarks/NCNN_REFACTOR_REPORT.md)。

## 构建与测试

安装测试依赖后运行：

```powershell
python -m unittest discover -s tests -v
```

现代单文件包：

```powershell
python -m pip install -r requirements-build.txt
python build_exe.py
```

WinUI 3 自包含便携目录（需 .NET 10 SDK）：

```powershell
python -m pip install -r requirements-build.txt
powershell -ExecutionPolicy Bypass -File windows\build_winui.ps1
```

Win7 单文件包必须使用 Python 3.8.10：

```powershell
py -3.8 -m pip install -r requirements-build-win7.txt
py -3.8 build_exe.py
```

FFmpeg、FFmpeg Worker 和 D3D11 Bridge 的重建说明见构建脚本：

```bash
./build_ffmpeg.sh
./light_video_enhancer/ffmpeg_bridge/build_worker.sh
./light_video_enhancer/bridge/build.sh
```

## 目录结构

```text
light_video_enhancer/       Python 包
  bridge/                   D3D11 Video Processor Bridge
  ffmpeg_bridge/            内嵌 FFmpeg C API Worker
  ffmpeg_dlls/              发布运行库
  fi/                       插帧后端
  sr/                       超分后端
  ncnn/                     便携 NCNN 程序、模型和各自许可证
benchmarks/                 GPU/NCNN 性能基准
tests/                      单元与真实编解码往返测试
windows/                    WinUI 3 前端、构建脚本与说明
build_exe.py                双目标 PyInstaller 构建脚本
launcher.py                 GUI 启动入口
backend_launcher.py         WinUI 独立处理后端入口
```

## 从旧版迁移

- Python 包：`nvidia_video_enhancer` → `light_video_enhancer`
- 命令行：`nve` → `lve`
- 源码启动：`python -m light_video_enhancer`
- v0.4.5 的 GUI 配置项有所增加，建议重新检查超分质量、插帧质量、编码 preset 和 NCNN GPU。

## 已知限制

- 驱动 VSR 的增强效果由显卡、驱动设置和输入内容决定，程序无法保证所有设备都启用厂商 AI 模型。
- Vulkan NCNN 常驻 worker 已消除逐批启动与 PNG 中转，但不同模型连续执行仍受显存带宽、模型复杂度和同步点限制。
- FlashVSR 与 SeedVR2 需要多 GiB 权重和独立的新式 CUDA 环境，仅支持 Win10/11 手动使用。
- `ultra` 通常启用 TTA，速度和显存成本可能远高于 `quality`。
- Win7 构建兼容 GUI 和基础流水线，但现代 CUDA、RTX VSR、AV1 硬件编解码是否可用取决于厂商是否仍提供兼容驱动。
- 发布包较大，因为它同时包含 FFmpeg 运行库、多个 NCNN 可执行程序和模型。

## 致谢

项目使用或参考了 [FFmpeg](https://ffmpeg.org/)、[ncnn](https://github.com/Tencent/ncnn)、[rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)、[Real-CUGAN](https://github.com/bilibili/ailab)、[realcugan-ncnn-vulkan](https://github.com/nihui/realcugan-ncnn-vulkan)、[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)、[Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)、[ESRGAN](https://github.com/xinntao/ESRGAN)、[RIFE](https://github.com/hzwer/ECCV2022-RIFE)、[OpenCV](https://github.com/opencv/opencv) 和 [VLC](https://github.com/videolan/vlc)。

项目最初由 TRAE / DeepSeek V4 Pro 辅助创建，后续重构、兼容性修复、性能优化和发布同样使用了 AI 辅助。所有生成或修改内容均应以实际代码、测试和目标硬件验证为准。

## 许可证

项目自身代码采用 [MIT License](LICENSE)。捆绑的 FFmpeg、x264、x265、dav1d、SVT-AV1、libaom、RIFE、Real-CUGAN、Real-ESRGAN、ESRGAN、ncnn 以及厂商 SDK/驱动分别受各自许可证或使用条款约束。当前 FFmpeg 构建启用了 GPL 的 x264/x265，因此分发包需要按 GPL v2 或更高版本履行相应义务；当前构建未启用 `nonfree`。

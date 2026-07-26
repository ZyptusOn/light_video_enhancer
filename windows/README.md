# Light Video Enhancer · WinUI 3

这是 Windows 10/11 的现代前端。它不复制视频算法，而是通过带版本号的标准输入输出协议调用独立的 Python/C++ 后端。

## 系统范围

- 最低系统：Windows 10 1809（build 17763）x64
- 推荐系统：Windows 11 x64
- Windows 7：冻结的 Tk LTS 全量包
- UI：WinUI 3 / Windows App SDK 1.8.6
- 运行方式：未打包、自包含，不要求目标电脑预装 .NET 或 Windows App Runtime

## 开发运行

需要 .NET 10 SDK。调试版会自动向上寻找 `pyproject.toml`，然后运行 `python -m light_video_enhancer`。

```powershell
dotnet run --project windows\LightVideoEnhancer.WinUI\LightVideoEnhancer.WinUI.csproj -p:Platform=x64
```

可用环境变量：

- `LVE_BACKEND`：明确指定后端 EXE
- `LVE_PYTHON`：源码调试时明确指定 Python 解释器
- `LVE_MODEL_DIR`：覆盖用户模型目录
- `LVE_LANG`：`zh-CN` 或 `en-US`

## 生成 Full / Lite 便携包

当前 Python 环境需要安装项目构建依赖和 PyInstaller 6+：

```powershell
python -m pip install -r requirements-build.txt
powershell -ExecutionPolicy Bypass -File windows\build_winui.ps1
```

默认同时生成：

```text
dist\LightVideoEnhancer-WinUI3-Full-Win10-11-x64\
dist\LightVideoEnhancer-WinUI3-Full-Win10-11-x64.zip
dist\LightVideoEnhancer-WinUI3-Lite-Win10-11-x64\
dist\LightVideoEnhancer-WinUI3-Lite-Win10-11-x64.zip
```

只生成一种包可传入 `-Profile Full` 或 `-Profile Lite`。后端已存在时可加 `-SkipBackend`。

两个目录都只有 `LightVideoEnhancer.WinUI.exe`、`LightVideoEnhancer-Backend.exe` 与 `CLI_GUIDE.md` 三个文件。前端使用 Windows App SDK 1.8 支持的自包含单文件发布，并仅保留中英文卫星资源；.NET 与 WinUI 运行库会在首次启动时解压到系统缓存，不再散落于程序目录。

Full 与 Lite 的前端逐字节相同；只有被统一重命名为 `LightVideoEnhancer-Backend.exe` 的后端内容不同。Lite 保留 FFmpeg Worker、Vulkan 执行器和所有传统算法，只排除模型权重。

## 独立后端 CLI

`LightVideoEnhancer-Backend.exe` 是不依赖 GUI 的完整控制台程序，归档中不包含 Tkinter、Tcl/Tk、IDLE 或旧 GUI 模块。双击或不带参数启动会进入交互向导；PowerShell 中可直接使用完整参数：

```powershell
.\LightVideoEnhancer-Backend.exe --help
.\LightVideoEnhancer-Backend.exe --system-info
.\LightVideoEnhancer-Backend.exe input.mp4 -o output.mp4 --scale 2 --fi-multiplier 2 --codec auto --overwrite
```

CLI 默认按 Windows 用户界面语言选择中文或英文，也可用 `--language zh-CN` / `--language en-US` 覆盖。所有面向前端的 JSON 查询保持标准输出纯净，诊断信息写入标准错误。完整说明见包内 `CLI_GUIDE.md`。

## 外部运行环境门控

WinUI 启动时不会自动扫描外部 Python。扫描前会禁用 NVIDIA Video Effects VSR、RIFE PyTorch 与 CUDA 光流；手动扫描后，只有在结果中分别确认 `PyTorch + CUDA + NV-VFX`、`PyTorch`、`PyTorch + CUDA` 时才会解锁。NCNN、D3D11 VSR 与传统 CPU 算法不受影响。

## 体积与部署取舍

当前验证构建中，自包含 WinUI 前端由约 85.6 MiB 降至约 42.4 MiB，Lite 后端约 96.9 MiB。前端仍自带 .NET 与 Windows App SDK，不要求目标电脑额外安装非系统运行库；首次启动会把单文件内容解压到系统缓存。

精简不是简单删除 DLL。构建会排除未使用的 Windows App SDK AI/ML、DirectML、ONNX Runtime、Widgets、NPU 检测和工作负载清单，再启用受控的 `partial` trimming，并把 WinUI、WinRT、Windows SDK 投影和本程序程序集显式设为裁剪根。逐层实验结果如下：

| 实验 | 前端大小 | 结果 |
|---|---:|---|
| 原始自包含单文件 | 约 85.6 MiB | 可用 |
| 排除未使用工作负载 | 约 68.1 MiB | 可用 |
| 无裁剪根的激进裁剪 | 约 34.1 MiB | WinUI 初始化崩溃 |
| 受控裁剪并保留必要根 | 约 42.4 MiB | 页面、主题、本地化、模型列表与后端协议实机通过 |

框架依赖构建还可以更小，但要求用户安装匹配版本的 .NET 与 Windows App Runtime，不符合便携版“解压即用”的目标。Windows App SDK / WinRT 仍会产生 IL2104 裁剪分析警告，因此每次升级 SDK 后必须重新做 GUI 启动、页面导航、语言切换和后端协议回归，不能仅以编译成功判定可发布。

## 模型包

运行：

```powershell
python tools\build_model_packs.py
```

会在 `dist\model-packs` 生成 10 个标准模型 ZIP，并更新 `light_video_enhancer\model_manifest.json` 的归档及逐文件 SHA-256；FlashVSR 与 SeedVR2 的多 GiB 权重继续由固定版本的远程清单按需下载。将标准 ZIP 上传到 GitHub Release `models-v1` 后，GUI 的 GitHub/镜像源即可直接下载；也可随时使用自定义基础 URL、含 `{archive}` 的模板或本地 ZIP 导入。

## 进程协议

- `--capabilities-json`：快速硬件、模型与编码器能力
- `--environments-json --force`：按需扫描 Python / PyTorch / CUDA 环境
- `--models-json`：模型源、路径、大小和安装状态
- `--download-model` / `--install-model-pack` / `--remove-model`：模型管理
- `--progress-json`：输出以 `__LVE_PROGRESS__` 开头的进度 JSON 行
- `--control-stdin`：从标准输入接收 `cancel`

协议版本和兼容规则见 [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)。

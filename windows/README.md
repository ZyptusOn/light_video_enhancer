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

两个目录都只有 `LightVideoEnhancer.WinUI.exe` 与 `LightVideoEnhancer-Backend.exe` 两个文件。前端使用 Windows App SDK 1.8 支持的自包含单文件发布，并仅保留中英文卫星资源；.NET 与 WinUI 运行库会在首次启动时解压到系统缓存，不再散落于程序目录。

Full 与 Lite 的前端逐字节相同；只有被统一重命名为 `LightVideoEnhancer-Backend.exe` 的后端内容不同。Lite 保留 FFmpeg Worker、Vulkan 执行器和所有传统算法，只排除模型权重。

## 外部运行环境门控

WinUI 启动时不会自动扫描外部 Python。扫描前会禁用 NVIDIA Video Effects VSR、RIFE PyTorch 与 CUDA 光流；手动扫描后，只有在结果中分别确认 `PyTorch + CUDA + NV-VFX`、`PyTorch`、`PyTorch + CUDA` 时才会解锁。NCNN、D3D11 VSR 与传统 CPU 算法不受影响。

## 体积与部署取舍

v0.6.0 当前构建中，自包含 WinUI 前端约 85.6 MiB，包含原生 NCNN worker 的 Lite 后端约 86.5 MiB。自包含模式无需目标电脑预装 .NET 或 Windows App SDK，也是保持双 EXE 便携目录的原因。

框架依赖试验构建的前端约 38.6 MiB，但要求用户另外安装 .NET 10 与 Windows App SDK 1.8，且不支持 WinUI 单文件发布。若以后发行 MSIX，可把它作为安装版选项，而不应替换当前便携包。

不要直接启用 `PublishTrimmed`：当前 Windows App SDK / WinRT 投影会产生裁剪警告，实测产物会在 `Microsoft.UI.Xaml` 初始化期间崩溃。正式脚本会在发布前运行 `dotnet clean`，防止不同部署模式的中间文件污染单文件包。

## 模型包

运行：

```powershell
python tools\build_model_packs.py
```

会在 `dist\model-packs` 生成 7 个可下载 ZIP，并更新 `light_video_enhancer\model_manifest.json` 的归档及逐文件 SHA-256。将这些 ZIP 上传到 GitHub Release `models-v1` 后，GUI 的 GitHub/镜像源即可直接下载；也可随时使用自定义基础 URL、含 `{archive}` 的模板或本地 ZIP 导入。

## 进程协议

- `--capabilities-json`：快速硬件、模型与编码器能力
- `--environments-json --force`：按需扫描 Python / PyTorch / CUDA 环境
- `--models-json`：模型源、路径、大小和安装状态
- `--download-model` / `--install-model-pack` / `--remove-model`：模型管理
- `--progress-json`：输出以 `__LVE_PROGRESS__` 开头的进度 JSON 行
- `--control-stdin`：从标准输入接收 `cancel`

协议版本和兼容规则见 [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)。

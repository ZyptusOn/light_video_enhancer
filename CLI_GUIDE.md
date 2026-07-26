# Light Video Enhancer CLI / 命令行指南

`LightVideoEnhancer-Backend.exe` is a complete standalone console application.
It does not require the WinUI frontend, Python, .NET, or Tkinter on the target
computer. Its language follows the Windows display language; use
`--language zh-CN` or `--language en-US` to override it.

`LightVideoEnhancer-Backend.exe` 是完整的独立控制台程序，不依赖 WinUI
前端，也不要求目标电脑安装 Python、.NET 或 Tkinter。界面语言会跟随
Windows 显示语言，也可用 `--language zh-CN` 或 `--language en-US` 覆盖。

## Quick start / 快速开始

Double-click the EXE or run it without arguments to open the interactive wizard:

双击 EXE，或无参数运行，即可进入交互向导：

```powershell
.\LightVideoEnhancer-Backend.exe
```

Process with automatic engine and encoder selection:

使用自动引擎和编码器选择：

```powershell
.\LightVideoEnhancer-Backend.exe input.mp4
```

Choose the main pipeline explicitly:

手动指定主要管线：

```powershell
.\LightVideoEnhancer-Backend.exe input.mp4 -o output.mp4 `
  --scale 2 --sr-engine nvvfx --sr-quality quality `
  --fi-engine rife --fi-multiplier 2 --fi-quality balanced `
  --codec hevc_nvenc --preset balanced --crf 23
```

## Help and diagnostics / 帮助与诊断

```powershell
# Concise processing help / 简洁的处理参数帮助
.\LightVideoEnhancer-Backend.exe --help

# Hardware, encoders, and built-in engines / 硬件、编码器与内置引擎
.\LightVideoEnhancer-Backend.exe --system-info

# Include slower environment checks / 加入较慢的环境检查
.\LightVideoEnhancer-Backend.exe --system-info --deep

# Machine-readable frontend protocol / 供前端读取的 JSON 协议
.\LightVideoEnhancer-Backend.exe --capabilities-json
.\LightVideoEnhancer-Backend.exe --environments-json
.\LightVideoEnhancer-Backend.exe --models-json
```

## Common choices / 常用取值

- Super resolution / 超分：`auto`, `dxva_vsr`, `nvvfx`, `span`,
  `flashvsr`, `seedvr2`, `realcugan`, `realesrgan`, `esrgan`,
  `bicubic`, `lanczos`, `none`
- Interpolation / 插帧：`auto`, `rife`, `ema_vfi`, `rife_ncnn`,
  `ifrnet_ncnn`, `dis`, `optical_flow`, `torch_flow`, `blend`, `none`
- Quality / 质量：`fast`, `balanced`, `quality`, `ultra`
- Container / 容器：`mp4`, `mkv`, `mov`
- NCNN device / NCNN 设备：`auto`, `cpu`, or a Vulkan GPU index
  (`0`, `1`, …)

Run `--help` on the exact build you are using for the authoritative codec list
and defaults. Hardware encoders are selected only when the current machine
reports them as available.

请以当前版本的 `--help` 输出为编码器列表和默认值的准确信息。只有在本机
确实报告可用时，程序才会自动选择硬件编码器。

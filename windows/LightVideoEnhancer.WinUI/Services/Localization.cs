using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;

namespace LightVideoEnhancer_WinUI.Services;

public static class Localization
{
    private static readonly IReadOnlyDictionary<string, string> ZhToEn =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["处理"] = "Process", ["环境与后端"] = "Environment & backend",
            ["模型与下载"] = "Models & downloads", ["关于"] = "About",
            ["视频增强"] = "Video enhancement", ["输入与输出"] = "Input & output",
            ["输入视频"] = "Input video", ["输出文件"] = "Output file",
            ["选择或拖入视频文件"] = "Choose or drop a video file",
            ["留空将自动生成"] = "Leave blank to generate automatically",
            ["浏览…"] = "Browse…", ["另存为…"] = "Save as…",
            ["超分辨率"] = "Super resolution", ["插帧"] = "Frame interpolation",
            ["引擎"] = "Engine", ["倍率"] = "Scale", ["质量"] = "Quality",
            ["自动选择（推荐）"] = "Auto (recommended)", ["自动选择"] = "Auto",
            ["D3D11 驱动 VSR"] = "D3D11 driver VSR", ["ESRGAN 经典模型 ncnn"] = "Classic ESRGAN ncnn",
            ["FlashVSR v1.1（实验）"] = "FlashVSR v1.1 (experimental)",
            ["SeedVR2 3B FP8（重型修复）"] = "SeedVR2 3B FP8 (heavy restoration)",
            ["不改变分辨率"] = "Keep original resolution", ["不插帧"] = "No interpolation",
            ["RIFE AI（PyTorch）"] = "RIFE AI (PyTorch)", ["DIS 稠密光流"] = "DIS dense optical flow",
            ["Farneback 光流"] = "Farneback optical flow", ["CUDA 块匹配光流"] = "CUDA block-matching flow",
            ["快速帧混合"] = "Fast frame blending", ["极速"] = "Fast", ["均衡"] = "Balanced",
            ["极致"] = "Ultra", ["先超分再插帧（显存占用更高）"] = "Super-resolve before interpolation (uses more VRAM)",
            ["编码、设备与片段"] = "Encoding, device & segment", ["编码器"] = "Encoder",
            ["容器"] = "Container", ["自动"] = "Auto", ["NCNN 设备"] = "NCNN device",
            ["开始（秒）"] = "Start (seconds)", ["时长（秒）"] = "Duration (seconds)", ["可选"] = "Optional",
            ["复制源音频"] = "Copy source audio", ["覆盖已有输出"] = "Overwrite existing output",
            ["外部 PyTorch / CUDA Python（可选）"] = "External PyTorch / CUDA Python (optional)",
            ["自动检测，或指定 python.exe"] = "Auto-detect or choose python.exe", ["选择…"] = "Choose…",
            ["任务"] = "Task", ["就绪"] = "Ready", ["打开输出位置"] = "Open output location",
            ["取消"] = "Cancel", ["开始处理"] = "Start processing",
            ["处理后端"] = "Processing backend", ["刷新快速检测"] = "Refresh quick check",
            ["扫描 Python / PyTorch 环境"] = "Scan Python / PyTorch environments",
            ["硬件与可用能力"] = "Hardware & available capabilities", ["Python 环境"] = "Python environments",
            ["下载源"] = "Download source", ["来源"] = "Source", ["官方（GitHub / Hugging Face）"] = "Official (GitHub / Hugging Face)",
            ["镜像（GitHub Proxy / HF Mirror）"] = "Mirror (GitHub Proxy / HF Mirror)", ["自定义地址"] = "Custom URL",
            ["自定义基础地址或 {archive} 模板"] = "Custom base URL or {archive} template",
            ["刷新状态"] = "Refresh status", ["正在读取模型状态…"] = "Reading model status…",
            ["导入 ZIP"] = "Import ZIP", ["移除"] = "Remove", ["下载"] = "Download",
            ["打开 GitHub 仓库"] = "Open GitHub repository", ["界面主题"] = "Theme",
            ["跟随系统"] = "Use system setting", ["浅色"] = "Light", ["深色"] = "Dark",
            ["界面语言"] = "Language", ["中文"] = "Chinese", ["英文"] = "English",
            ["在独立处理进程中组合超分、插帧与硬件编码。界面关闭或取消时不会把 CUDA / Vulkan 运行时留在主进程。"] =
                "Combine super resolution, interpolation, and hardware encoding in an isolated process. Closing or cancelling the UI leaves no CUDA or Vulkan runtime in the frontend process.",
            ["可将视频直接拖到此卡片。选择输入后会自动建议输出文件名。"] =
                "Drop a video onto this card. An output name is suggested automatically after input selection.",
            ["只影响支持质量档位的超分引擎。"] = "Only affects super-resolution engines with quality presets.",
            ["光流类引擎支持质量档位；RIFE 模型使用自身固定参数。"] =
                "Optical-flow engines support quality presets; RIFE models use fixed model parameters.",
            ["综合设备、已扫描环境、目标尺寸、质量档和处理顺序评分；不会自动启用重型实验模型。"] =
                "Scores hardware, scanned runtimes, target size, quality preset, and stage order; heavy experimental models are never enabled automatically.",
            ["自动编码器会根据显卡厂商选择可用硬件后端，并在同格式内依次回退。"] =
                "Auto encoder selects an available hardware backend by GPU vendor and falls back within the same format.",
            ["快速检测不会导入 PyTorch；完整扫描只会在你手动触发时运行，并写入 24 小时缓存。"] =
                "Quick checks do not import PyTorch. A full scan runs only on request and is cached for 24 hours.",
            ["检测结果按 CUDA + NVIDIA VFX、CUDA、PyTorch 的可用性排序。可把合适的 python.exe 填入处理页。"] =
                "Results are ranked by CUDA + NVIDIA VFX, CUDA, and PyTorch availability. Choose a suitable python.exe on the Process page.",
            ["轻量版可按需下载权重；全量版使用同一界面并将内置权重显示为已安装。模型保存在当前用户目录，程序升级不会覆盖。"] =
                "The Lite package downloads weights on demand. The Full package uses the same UI and shows bundled weights as installed. User models survive application updates.",
            ["WinUI 3 前端面向 Windows 10 1809 及以上系统；Windows 7 继续使用原有 Tk 前端。两个界面共享相同的 Python、C/C++、NCNN、CUDA 与 FFmpeg 处理核心。"] =
                "The WinUI 3 frontend supports Windows 10 version 1809 and newer. Windows 7 keeps the legacy Tk frontend. Both share the same Python, C/C++, NCNN, CUDA, and FFmpeg processing core.",
        };

    private static readonly IReadOnlyDictionary<string, string> EnToZh =
        ZhToEn.GroupBy(pair => pair.Value, StringComparer.Ordinal).ToDictionary(
            group => group.Key, group => group.OrderBy(pair => pair.Key.Length).First().Key,
            StringComparer.Ordinal);

    public static string Text(string value, bool chinese)
    {
        if (chinese)
        {
            return EnToZh.TryGetValue(value, out string? translated) ? translated : value;
        }
        return ZhToEn.TryGetValue(value, out string? english) ? english : value;
    }

    public static void Apply(DependencyObject root, bool chinese)
    {
        switch (root)
        {
            case TextBlock text:
                text.Text = Text(text.Text, chinese);
                break;
            case TextBox box:
                box.Header = TranslateObject(box.Header, chinese);
                box.PlaceholderText = Text(box.PlaceholderText, chinese);
                break;
            case ComboBox combo:
                combo.Header = TranslateObject(combo.Header, chinese);
                break;
            case NumberBox number:
                number.Header = TranslateObject(number.Header, chinese);
                break;
            case ContentControl content:
                content.Content = TranslateObject(content.Content, chinese);
                break;
        }
        object? tooltip = ToolTipService.GetToolTip(root);
        if (tooltip is string tooltipText)
        {
            ToolTipService.SetToolTip(root, Text(tooltipText, chinese));
        }
        int count = VisualTreeHelper.GetChildrenCount(root);
        for (int index = 0; index < count; index++)
        {
            Apply(VisualTreeHelper.GetChild(root, index), chinese);
        }
    }

    private static object? TranslateObject(object? value, bool chinese) =>
        value is string text ? Text(text, chinese) : value;
}

using System.Text.Json;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace LightVideoEnhancer_WinUI;

public sealed partial class MainPage
{
    private bool _emaVfiModelAvailable;
    private bool _flashVsrModelAvailable;
    private string? _flashVsrPython;
    private bool _seedVr2ModelAvailable;
    private string? _seedVr2Python;

    private void UpdateExternalEngineAvailability()
    {
        bool scanned = _environmentsJson is not null;
        bool hasTorch = false;
        bool hasCudaTorch = false;
        bool hasNvVfx = false;
        bool hasFlashVsr = false;
        bool hasSeedVr2 = false;

        if (scanned)
        {
            try
            {
                using JsonDocument document = JsonDocument.Parse(_environmentsJson!);
                if (document.RootElement.ValueKind != JsonValueKind.Array)
                {
                    scanned = false;
                }
                else
                {
                    foreach (JsonElement environment in document.RootElement.EnumerateArray())
                    {
                        bool torch = BoolProperty(environment, "torch");
                        bool cuda = BoolProperty(environment, "cuda");
                        hasTorch |= torch;
                        hasCudaTorch |= torch && cuda;
                        hasNvVfx |= torch && cuda && BoolProperty(environment, "nvvfx");
                        hasFlashVsr |= torch && cuda && BoolProperty(environment, "flashvsr");
                        hasSeedVr2 |= torch && cuda && BoolProperty(environment, "seedvr2");
                    }
                }
            }
            catch (JsonException)
            {
                scanned = false;
            }
        }

        string notScanned = T(
            "请先在“环境与后端”页手动扫描 Python / PyTorch 环境。",
            "Run the manual Python / PyTorch scan on the Environment & backend page first.");
        SetExternalEngineState(
            SrEngineBox, NvVfxSrItem, scanned && hasNvVfx,
            scanned ? T(
                "扫描结果中没有同时支持 PyTorch、CUDA 与 NVIDIA VFX 的环境。",
                "No scanned environment provides PyTorch, CUDA, and NVIDIA VFX together.") : notScanned);
        SetExternalEngineState(
            SrEngineBox, FlashVsrSrItem,
            scanned && hasFlashVsr && _flashVsrModelAvailable,
            scanned ? T(
                "需要 Python 3.11 CUDA、Block-Sparse Attention 与 FlashVSR 模型包。",
                "Python 3.11 CUDA, Block-Sparse Attention, and the FlashVSR model pack are required.")
                : notScanned);
        SetExternalEngineState(
            SrEngineBox, SeedVr2SrItem,
            scanned && hasSeedVr2 && _seedVr2ModelAvailable,
            scanned ? T(
                "需要兼容的 CUDA PyTorch 环境与 SeedVR2 3B FP8 模型包。",
                "A compatible CUDA PyTorch environment and the SeedVR2 3B FP8 model pack are required.")
                : notScanned);
        SetExternalEngineState(
            FiEngineBox, RifeFiItem, scanned && hasTorch,
            scanned ? T(
                "扫描结果中没有可用的 PyTorch 环境。",
                "No usable PyTorch environment was found by the scan.") : notScanned);
        SetExternalEngineState(
            FiEngineBox, EmaVfiFiItem,
            scanned && hasCudaTorch && _emaVfiModelAvailable,
            scanned ? T(
                "需要 CUDA PyTorch 环境与 EMA-VFI Small 模型包。",
                "A CUDA PyTorch environment and the EMA-VFI Small model pack are required.") : notScanned);
        SetExternalEngineState(
            FiEngineBox, TorchFlowFiItem, scanned && hasCudaTorch,
            scanned ? T(
                "扫描结果中没有支持 CUDA 的 PyTorch 环境。",
                "No scanned PyTorch environment has CUDA support.") : notScanned);
    }

    private void UpdateBuiltInEngineAvailability(JsonElement capabilities)
    {
        _emaVfiModelAvailable = BoolProperty(capabilities, "ema_vfi_model");
        _flashVsrModelAvailable = BoolProperty(capabilities, "flashvsr_model");
        _seedVr2ModelAvailable = BoolProperty(capabilities, "seedvr2_model");
        bool ifrnet = BoolProperty(capabilities, "ncnn_ifrnet");
        bool span = BoolProperty(capabilities, "ncnn_span");
        SetExternalEngineState(
            SrEngineBox, SpanSrItem, span,
            T(
                "SPAN 原生 Worker 或模型尚未安装；请在“模型与下载”页安装 SPAN 模型包。",
                "The SPAN native worker or models are unavailable. Install the SPAN model pack on Models & downloads."));
        SetExternalEngineState(
            FiEngineBox, IfrNetFiItem, ifrnet,
            T(
                "IFRNet 原生 Worker 或三档模型尚未安装；请在“模型与下载”页安装 IFRNet 模型包。",
                "The IFRNet native worker or all three model presets are unavailable. Install the IFRNet model pack on Models & downloads."));
    }

    private static void SetExternalEngineState(
        ComboBox owner, ComboBoxItem item, bool enabled, string unavailableReason)
    {
        item.IsEnabled = enabled;
        ToolTipService.SetToolTip(item, enabled ? null : unavailableReason);
        if (!enabled && ReferenceEquals(owner.SelectedItem, item))
        {
            owner.SelectedIndex = 0;
        }
    }

    private void ValidateExternalEngineSelection(string srEngine, string fiEngine)
    {
        if (srEngine == "nvvfx" && !NvVfxSrItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "NVIDIA Video Effects VSR 尚未通过手动环境扫描。",
                "NVIDIA Video Effects VSR has not been enabled by a manual environment scan."));
        }
        if (srEngine == "flashvsr" && !FlashVsrSrItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "FlashVSR 尚未通过环境与模型检查。",
                "FlashVSR has not passed the environment and model checks."));
        }
        if (srEngine == "seedvr2" && !SeedVr2SrItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "SeedVR2 尚未通过环境与模型检查。",
                "SeedVR2 has not passed the environment and model checks."));
        }
        if (fiEngine == "rife" && !RifeFiItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "RIFE AI 尚未通过手动 PyTorch 环境扫描。",
                "RIFE AI has not been enabled by a manual PyTorch environment scan."));
        }
        if (fiEngine == "ema_vfi" && !EmaVfiFiItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "EMA-VFI Small 尚未通过环境与模型检查。",
                "EMA-VFI Small has not passed the environment and model checks."));
        }
        if (fiEngine == "torch_flow" && !TorchFlowFiItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "CUDA 光流尚未通过手动 CUDA PyTorch 环境扫描。",
                "CUDA optical flow has not been enabled by a manual CUDA PyTorch environment scan."));
        }
    }
}

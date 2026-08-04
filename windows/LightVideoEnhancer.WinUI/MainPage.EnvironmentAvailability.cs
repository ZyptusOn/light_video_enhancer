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
    private bool _dloralModelAvailable;
    private string? _dloralPython;
    private bool _osdEnhancerModelAvailable;
    private string? _osdEnhancerPython;
    private bool _sparkVsrModelAvailable;
    private string? _sparkVsrPython;
    private bool _vfiMambaModelAvailable;
    private string? _vfiMambaPython;

    private void UpdateExternalEngineAvailability()
    {
        bool scanned = _environmentsJson is not null;
        bool hasTorch = false;
        bool hasCudaTorch = false;
        bool hasNvVfx = false;
        bool hasFlashVsr = false;
        bool hasSeedVr2 = false;
        bool hasDloral = false;
        bool hasOsdEnhancer = false;
        bool hasSparkVsr = false;
        bool hasVfiMamba = false;

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
                        hasDloral |= torch && cuda && BoolProperty(environment, "dloral");
                        hasOsdEnhancer |= torch && cuda && BoolProperty(environment, "osdenhancer");
                        hasSparkVsr |= torch && cuda && BoolProperty(environment, "sparkvsr");
                        hasVfiMamba |= torch && cuda && BoolProperty(environment, "vfimamba");
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
            SrEngineBox, DloralSrItem,
            scanned && hasDloral && _dloralModelAvailable,
            scanned ? T(
                "需要兼容的 CUDA PyTorch 环境与约 8.1 GiB DLoRAL 核心模型包；仅支持原生 4×。",
                "A compatible CUDA PyTorch environment and the roughly 8.1 GiB DLoRAL core pack are required; native 4x only.")
                : notScanned);
        SetExternalEngineState(
            SrEngineBox, OsdEnhancerSrItem,
            scanned && hasOsdEnhancer && _osdEnhancerModelAvailable,
            scanned ? T(
                "需要约 12.0 GiB 模型包、兼容的 CUDA PyTorch 环境及至少 80 GB 显存；固定为 4× 超分和 2× 插帧。",
                "The roughly 12.0 GiB model pack, a compatible CUDA PyTorch environment, and at least 80 GB VRAM are required; fixed 4x SR and 2x interpolation.")
                : notScanned);
        SetExternalEngineState(
            SrEngineBox, SparkVsrSrItem,
            scanned && hasSparkVsr && _sparkVsrModelAvailable,
            scanned ? T(
                "需要约 39.3 GiB 模型、兼容的 CUDA PyTorch 环境；低于 40 GiB 显存时安全门还要求至少 56 GiB 内存。",
                "The roughly 39.3 GiB model, a compatible CUDA PyTorch environment, and (below 40 GiB VRAM) at least 56 GiB system RAM are required by the safety gate.")
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
            FiEngineBox, VfiMambaFiItem,
            scanned && hasVfiMamba && _vfiMambaModelAvailable,
            scanned ? T(
                "需要兼容的 CUDA PyTorch 环境、timm/einops 与 VFIMamba 模型包。",
                "A compatible CUDA PyTorch environment, timm/einops, and the VFIMamba model pack are required.") : notScanned);
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
        _dloralModelAvailable = BoolProperty(capabilities, "dloral_model");
        _osdEnhancerModelAvailable = BoolProperty(capabilities, "osdenhancer_model");
        _sparkVsrModelAvailable = BoolProperty(capabilities, "sparkvsr_model");
        _vfiMambaModelAvailable = BoolProperty(capabilities, "vfimamba_model");
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
        if (srEngine == "dloral" && !DloralSrItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "DLoRAL 尚未通过环境与模型检查。",
                "DLoRAL has not passed the environment and model checks."));
        }
        if (srEngine == "osdenhancer" && !OsdEnhancerSrItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "OSDEnhancer 尚未通过环境、模型与显存检查。",
                "OSDEnhancer has not passed the environment, model, and VRAM checks."));
        }
        if (srEngine == "sparkvsr" && !SparkVsrSrItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "SparkVSR 尚未通过环境、模型与内存安全检查。",
                "SparkVSR has not passed the environment, model, and memory-safety checks."));
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
        if (fiEngine == "vfimamba" && !VfiMambaFiItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "VFIMamba 尚未通过环境与模型检查。",
                "VFIMamba has not passed the environment and model checks."));
        }
        if (fiEngine == "torch_flow" && !TorchFlowFiItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "CUDA 光流尚未通过手动 CUDA PyTorch 环境扫描。",
                "CUDA optical flow has not been enabled by a manual CUDA PyTorch environment scan."));
        }
    }
}

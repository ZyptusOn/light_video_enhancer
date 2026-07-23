using System.Text.Json;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace LightVideoEnhancer_WinUI;

public sealed partial class MainPage
{
    private void UpdateExternalEngineAvailability()
    {
        bool scanned = _environmentsJson is not null;
        bool hasTorch = false;
        bool hasCudaTorch = false;
        bool hasNvVfx = false;

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
            FiEngineBox, RifeFiItem, scanned && hasTorch,
            scanned ? T(
                "扫描结果中没有可用的 PyTorch 环境。",
                "No usable PyTorch environment was found by the scan.") : notScanned);
        SetExternalEngineState(
            FiEngineBox, TorchFlowFiItem, scanned && hasCudaTorch,
            scanned ? T(
                "扫描结果中没有支持 CUDA 的 PyTorch 环境。",
                "No scanned PyTorch environment has CUDA support.") : notScanned);
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
        if (fiEngine == "rife" && !RifeFiItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "RIFE AI 尚未通过手动 PyTorch 环境扫描。",
                "RIFE AI has not been enabled by a manual PyTorch environment scan."));
        }
        if (fiEngine == "torch_flow" && !TorchFlowFiItem.IsEnabled)
        {
            throw new ArgumentException(T(
                "CUDA 光流尚未通过手动 CUDA PyTorch 环境扫描。",
                "CUDA optical flow has not been enabled by a manual CUDA PyTorch environment scan."));
        }
    }
}

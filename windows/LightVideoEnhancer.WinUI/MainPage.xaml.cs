using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using LightVideoEnhancer_WinUI.Services;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Windows.ApplicationModel.DataTransfer;
using Windows.Storage;
using Windows.Storage.Pickers;
using WinRT.Interop;

namespace LightVideoEnhancer_WinUI;

public sealed class ModelPackViewModel
{
    public required string Id { get; init; }
    public required string DisplayName { get; init; }
    public required string Description { get; init; }
    public required string StatusText { get; init; }
    public required string ImportText { get; init; }
    public required string RemoveText { get; init; }
    public required string DownloadText { get; init; }
    public bool CanDownload { get; init; }
    public bool CanRemove { get; init; }
}

public sealed partial class MainPage : Page
{
    private readonly BackendProcess _backend = new();
    private readonly StringBuilder _logText = new();
    private string _lastSuggestedOutput = string.Empty;
    private string? _capabilitiesJson;
    private string? _environmentsJson;
    private bool _loaded;
    private readonly ObservableCollection<ModelPackViewModel> _modelPacks = [];
    private bool _modelOperation;
    private string _language = CultureInfo.CurrentUICulture.TwoLetterISOLanguageName == "zh" ? "zh-CN" : "en-US";

    public MainPage()
    {
        InitializeComponent();
        ModelPacksList.ItemsSource = _modelPacks;
        _backend.OutputReceived += Backend_OutputReceived;
        _backend.ProgressReceived += Backend_ProgressReceived;
        Loaded += MainPage_Loaded;
        Unloaded += (_, _) => _backend.Dispose();
    }

    private async void MainPage_Loaded(object sender, RoutedEventArgs e)
    {
        if (_loaded)
        {
            return;
        }
        _loaded = true;
        ApplyLocalization();
        UpdateExternalEngineAvailability();
        RenderBackendPath();
        await RefreshCapabilitiesAsync();
        await RefreshModelsAsync();
    }

    private void MainNavigation_SelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        string tag = (args.SelectedItem as NavigationViewItem)?.Tag?.ToString() ?? "process";
        ProcessView.Visibility = tag == "process" ? Visibility.Visible : Visibility.Collapsed;
        EnvironmentView.Visibility = tag == "environment" ? Visibility.Visible : Visibility.Collapsed;
        ModelsView.Visibility = tag == "models" ? Visibility.Visible : Visibility.Collapsed;
        AboutView.Visibility = tag == "about" ? Visibility.Visible : Visibility.Collapsed;
        FrameworkElement selectedView = tag switch
        {
            "environment" => EnvironmentView,
            "models" => ModelsView,
            "about" => AboutView,
            _ => ProcessView,
        };
        if (_loaded)
        {
            // Collapsed views do not have a complete WinUI visual tree. Force the
            // selected page to materialize, then run one low-priority pass after
            // layout so every lazily-created label is translated.
            selectedView.UpdateLayout();
            ApplyLocalization();
            DispatcherQueue.TryEnqueue(Microsoft.UI.Dispatching.DispatcherQueuePriority.Low, () =>
            {
                selectedView.UpdateLayout();
                ApplyLocalization();
            });
        }
    }

    private async void BrowseInput_Click(object sender, RoutedEventArgs e)
    {
        FileOpenPicker picker = CreateOpenPicker(T("选择输入视频", "Choose input video"), ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts");
        StorageFile? file = await picker.PickSingleFileAsync();
        if (file is not null)
        {
            InputPathBox.Text = file.Path;
            SuggestOutputPath();
        }
    }

    private async void BrowseOutput_Click(object sender, RoutedEventArgs e)
    {
        string container = SelectedTag(ContainerBox, "mp4");
        FileSavePicker picker = new()
        {
            SuggestedStartLocation = PickerLocationId.VideosLibrary,
            SuggestedFileName = SuggestedStem(),
        };
        picker.FileTypeChoices.Add(container.ToUpperInvariant(), ["." + container]);
        InitializeWithWindow.Initialize(picker, MainWindowHandle());
        StorageFile? file = await picker.PickSaveFileAsync();
        if (file is not null)
        {
            OutputPathBox.Text = file.Path;
            _lastSuggestedOutput = file.Path;
        }
    }

    private async void BrowseTorchPython_Click(object sender, RoutedEventArgs e)
    {
        FileOpenPicker picker = CreateOpenPicker(T("选择 PyTorch 环境中的 python.exe", "Choose python.exe from a PyTorch environment"), ".exe");
        StorageFile? file = await picker.PickSingleFileAsync();
        if (file is not null)
        {
            TorchPythonBox.Text = file.Path;
        }
    }

    private void InputCard_DragOver(object sender, DragEventArgs e)
    {
        e.AcceptedOperation = DataPackageOperation.Copy;
        e.DragUIOverride.Caption = T("使用此视频", "Use this video");
        e.DragUIOverride.IsCaptionVisible = true;
    }

    private async void InputCard_Drop(object sender, DragEventArgs e)
    {
        if (!e.DataView.Contains(StandardDataFormats.StorageItems))
        {
            return;
        }
        IReadOnlyList<IStorageItem> items = await e.DataView.GetStorageItemsAsync();
        StorageFile? file = items.OfType<StorageFile>().FirstOrDefault();
        if (file is not null)
        {
            InputPathBox.Text = file.Path;
            SuggestOutputPath();
        }
    }

    private void InputPathBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        SuggestOutputPath();
    }

    private void QualitySelection_Changed(object sender, SelectionChangedEventArgs e)
    {
        if (SrQualityBox is null || FiQualityBox is null)
        {
            return;
        }
        string sr = SelectedTag(SrEngineBox, "auto");
        string fi = SelectedTag(FiEngineBox, "auto");
        SrQualityBox.IsEnabled = sr is "auto" or "nvvfx" or "span" or "flashvsr" or "seedvr2" or "realcugan" or "realesrgan" or "esrgan";
        FiQualityBox.IsEnabled = fi is "auto" or "ema_vfi" or "dis" or "optical_flow" or "torch_flow";
        SuggestOutputPath();
    }

    private void CodecBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (PresetBox is null)
        {
            return;
        }
        string codec = SelectedTag(CodecBox, "auto");
        PresetBox.Text = codec.Contains("nvenc", StringComparison.Ordinal) ? "p5" :
            codec == "auto" ? "balanced" : "medium";
    }

    private async void Start_Click(object sender, RoutedEventArgs e)
    {
        if (_backend.IsRunning)
        {
            return;
        }

        IReadOnlyList<string> arguments;
        try
        {
            arguments = BuildArguments();
        }
        catch (Exception exception) when (exception is ArgumentException or IOException)
        {
            ShowInfo(T("无法开始处理", "Cannot start processing"), exception.Message, InfoBarSeverity.Error);
            return;
        }

        SetProcessingState(true);
        _logText.Clear();
        LogBox.Text = string.Empty;
        TaskProgress.IsIndeterminate = true;
        StatusText.Text = T("正在初始化处理管线…", "Initializing processing pipeline…");
        BackendInfoBar.IsOpen = false;

        try
        {
            int exitCode = await _backend.RunAsync(arguments);
            if (exitCode == 0)
            {
                TaskProgress.IsIndeterminate = false;
                TaskProgress.Maximum = 100;
                TaskProgress.Value = 100;
                StatusText.Text = T("处理完成", "Processing complete");
                OpenOutputButton.IsEnabled = true;
                ShowInfo(T("处理完成", "Processing complete"), OutputPathBox.Text, InfoBarSeverity.Success);
            }
            else if (exitCode == 130)
            {
                TaskProgress.IsIndeterminate = false;
                StatusText.Text = T("已取消", "Cancelled");
                ShowInfo(T("任务已取消", "Task cancelled"), T("后端已释放资源并清理未完成文件。", "The backend released its resources and removed incomplete output."), InfoBarSeverity.Warning);
            }
            else
            {
                TaskProgress.IsIndeterminate = false;
                StatusText.Text = IsChinese ? $"处理失败（退出代码 {exitCode}）" : $"Processing failed (exit code {exitCode})";
                ShowInfo(T("处理未完成", "Processing did not complete"), T("请查看下方日志中的最后一条错误。", "See the final error in the log below."), InfoBarSeverity.Error);
            }
        }
        catch (Exception exception)
        {
            TaskProgress.IsIndeterminate = false;
            StatusText.Text = T("无法启动或连接处理后端", "Cannot start or connect to the processing backend");
            AppendLog("[ERROR] " + exception.Message);
            ShowInfo(T("后端错误", "Backend error"), exception.Message, InfoBarSeverity.Error);
        }
        finally
        {
            SetProcessingState(false);
        }
    }

    private async void Cancel_Click(object sender, RoutedEventArgs e)
    {
        CancelButton.IsEnabled = false;
        StatusText.Text = T("正在安全停止并封装已编码数据…", "Stopping safely and finalizing encoded data…");
        await _backend.CancelAsync();
    }

    private void OpenOutput_Click(object sender, RoutedEventArgs e)
    {
        string output = OutputPathBox.Text.Trim();
        if (string.IsNullOrEmpty(output))
        {
            return;
        }
        string directory = Path.GetDirectoryName(output) ?? Environment.CurrentDirectory;
        ProcessStartInfo startInfo = new("explorer.exe") { UseShellExecute = true };
        if (File.Exists(output))
        {
            startInfo.ArgumentList.Add("/select,");
            startInfo.ArgumentList.Add(output);
        }
        else
        {
            startInfo.ArgumentList.Add(directory);
        }
        Process.Start(startInfo);
    }

    private void ModelSourceBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (CustomSourceBox is not null)
        {
            CustomSourceBox.IsEnabled = SelectedTag(ModelSourceBox, "github") == "custom";
        }
    }

    private async void RefreshModels_Click(object sender, RoutedEventArgs e)
    {
        await RefreshModelsAsync();
    }

    private async Task RefreshModelsAsync()
    {
        if (_backend.IsRunning)
        {
            return;
        }
        try
        {
            BackendCommandResult result = await _backend.QueryAsync(
                "--language", _language, "--models-json");
            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException(result.StandardError.Trim());
            }
            RenderModels(result.StandardOutput);
        }
        catch (Exception exception)
        {
            ModelStatusText.Text = T("无法读取模型状态：", "Cannot read model status: ") + exception.Message;
            ShowModelInfo(T("模型后端错误", "Model backend error"), exception.Message, InfoBarSeverity.Error);
        }
    }

    private async void DownloadModel_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not Button { Tag: string packId })
        {
            return;
        }
        List<string> arguments =
        [
            "--download-model", packId,
            "--model-source", SelectedTag(ModelSourceBox, "github"),
        ];
        if (SelectedTag(ModelSourceBox, "github") == "custom")
        {
            arguments.Add("--source-base");
            arguments.Add(CustomSourceBox.Text.Trim());
        }
        await RunModelCommandAsync(arguments, T("正在下载并校验模型…", "Downloading and verifying model…"));
    }

    private async void ImportModel_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not Button { Tag: string packId })
        {
            return;
        }
        FileOpenPicker picker = CreateOpenPicker(T("选择模型 ZIP", "Choose model ZIP"), ".zip");
        StorageFile? file = await picker.PickSingleFileAsync();
        if (file is not null)
        {
            await RunModelCommandAsync(
                ["--install-model-pack", packId, file.Path],
                T("正在校验并安装本地模型包…", "Verifying and installing local model pack…"));
        }
    }

    private async void RemoveModel_Click(object sender, RoutedEventArgs e)
    {
        if (sender is Button { Tag: string packId })
        {
            await RunModelCommandAsync(["--remove-model", packId], T("正在移除已下载模型…", "Removing downloaded model…"));
        }
    }

    private async Task RunModelCommandAsync(IReadOnlyList<string> arguments, string status)
    {
        if (_backend.IsRunning)
        {
            ShowModelInfo(T("后端忙碌", "Backend busy"), T("请先等待当前处理或检测任务结束。", "Wait for the current processing or detection task to finish."), InfoBarSeverity.Warning);
            return;
        }
        _modelOperation = true;
        ModelPacksList.IsEnabled = false;
        ModelProgress.IsIndeterminate = true;
        ModelProgress.Value = 0;
        ModelStatusText.Text = status;
        ModelInfoBar.IsOpen = false;
        try
        {
            List<string> localizedArguments = ["--language", _language];
            localizedArguments.AddRange(arguments);
            int exitCode = await _backend.RunAsync(localizedArguments);
            if (exitCode != 0)
            {
                throw new InvalidOperationException(IsChinese ? $"模型命令失败（退出代码 {exitCode}）。" : $"Model command failed (exit code {exitCode}).");
            }
            ShowModelInfo(T("模型已就绪", "Model ready"), T("校验完成，处理引擎无需重启即可使用。", "Verification completed. The processing engine can use it without restarting."), InfoBarSeverity.Success);
        }
        catch (Exception exception)
        {
            ShowModelInfo(T("模型操作失败", "Model operation failed"), exception.Message, InfoBarSeverity.Error);
        }
        finally
        {
            _modelOperation = false;
            ModelProgress.IsIndeterminate = false;
            ModelPacksList.IsEnabled = true;
            await RefreshModelsAsync();
            await RefreshCapabilitiesAsync();
        }
    }

    private void RenderModels(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        if (!root.TryGetProperty("protocol_version", out JsonElement modelProtocol) ||
            modelProtocol.GetInt32() != 1)
        {
            throw new InvalidOperationException(T("模型协议版本不兼容。", "The model protocol version is incompatible."));
        }
        bool chinese = IsChinese;
        _modelPacks.Clear();
        if (root.TryGetProperty("model_root", out JsonElement modelRoot))
        {
            ModelRootText.Text = T("模型目录：", "Model directory: ") + modelRoot.GetString();
        }
        int installedCount = 0;
        foreach (JsonElement pack in root.GetProperty("packs").EnumerateArray())
        {
            string status = StringProperty(pack, "status", "missing");
            long installedSize = Int64Property(pack, "installed_size");
            long downloadSize = Int64Property(pack, "download_size");
            JsonElement names = pack.GetProperty("name");
            JsonElement descriptions = pack.GetProperty("description");
            string language = chinese ? "zh-CN" : "en-US";
            string statusText = status switch
            {
                "bundled" => (chinese ? "已内置" : "Included") + " · " + FormatBytes(installedSize),
                "downloaded" => (chinese ? "已下载" : "Downloaded") + " · " + FormatBytes(installedSize),
                "partial" => chinese ? "文件不完整，请重新下载" : "Incomplete; download again",
                _ => (chinese ? "未安装" : "Not installed") + " · " + FormatBytes(downloadSize),
            };
            bool installed = status is "bundled" or "downloaded";
            if (installed)
            {
                installedCount++;
            }
            _modelPacks.Add(new ModelPackViewModel
            {
                Id = StringProperty(pack, "id", ""),
                DisplayName = StringProperty(names, language, StringProperty(names, "en-US", "?")),
                Description = StringProperty(descriptions, language, ""),
                StatusText = statusText,
                ImportText = chinese ? "导入 ZIP" : "Import ZIP",
                RemoveText = chinese ? "移除" : "Remove",
                DownloadText = chinese ? "下载" : "Download",
                CanDownload = !installed,
                CanRemove = status is "downloaded" or "partial",
            });
        }
        ModelStatusText.Text = chinese
            ? $"已安装 {installedCount} / {_modelPacks.Count} 个模型包"
            : $"{installedCount} of {_modelPacks.Count} model packs installed";
    }

    private static string FormatBytes(long value)
    {
        if (value <= 0)
        {
            return "—";
        }
        return value >= 1073741824
            ? $"{value / 1073741824.0:F1} GiB"
            : $"{value / 1048576.0:F1} MiB";
    }

    private void ShowModelInfo(string title, string message, InfoBarSeverity severity)
    {
        title = title.Trim();
        message = message.Trim();
        if (title.Length == 0 && message.Length == 0)
        {
            HideModelInfo();
            return;
        }
        ModelInfoBar.Title = title;
        ModelInfoBar.Message = message;
        ModelInfoBar.Severity = severity;
        ModelInfoBar.Visibility = Visibility.Visible;
        ModelInfoBar.IsOpen = true;
    }

    private void HideModelInfo()
    {
        ModelInfoBar.IsOpen = false;
        ModelInfoBar.Visibility = Visibility.Collapsed;
        ModelInfoBar.Title = string.Empty;
        ModelInfoBar.Message = string.Empty;
    }

    private void ModelInfoBar_Closed(InfoBar sender, InfoBarClosedEventArgs args)
    {
        ModelInfoBar.Visibility = Visibility.Collapsed;
    }

    private async void RefreshCapabilities_Click(object sender, RoutedEventArgs e)
    {
        await RefreshCapabilitiesAsync();
    }

    private async Task RefreshCapabilitiesAsync()
    {
        if (_backend.IsRunning)
        {
            ShowInfo(T("后端忙碌", "Backend busy"), T("请等待当前任务结束后再刷新能力。", "Wait for the current task before refreshing capabilities."), InfoBarSeverity.Informational);
            return;
        }

        BackendBadgeText.Text = T("正在检测", "Detecting");
        try
        {
            BackendCommandResult result = await _backend.QueryAsync("--capabilities-json");
            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException(result.StandardError.Trim());
            }
            _capabilitiesJson = result.StandardOutput;
            RenderCapabilities(_capabilitiesJson);
            BackendBadgeText.Text = T("后端已就绪", "Backend ready");
            HideInfo();
        }
        catch (Exception exception)
        {
            BackendBadgeText.Text = T("后端不可用", "Backend unavailable");
            CapabilitiesBox.Text = exception.Message;
            ShowInfo(T("无法连接处理后端", "Cannot connect to processing backend"), exception.Message, InfoBarSeverity.Error);
        }
    }

    private async void ScanEnvironments_Click(object sender, RoutedEventArgs e)
    {
        if (_backend.IsRunning)
        {
            ShowInfo(T("后端忙碌", "Backend busy"), T("请等待当前任务结束后再扫描环境。", "Wait for the current task before scanning environments."), InfoBarSeverity.Informational);
            return;
        }
        ScanEnvironmentsButton.IsEnabled = false;
        _environmentsJson = null;
        UpdateExternalEngineAvailability();
        if (!string.IsNullOrWhiteSpace(_capabilitiesJson))
        {
            RenderCapabilities(_capabilitiesJson);
        }
        EnvironmentResultsBox.Text = T("正在并行扫描 Python、Conda、uv、pyenv 与虚拟环境…", "Scanning Python, Conda, uv, pyenv, and virtual environments in parallel…");
        try
        {
            BackendCommandResult result = await _backend.QueryAsync("--environments-json", "--force");
            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException(result.StandardError.Trim());
            }
            _environmentsJson = result.StandardOutput;
            RenderEnvironments(_environmentsJson);
            if (!string.IsNullOrWhiteSpace(_capabilitiesJson))
            {
                RenderCapabilities(_capabilitiesJson);
            }
        }
        catch (Exception exception)
        {
            EnvironmentResultsBox.Text = T("扫描失败：", "Scan failed: ") + exception.Message;
        }
        finally
        {
            ScanEnvironmentsButton.IsEnabled = true;
        }
    }

    private void ThemeBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (ThemeBox?.SelectedItem is not ComboBoxItem item)
        {
            return;
        }
        ElementTheme theme = item.Tag?.ToString() switch
        {
            "light" => ElementTheme.Light,
            "dark" => ElementTheme.Dark,
            _ => ElementTheme.Default,
        };
        if (Application.Current is App app && app.MainWindow is MainWindow window)
        {
            window.ApplyTheme(theme);
        }
        else
        {
            RequestedTheme = theme;
        }
    }

    private async void LanguageBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (LanguageBox?.SelectedItem is not ComboBoxItem item)
        {
            return;
        }
        _language = item.Tag?.ToString() switch
        {
            "zh-CN" => "zh-CN",
            "en-US" => "en-US",
            _ => CultureInfo.CurrentUICulture.TwoLetterISOLanguageName == "zh" ? "zh-CN" : "en-US",
        };
        if (!_loaded)
        {
            return;
        }
        ApplyLocalization();
        UpdateExternalEngineAvailability();
        RenderBackendPath();
        await RefreshModelsAsync();
        await RefreshCapabilitiesAsync();
    }

    private void ApplyLocalization()
    {
        Localization.Apply(this, IsChinese);

        // Language names are autonyms and must never be translated.
        ChineseLanguageItem.Content = "中文";
        EnglishLanguageItem.Content = "English";

        // Data-template content is created after the initial visual-tree pass,
        // so keep the model page explicit as well as refreshing its view models.
        ModelsTitle.Text = T("模型与下载", "Models & downloads");
        ModelsDescription.Text = T(
            "轻量版可按需下载权重；全量版使用同一界面并将内置权重显示为已安装。模型保存在当前用户目录，程序升级不会覆盖。",
            "The Lite package downloads weights on demand. The Full package uses the same UI and shows bundled weights as installed. User models survive application updates.");
        ModelSourcesTitle.Text = T("下载源", "Download source");
        ModelSourceBox.Header = T("来源", "Source");
        OfficialModelSourceItem.Content = T("官方（GitHub / Hugging Face）", "Official (GitHub / Hugging Face)");
        MirrorModelSourceItem.Content = T("镜像（GitHub Proxy / HF Mirror）", "Mirror (GitHub Proxy / HF Mirror)");
        CustomModelSourceItem.Content = T("自定义地址", "Custom URL");
        CustomSourceBox.Header = T("自定义基础地址或 {archive} 模板", "Custom base URL or {archive} template");
        RefreshModelsButton.Content = T("刷新状态", "Refresh status");
        if (_modelPacks.Count == 0)
        {
            ModelStatusText.Text = T("正在读取模型状态…", "Reading model status…");
        }
    }

    private bool IsChinese => _language == "zh-CN";

    private string T(string chinese, string english) => IsChinese ? chinese : english;

    private IReadOnlyList<string> BuildArguments()
    {
        string input = InputPathBox.Text.Trim();
        if (!File.Exists(input))
        {
            throw new ArgumentException(T("请选择有效的输入视频。", "Choose a valid input video."));
        }

        SuggestOutputPath();
        string output = OutputPathBox.Text.Trim();
        if (string.IsNullOrEmpty(output))
        {
            throw new ArgumentException(T("请选择输出文件。", "Choose an output file."));
        }

        string srEngine = SelectedTag(SrEngineBox, "auto");
        string fiEngine = SelectedTag(FiEngineBox, "auto");
        ValidateExternalEngineSelection(srEngine, fiEngine);

        List<string> values =
        [
            "--language", _language, "--progress-json", "--control-stdin", input,
            "-o", output,
            "--scale", Numeric(ScaleBox.Value),
            "--sr-engine", srEngine,
            "--fi-engine", fiEngine,
            "--sr-quality", SelectedTag(SrQualityBox, "quality"),
            "--fi-quality", SelectedTag(FiQualityBox, "quality"),
            "--fi-multiplier", Math.Round(FiMultiplierBox.Value).ToString(CultureInfo.InvariantCulture),
            "--codec", SelectedTag(CodecBox, "auto"),
            "--preset", string.IsNullOrWhiteSpace(PresetBox.Text) ? "balanced" : PresetBox.Text.Trim(),
            "--crf", Math.Round(CrfBox.Value).ToString(CultureInfo.InvariantCulture),
            "--container", SelectedTag(ContainerBox, "mp4"),
            "--ncnn-gpu", SelectedTag(NcnnGpuBox, "auto"),
        ];

        AddOptionalNumber(values, "--start", StartTimeBox.Text, T("开始时间", "Start time"));
        AddOptionalNumber(values, "--duration", DurationBox.Text, T("时长", "Duration"));
        string requestedPython = srEngine switch
        {
            "flashvsr" when !string.IsNullOrWhiteSpace(_flashVsrPython) => _flashVsrPython!,
            "seedvr2" when !string.IsNullOrWhiteSpace(_seedVr2Python) => _seedVr2Python!,
            _ => TorchPythonBox.Text.Trim(),
        };
        if (!string.IsNullOrWhiteSpace(requestedPython))
        {
            string python = requestedPython;
            if (!File.Exists(python))
            {
                throw new ArgumentException(T("指定的 PyTorch Python 不存在。", "The selected PyTorch Python does not exist."));
            }
            values.Add("--torch-python");
            values.Add(python);
        }
        if (SrFirstCheck.IsChecked == true)
        {
            values.Add("--sr-first");
        }
        if (CopyAudioCheck.IsChecked != true)
        {
            values.Add("--no-audio");
        }
        if (OverwriteCheck.IsChecked == true)
        {
            values.Add("--overwrite");
        }
        return values;
    }

    private void AddOptionalNumber(List<string> values, string option, string text, string label)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }
        if (!double.TryParse(text, NumberStyles.Float, CultureInfo.CurrentCulture, out double parsed) &&
            !double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out parsed))
        {
            throw new ArgumentException(label + T("必须是数字。", " must be numeric."));
        }
        values.Add(option);
        values.Add(parsed.ToString(CultureInfo.InvariantCulture));
    }

    private void RenderCapabilities(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        if (!root.TryGetProperty("protocol_version", out JsonElement protocol) ||
            protocol.GetInt32() != 1)
        {
            throw new InvalidOperationException(T("后端协议版本不兼容。", "The backend protocol version is incompatible."));
        }
        StringBuilder text = new();
        text.AppendLine("GPU");
        string selectedNcnnGpu = SelectedTag(NcnnGpuBox, "auto");
        NcnnGpuBox.Items.Clear();
        NcnnGpuBox.Items.Add(new ComboBoxItem { Content = T("自动", "Auto"), Tag = "auto" });
        NcnnGpuBox.Items.Add(new ComboBoxItem { Content = "CPU", Tag = "cpu" });
        if (root.TryGetProperty("gpus", out JsonElement gpus) && gpus.GetArrayLength() > 0)
        {
            int index = 0;
            foreach (JsonElement gpu in gpus.EnumerateArray())
            {
                string name = StringProperty(gpu, "name", T("未知 GPU", "Unknown GPU"));
                string vendor = StringProperty(gpu, "vendor", "unknown").ToUpperInvariant();
                text.AppendLine($"  GPU {index}: {name} [{vendor}]");
                NcnnGpuBox.Items.Add(new ComboBoxItem { Content = $"GPU {index} · {name}", Tag = index.ToString() });
                index++;
            }
        }
        else
        {
            text.AppendLine(T("  未检测到活动显示设备", "  No active display device detected"));
        }
        NcnnGpuBox.SelectedIndex = 0;
        for (int index = 0; index < NcnnGpuBox.Items.Count; index++)
        {
            if (NcnnGpuBox.Items[index] is ComboBoxItem item &&
                string.Equals(item.Tag?.ToString(), selectedNcnnGpu, StringComparison.Ordinal))
            {
                NcnnGpuBox.SelectedIndex = index;
                break;
            }
        }

        text.AppendLine();
        text.AppendLine(T("处理组件", "Processing components"));
        AppendCapability(text, root, "worker", "FFmpeg Worker");
        AppendCapability(text, root, "vsr_dll", "D3D11 VSR Bridge");
        AppendCapability(text, root, "rife_model", T("RIFE PyTorch 模型", "RIFE PyTorch model"));
        AppendCapability(text, root, "ncnn_rife", "RIFE ncnn-vulkan");
        AppendCapability(text, root, "ema_vfi_model", "EMA-VFI Small model");
        AppendCapability(text, root, "flashvsr_model", "FlashVSR v1.1 model");
        AppendCapability(text, root, "seedvr2_model", "SeedVR2 3B FP8 model");
        AppendCapability(text, root, "ncnn_ifrnet", "IFRNet ncnn-vulkan");
        AppendCapability(text, root, "ncnn_span", "SPAN ncnn-vulkan");
        AppendCapability(text, root, "ncnn_cugan", "Real-CUGAN ncnn");
        AppendCapability(text, root, "ncnn_esrgan", "Real-ESRGAN ncnn");
        AppendCapability(text, root, "ncnn_classic_esrgan", T("ESRGAN 经典模型", "Classic ESRGAN model"));

        AppendExternalEnvironmentCapabilities(text);

        text.AppendLine();
        text.AppendLine(T("可用编码器", "Available encoders"));
        if (root.TryGetProperty("encoders", out JsonElement encoders))
        {
            foreach (JsonElement encoder in encoders.EnumerateArray())
            {
                text.AppendLine("  " + encoder.GetString());
            }
        }
        CapabilitiesBox.Text = text.ToString().TrimEnd();
        UpdateBuiltInEngineAvailability(root);
        UpdateExternalEngineAvailability();
        if (root.TryGetProperty("version", out JsonElement version))
        {
            VersionText.Text = (IsChinese ? "版本 " : "Version ") + version.GetString() + " · WinUI 3";
        }
    }

    private void RenderEnvironments(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        StringBuilder text = new();
        int index = 0;
        string? recommended = null;
        _flashVsrPython = null;
        _seedVr2Python = null;
        foreach (JsonElement environment in document.RootElement.EnumerateArray())
        {
            index++;
            string exe = StringProperty(environment, "exe", "?");
            string python = StringProperty(environment, "version", "?");
            bool torch = BoolProperty(environment, "torch");
            bool cuda = BoolProperty(environment, "cuda");
            bool nvvfx = BoolProperty(environment, "nvvfx");
            bool flashvsr = BoolProperty(environment, "flashvsr");
            bool seedvr2 = BoolProperty(environment, "seedvr2");
            string torchVersion = StringProperty(environment, "torch_version", "-");
            string gpu = StringProperty(environment, "gpu_name", "-");
            text.AppendLine($"[{index}] {exe}");
            text.AppendLine($"    Python {python} | PyTorch {(torch ? torchVersion : "--")} | CUDA {(cuda ? "OK" : "--")} | NV-VFX {(nvvfx ? "OK" : "--")} | FlashVSR {(flashvsr ? "OK" : "--")} | SeedVR2 {(seedvr2 ? "OK" : "--")}");
            if (flashvsr)
            {
                _flashVsrPython ??= exe;
            }
            if (seedvr2)
            {
                _seedVr2Python ??= exe;
            }
            if (cuda)
            {
                text.AppendLine($"    GPU: {gpu}");
                recommended ??= exe;
            }
            else if (torch)
            {
                recommended ??= exe;
            }
            if (environment.TryGetProperty("error", out JsonElement error) && !string.IsNullOrWhiteSpace(error.GetString()))
            {
                text.AppendLine(T("    错误: ", "    Error: ") + error.GetString());
            }
            text.AppendLine();
        }
        EnvironmentResultsBox.Text = index == 0 ? T("未发现可用 Python 环境。", "No usable Python environment found.") : text.ToString().TrimEnd();
        if (recommended is not null && string.IsNullOrWhiteSpace(TorchPythonBox.Text))
        {
            TorchPythonBox.Text = recommended;
        }
        UpdateExternalEngineAvailability();
    }

    private void AppendExternalEnvironmentCapabilities(StringBuilder text)
    {
        text.AppendLine();
        text.AppendLine(T("外部 Python 能力", "External Python capabilities"));
        if (string.IsNullOrWhiteSpace(_environmentsJson))
        {
            text.AppendLine(T("  — 尚未扫描；点击上方按钮检测 PyTorch、CUDA 与 NV-VFX。",
                              "  — Not scanned; use the button above to detect PyTorch, CUDA, and NV-VFX."));
            return;
        }

        using JsonDocument document = JsonDocument.Parse(_environmentsJson);
        int environments = 0;
        int torch = 0;
        int cuda = 0;
        int nvvfx = 0;
        int flashvsr = 0;
        int seedvr2 = 0;
        foreach (JsonElement environment in document.RootElement.EnumerateArray())
        {
            environments++;
            torch += BoolProperty(environment, "torch") ? 1 : 0;
            cuda += BoolProperty(environment, "cuda") ? 1 : 0;
            nvvfx += BoolProperty(environment, "nvvfx") ? 1 : 0;
            flashvsr += BoolProperty(environment, "flashvsr") ? 1 : 0;
            seedvr2 += BoolProperty(environment, "seedvr2") ? 1 : 0;
        }
        text.AppendLine($"  {(torch > 0 ? "✓" : "—")} PyTorch ({torch}/{environments})");
        text.AppendLine($"  {(cuda > 0 ? "✓" : "—")} CUDA ({cuda}/{environments})");
        text.AppendLine($"  {(nvvfx > 0 ? "✓" : "—")} NVIDIA VFX ({nvvfx}/{environments})");
        text.AppendLine($"  {(flashvsr > 0 ? "✓" : "—")} FlashVSR ({flashvsr}/{environments})");
        text.AppendLine($"  {(seedvr2 > 0 ? "✓" : "—")} SeedVR2 ({seedvr2}/{environments})");
    }

    private static void AppendCapability(StringBuilder text, JsonElement root, string property, string label)
    {
        text.AppendLine($"  {(BoolProperty(root, property) ? "✓" : "—")} {label}");
    }

    private void Backend_OutputReceived(string line)
    {
        DispatcherQueue.TryEnqueue(() => AppendLog(line));
    }

    private void Backend_ProgressReceived(BackendProgress progress)
    {
        DispatcherQueue.TryEnqueue(() =>
        {
            if (_modelOperation)
            {
                ModelProgress.IsIndeterminate = progress.Total <= 0;
                if (progress.Total > 0)
                {
                    ModelProgress.Maximum = progress.Total;
                    ModelProgress.Value = Math.Clamp(progress.Current, 0, progress.Total);
                    double modelPercent = 100.0 * progress.Current / progress.Total;
                    ModelStatusText.Text = $"{progress.Stage}: {modelPercent:F1}%";
                }
                else
                {
                    ModelStatusText.Text = progress.Stage;
                }
                return;
            }
            TaskProgress.IsIndeterminate = progress.Total <= 0;
            if (progress.Total > 0)
            {
                TaskProgress.Maximum = progress.Total;
                TaskProgress.Value = Math.Clamp(progress.Current, 0, progress.Total);
                double percent = 100.0 * progress.Current / progress.Total;
                StatusText.Text = $"{progress.Stage}：{progress.Current} / {progress.Total}（{percent:F1}%）";
            }
            else
            {
                StatusText.Text = progress.Stage;
            }
        });
    }

    private void AppendLog(string line)
    {
        _logText.AppendLine(line);
        if (_logText.Length > 160_000)
        {
            int boundary = _logText.ToString().IndexOf('\n', 20_000);
            _logText.Remove(0, boundary > 0 ? boundary + 1 : 20_000);
        }
        LogBox.Text = _logText.ToString();
        LogBox.SelectionStart = LogBox.Text.Length;
    }

    private void SetProcessingState(bool running)
    {
        StartButton.IsEnabled = !running;
        CancelButton.IsEnabled = running;
        ScanEnvironmentsButton.IsEnabled = !running;
    }

    private void SuggestOutputPath()
    {
        if (InputPathBox is null || OutputPathBox is null || ContainerBox is null)
        {
            return;
        }
        string input = InputPathBox.Text.Trim();
        if (string.IsNullOrEmpty(input))
        {
            return;
        }
        if (!string.IsNullOrWhiteSpace(OutputPathBox.Text) &&
            !string.Equals(OutputPathBox.Text, _lastSuggestedOutput, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }
        string directory = Path.GetDirectoryName(input) ?? Environment.CurrentDirectory;
        string suggested = Path.Combine(directory, SuggestedStem() + "." + SelectedTag(ContainerBox, "mp4"));
        _lastSuggestedOutput = suggested;
        OutputPathBox.Text = suggested;
    }

    private string SuggestedStem()
    {
        string? stem = Path.GetFileNameWithoutExtension(InputPathBox?.Text.Trim());
        if (string.IsNullOrWhiteSpace(stem))
        {
            stem = "enhanced";
        }
        List<string> tags = [];
        if (SrEngineBox is not null && SelectedTag(SrEngineBox, "auto") != "none" && ScaleBox is not null && ScaleBox.Value != 1)
        {
            tags.Add("x" + Numeric(ScaleBox.Value));
        }
        if (FiEngineBox is not null && SelectedTag(FiEngineBox, "auto") != "none" && FiMultiplierBox is not null)
        {
            tags.Add("f" + Math.Round(FiMultiplierBox.Value).ToString(CultureInfo.InvariantCulture));
        }
        return stem + (tags.Count > 0 ? "_" + string.Join("_", tags) : "_enhanced");
    }

    private FileOpenPicker CreateOpenPicker(string title, params string[] extensions)
    {
        FileOpenPicker picker = new()
        {
            ViewMode = PickerViewMode.Thumbnail,
            SuggestedStartLocation = PickerLocationId.VideosLibrary,
            CommitButtonText = title,
        };
        foreach (string extension in extensions)
        {
            picker.FileTypeFilter.Add(extension);
        }
        InitializeWithWindow.Initialize(picker, MainWindowHandle());
        return picker;
    }

    private nint MainWindowHandle()
    {
        if (Application.Current is not App app || app.MainWindow is null)
        {
            throw new InvalidOperationException(T("主窗口尚未初始化。", "The main window is not initialized."));
        }
        return WindowNative.GetWindowHandle(app.MainWindow);
    }

    private void ShowInfo(string title, string message, InfoBarSeverity severity)
    {
        title = title.Trim();
        message = message.Trim();
        if (title.Length == 0 && message.Length == 0)
        {
            HideInfo();
            return;
        }
        BackendInfoBar.Title = title;
        BackendInfoBar.Message = message;
        BackendInfoBar.Severity = severity;
        BackendInfoBar.Visibility = Visibility.Visible;
        BackendInfoBar.IsOpen = true;
    }

    private void HideInfo()
    {
        BackendInfoBar.IsOpen = false;
        BackendInfoBar.Visibility = Visibility.Collapsed;
        BackendInfoBar.Title = string.Empty;
        BackendInfoBar.Message = string.Empty;
    }

    private void BackendInfoBar_Closed(InfoBar sender, InfoBarClosedEventArgs args)
    {
        BackendInfoBar.Visibility = Visibility.Collapsed;
    }

    private void RenderBackendPath()
    {
        BackendPathText.Text = $"{_backend.Location.DisplayName}\n{_backend.Location.FileName}\n" +
            T("工作目录：", "Working directory: ") + _backend.Location.WorkingDirectory;
    }

    private static string SelectedTag(ComboBox box, string fallback)
    {
        return (box.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? fallback;
    }

    private string Numeric(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
        {
            throw new ArgumentException(T("倍率或质量数值无效。", "Scale or quality value is invalid."));
        }
        return value.ToString("0.##", CultureInfo.InvariantCulture);
    }

    private static bool BoolProperty(JsonElement element, string name)
    {
        return element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.True;
    }

    private static long Int64Property(JsonElement element, string name)
    {
        return element.TryGetProperty(name, out JsonElement value) && value.TryGetInt64(out long result)
            ? result
            : 0;
    }

    private static string StringProperty(JsonElement element, string name, string fallback)
    {
        return element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? fallback
            : fallback;
    }
}

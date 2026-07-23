using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace LightVideoEnhancer_WinUI.Services;

public sealed record BackendProgress(string Stage, int Current, int Total);

public sealed record BackendCommandResult(int ExitCode, string StandardOutput, string StandardError);

public sealed record BackendLocation(
    string FileName,
    string WorkingDirectory,
    IReadOnlyList<string> PrefixArguments,
    string DisplayName);

/// <summary>
/// Runs the existing Python processing core out of process.  The protocol is
/// deliberately line based so the WinUI frontend never imports Python or CUDA
/// into its own process.
/// </summary>
public sealed class BackendProcess : IDisposable
{
    private const string ProgressPrefix = "__LVE_PROGRESS__";
    private readonly object _sync = new();
    private Process? _process;
    private int _busy;

    public BackendProcess()
    {
        Location = ResolveBackend();
    }

    public BackendLocation Location { get; }

    public bool IsRunning => Volatile.Read(ref _busy) != 0;

    public event Action<string>? OutputReceived;

    public event Action<BackendProgress>? ProgressReceived;

    public async Task<int> RunAsync(IEnumerable<string> arguments)
    {
        EnterBusy();
        try
        {
            using Process process = CreateProcess(arguments);
            SetCurrent(process);
            if (!process.Start())
            {
                throw new InvalidOperationException("Unable to start the video processing backend.");
            }

            Task stdout = PumpAsync(process.StandardOutput, false);
            Task stderr = PumpAsync(process.StandardError, true);
            await process.WaitForExitAsync();
            await Task.WhenAll(stdout, stderr);
            return process.ExitCode;
        }
        finally
        {
            SetCurrent(null);
            Volatile.Write(ref _busy, 0);
        }
    }

    public async Task<BackendCommandResult> QueryAsync(params string[] arguments)
    {
        EnterBusy();
        try
        {
            using Process process = CreateProcess(arguments);
            SetCurrent(process);
            if (!process.Start())
            {
                throw new InvalidOperationException("Unable to start the capability detection backend.");
            }

            Task<string> stdout = process.StandardOutput.ReadToEndAsync();
            Task<string> stderr = process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();
            return new BackendCommandResult(
                process.ExitCode, await stdout, await stderr);
        }
        finally
        {
            SetCurrent(null);
            Volatile.Write(ref _busy, 0);
        }
    }

    public async Task CancelAsync(TimeSpan? gracefulTimeout = null)
    {
        Process? process;
        lock (_sync)
        {
            process = _process;
        }

        if (process is null || process.HasExited)
        {
            return;
        }

        try
        {
            await process.StandardInput.WriteLineAsync("cancel");
            await process.StandardInput.FlushAsync();
        }
        catch (InvalidOperationException)
        {
            return;
        }
        catch (IOException)
        {
            return;
        }

        TimeSpan timeout = gracefulTimeout ?? TimeSpan.FromSeconds(12);
        using CancellationTokenSource timeoutSource = new(timeout);
        try
        {
            await process.WaitForExitAsync(timeoutSource.Token);
        }
        catch (OperationCanceledException)
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }
        }
    }

    private void EnterBusy()
    {
        if (Interlocked.CompareExchange(ref _busy, 1, 0) != 0)
        {
            throw new InvalidOperationException("The backend is already running another task.");
        }
    }

    private void SetCurrent(Process? process)
    {
        lock (_sync)
        {
            _process = process;
        }
    }

    private Process CreateProcess(IEnumerable<string> arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = Location.FileName,
            WorkingDirectory = Location.WorkingDirectory,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        startInfo.Environment["PYTHONUTF8"] = "1";
        startInfo.Environment["PYTHONIOENCODING"] = "utf-8";
        startInfo.Environment["PYTHONUNBUFFERED"] = "1";
        foreach (string argument in Location.PrefixArguments.Concat(arguments))
        {
            startInfo.ArgumentList.Add(argument);
        }
        return new Process { StartInfo = startInfo, EnableRaisingEvents = true };
    }

    private async Task PumpAsync(StreamReader reader, bool errorStream)
    {
        while (await reader.ReadLineAsync() is { } line)
        {
            if (TryParseProgress(line, out BackendProgress? progress) && progress is not null)
            {
                try
                {
                    ProgressReceived?.Invoke(progress);
                }
                catch
                {
                    // UI event handlers must not interrupt video processing.
                }
                continue;
            }

            try
            {
                OutputReceived?.Invoke(errorStream ? "[STDERR] " + line : line);
            }
            catch
            {
                // UI event handlers must not interrupt video processing.
            }
        }
    }

    private static bool TryParseProgress(string line, out BackendProgress? progress)
    {
        progress = null;
        if (!line.StartsWith(ProgressPrefix, StringComparison.Ordinal))
        {
            return false;
        }

        try
        {
            using JsonDocument document = JsonDocument.Parse(line[ProgressPrefix.Length..]);
            JsonElement root = document.RootElement;
            progress = new BackendProgress(
                root.GetProperty("stage").GetString() ?? "Processing",
                root.GetProperty("current").GetInt32(),
                root.GetProperty("total").GetInt32());
        }
        catch (JsonException)
        {
            progress = new BackendProgress("Processing", 0, 0);
        }
        return true;
    }

    private static BackendLocation ResolveBackend()
    {
        string? overridden = Environment.GetEnvironmentVariable("LVE_BACKEND");
        if (!string.IsNullOrWhiteSpace(overridden) && File.Exists(overridden))
        {
            return new BackendLocation(
                Path.GetFullPath(overridden),
                Path.GetDirectoryName(Path.GetFullPath(overridden))!,
                Array.Empty<string>(),
                "External backend · " + Path.GetFileName(overridden));
        }

        List<string> baseDirectories = [AppContext.BaseDirectory];
        string? processDirectory = Path.GetDirectoryName(Environment.ProcessPath);
        if (!string.IsNullOrWhiteSpace(processDirectory))
        {
            baseDirectories.Insert(0, processDirectory);
        }
        string[] executableNames =
        [
            "LightVideoEnhancer-Backend.exe",
            "LightVideoEnhancer-Backend-Win10-11-x64.exe",
        ];
        foreach (string baseDirectory in baseDirectories.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            foreach (string name in executableNames)
            {
                string candidate = Path.Combine(baseDirectory, name);
                if (File.Exists(candidate))
                {
                    return new BackendLocation(
                        candidate, baseDirectory, Array.Empty<string>(),
                        "Portable backend · " + name);
                }
            }
        }

        string? projectRoot = baseDirectories.Select(FindProjectRoot).FirstOrDefault(root => root is not null);
        string python = Environment.GetEnvironmentVariable("LVE_PYTHON") ?? "python.exe";
        if (projectRoot is not null)
        {
            string virtualPython = Path.Combine(projectRoot, ".venv", "Scripts", "python.exe");
            if (File.Exists(virtualPython))
            {
                python = virtualPython;
            }
        }
        return new BackendLocation(
            python,
            projectRoot ?? Environment.CurrentDirectory,
            ["-m", "light_video_enhancer"],
            "Development backend · " + python);
    }

    private static string? FindProjectRoot(string start)
    {
        DirectoryInfo? directory = new(start);
        for (int depth = 0; directory is not null && depth < 14; depth++, directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "pyproject.toml")) &&
                Directory.Exists(Path.Combine(directory.FullName, "light_video_enhancer")))
            {
                return directory.FullName;
            }
        }
        return null;
    }

    public void Dispose()
    {
        Process? process;
        lock (_sync)
        {
            process = _process;
            _process = null;
        }
        if (process is not null)
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch (InvalidOperationException)
            {
            }
            finally
            {
                process.Dispose();
            }
        }
    }
}

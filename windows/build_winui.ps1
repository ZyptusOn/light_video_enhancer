[CmdletBinding()]
param(
    [string]$Python = "python",
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [ValidateSet("Both", "Full", "Lite")]
    [string]$Profile = "Both",
    [switch]$SkipBackend
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$project = Join-Path $PSScriptRoot "LightVideoEnhancer.WinUI\LightVideoEnhancer.WinUI.csproj"
$dist = Join-Path $root "dist"
$profiles = if ($Profile -eq "Both") { @("Full", "Lite") } else { @($Profile) }

if (-not $SkipBackend) {
    foreach ($item in $profiles) {
        $backendProfile = if ($item -eq "Lite") { "light" } else { "full" }
        & $Python (Join-Path $root "build_exe.py") --backend --profile $backendProfile
        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller $item backend build failed with exit code $LASTEXITCODE."
        }
    }
}

foreach ($item in $profiles) {
    $backend = Join-Path $dist "LightVideoEnhancer-Backend-$item.exe"
    if (-not (Test-Path -LiteralPath $backend)) {
        throw "Backend executable is missing: $backend"
    }
}

dotnet build $project `
    --configuration $Configuration `
    --runtime win-x64 `
    -p:Platform=x64
if ($LASTEXITCODE -ne 0) {
    throw "WinUI build failed with exit code $LASTEXITCODE."
}

$buildOutput = Join-Path $PSScriptRoot "LightVideoEnhancer.WinUI\bin\x64\$Configuration\net10.0-windows10.0.26100.0\win-x64"
if (-not (Test-Path -LiteralPath (Join-Path $buildOutput "LightVideoEnhancer.WinUI.pri"))) {
    throw "Compiled XAML resources are missing: $buildOutput"
}

foreach ($item in $profiles) {
    $output = Join-Path $dist "LightVideoEnhancer-WinUI3-$item-Win10-11-x64"
    $zip = "$output.zip"
    $backend = Join-Path $dist "LightVideoEnhancer-Backend-$item.exe"
    if (Test-Path -LiteralPath $output) {
        Remove-Item -LiteralPath $output -Recurse -Force
    }
    if (Test-Path -LiteralPath $zip) {
        Remove-Item -LiteralPath $zip -Force
    }
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    Copy-Item -Path (Join-Path $buildOutput "*") -Destination $output -Recurse -Force
    Copy-Item -LiteralPath $backend -Destination (Join-Path $output "LightVideoEnhancer-Backend.exe") -Force

    $frontend = Join-Path $output "LightVideoEnhancer.WinUI.exe"
    if (-not (Test-Path -LiteralPath $frontend)) {
        throw "Frontend executable is missing after staging: $frontend"
    }
    Compress-Archive -Path (Join-Path $output "*") -DestinationPath $zip -CompressionLevel Optimal
    $bytes = (Get-ChildItem -LiteralPath $output -File -Recurse | Measure-Object Length -Sum).Sum
    Write-Host ("WinUI 3 {0} package ready: {1} ({2:N1} MiB)" -f $item, $zip, ($bytes / 1MB))
    Get-FileHash -Algorithm SHA256 -LiteralPath $zip, $frontend, (Join-Path $output "LightVideoEnhancer-Backend.exe")
}

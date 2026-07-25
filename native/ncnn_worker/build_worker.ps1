param(
    [string]$NcnnSource = "",
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $NcnnSource) {
    $NcnnSource = Join-Path $ProjectRoot "build\native_deps\ncnn"
}
$NcnnSource = [System.IO.Path]::GetFullPath($NcnnSource)
$BuildDir = Join-Path $ProjectRoot "build\ncnn_worker"
$OutputDir = Join-Path $ProjectRoot "light_video_enhancer\ncnn\lve_worker"

$VsRoot = "C:\Program Files\Microsoft Visual Studio\18\Community"
$Cmake = Join-Path $VsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$Ninja = Join-Path $VsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$VsDevCmd = Join-Path $VsRoot "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $Cmake) -or -not (Test-Path $Ninja) -or -not (Test-Path $VsDevCmd)) {
    throw "Visual Studio C++/CMake tools were not found."
}
if (-not (Test-Path (Join-Path $NcnnSource "CMakeLists.txt"))) {
    throw "NCNN source was not found: $NcnnSource"
}

$Configure = 'call "{0}" -arch=x64 -host_arch=x64 && "{1}" -S "{2}" -B "{3}" -G Ninja "-DCMAKE_MAKE_PROGRAM={4}" "-DCMAKE_BUILD_TYPE={5}" "-DNCNN_SOURCE_DIR={6}"' -f `
    $VsDevCmd, $Cmake, $PSScriptRoot, $BuildDir, $Ninja, $Configuration, $NcnnSource
& cmd.exe /d /s /c $Configure
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }

$Build = 'call "{0}" -arch=x64 -host_arch=x64 && "{1}" --build "{2}" --config "{3}" --target lve-ncnn-worker' -f `
    $VsDevCmd, $Cmake, $BuildDir, $Configuration
& cmd.exe /d /s /c $Build
if ($LASTEXITCODE -ne 0) { throw "NCNN worker build failed." }

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Copy-Item (Join-Path $BuildDir "lve-ncnn-worker.exe") `
    (Join-Path $OutputDir "lve-ncnn-worker.exe") -Force
# NCNN uses OpenMP for CPU-side packing and preprocessing.  Keep the matching
# redistributable beside the worker so clean Windows 7/10/11 systems do not
# depend on a Visual C++ or Visual Studio installation.
$OpenMpRuntime = Join-Path $ProjectRoot `
    "light_video_enhancer\ncnn\rife\vcomp140.dll"
if (-not (Test-Path -LiteralPath $OpenMpRuntime)) {
    throw "OpenMP runtime was not found: $OpenMpRuntime"
}
Copy-Item -LiteralPath $OpenMpRuntime -Destination $OutputDir -Force
Write-Host "Built: $(Join-Path $OutputDir 'lve-ncnn-worker.exe')"

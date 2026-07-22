#!/usr/bin/env python3
"""Build a portable Light Video Enhancer GUI executable.

Use CPython 3.8.10 + PyInstaller 5.13.2 for the Windows 7 package. Use a
64-bit CPython 3.10+ environment + PyInstaller 6 for the Windows 10/11 package.
"""

import os
import platform
import re
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "light_video_enhancer"
PACKAGE_DIR = os.path.join(PROJECT_DIR, PACKAGE_NAME)
LAUNCHER = os.path.join(PROJECT_DIR, "launcher.py")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "dist")
MODERN_MANIFEST = os.path.join(PROJECT_DIR, "windows_manifest_win10.xml")


def _project_version():
    path = os.path.join(PACKAGE_DIR, "__init__.py")
    with open(path, "r", encoding="utf-8") as handle:
        match = re.search(r'^__version__\s*=\s*["\']([^"\']+)', handle.read(), re.MULTILINE)
    if not match:
        raise SystemExit("无法读取项目版本。")
    return match.group(1)


def _data_files():
    relative = [
        "_shared_frames.py",
        "_fused_rife_nvvfx_infer.py",
        "ffmpeg_dlls",
        os.path.join("ffmpeg_bridge", "ffmpeg_worker.dll"),
        os.path.join("bridge", "dxva_vsr_bridge.dll"),
        os.path.join("fi", "_rife_infer.py"),
        os.path.join("fi", "_rife_model.py"),
        os.path.join("fi", "warplayer.py"),
        os.path.join("fi", "flownet.pkl"),
        os.path.join("sr", "_nvvfx_infer.py"),
        "ncnn",
    ]
    result = []
    for item in relative:
        source = os.path.join(PACKAGE_DIR, item)
        if not os.path.exists(source):
            continue
        if os.path.isdir(source):
            for root, _, files in os.walk(source):
                for name in files:
                    path = os.path.join(root, name)
                    target = os.path.join(PACKAGE_NAME, os.path.relpath(root, PACKAGE_DIR))
                    result.extend(["--add-data", path + os.pathsep + target])
        else:
            target = os.path.join(PACKAGE_NAME, os.path.dirname(item))
            result.extend(["--add-data", source + os.pathsep + target])
    return result


def _version_file(version):
    numbers = [int(part) for part in re.findall(r"\d+", version)[:4]]
    numbers.extend([0] * (4 - len(numbers)))
    dotted = ".".join(str(value) for value in numbers)
    comma = ", ".join(str(value) for value in numbers)
    path = os.path.join(PROJECT_DIR, "build", "version_info.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = """VSVersionInfo(
  ffi=FixedFileInfo(filevers=({comma}), prodvers=({comma}), mask=0x3f,
    flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('080404b0', [
    StringStruct('CompanyName', 'ZyptusOn'),
    StringStruct('FileDescription', 'Light Video Enhancer'),
    StringStruct('FileVersion', '{dotted}'),
    StringStruct('InternalName', 'LightVideoEnhancer'),
    StringStruct('OriginalFilename', 'LightVideoEnhancer.exe'),
    StringStruct('ProductName', 'Light Video Enhancer'),
    StringStruct('ProductVersion', '{dotted}')])]),
    VarFileInfo([VarStruct('Translation', [2052, 1200])])])
""".format(comma=comma, dotted=dotted)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def _build_target(pyinstaller_version):
    if platform.architecture()[0] != "64bit":
        raise SystemExit("发布包必须使用 64 位 Python 构建。")
    if sys.version_info[:2] == (3, 8):
        if pyinstaller_version != "5.13.2":
            raise SystemExit("Windows 7 构建必须使用 PyInstaller 5.13.2。")
        return "win7", "LightVideoEnhancer-Win7-x64"
    if sys.version_info >= (3, 10):
        major = int(pyinstaller_version.split(".", 1)[0])
        if major < 6:
            raise SystemExit("Windows 10/11 构建必须使用 PyInstaller 6 或更高版本。")
        return "modern", "LightVideoEnhancer-Win10-11-x64"
    raise SystemExit("请使用 Python 3.8.10 构建 Win7 版，或用 64 位 Python 3.10+ 构建 Win10/11 版。")


def main() -> None:
    try:
        import PyInstaller
    except ImportError:
        raise SystemExit("缺少 PyInstaller。请先安装对应的构建依赖。")
    if not os.path.isdir(PACKAGE_DIR):
        raise SystemExit("找不到包目录: %s" % PACKAGE_DIR)

    target, executable_name = _build_target(PyInstaller.__version__)
    version = _project_version()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hidden = [
        PACKAGE_NAME + ".fused_rife_nvvfx",
        PACKAGE_NAME, PACKAGE_NAME + ".pipeline", PACKAGE_NAME + ".config",
        PACKAGE_NAME + ".cli", PACKAGE_NAME + ".gui", PACKAGE_NAME + ".capabilities",
        PACKAGE_NAME + ".sr", PACKAGE_NAME + ".sr.dxva_vsr",
        PACKAGE_NAME + ".sr.nvvfx_sr", PACKAGE_NAME + ".sr.realcugan_ncnn",
        PACKAGE_NAME + ".sr.realesrgan_ncnn", PACKAGE_NAME + ".fi",
        PACKAGE_NAME + ".fi.rife", PACKAGE_NAME + ".fi.rife_ncnn",
        PACKAGE_NAME + ".fi.optical_flow", PACKAGE_NAME + ".fi.dis_flow",
        PACKAGE_NAME + ".fi.blend", PACKAGE_NAME + ".ffmpeg_bridge",
    ]
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile",
        "--windowed", "--name", executable_name, "--distpath", OUTPUT_DIR,
        "--workpath", os.path.join(PROJECT_DIR, "build", "pyinstaller"),
        "--specpath", os.path.join(PROJECT_DIR, "build"), "--noupx",
        "--version-file", _version_file(version),
    ]
    if target == "modern":
        command.extend(["--manifest", MODERN_MANIFEST])
    for module in hidden:
        command.extend(["--hidden-import", module])
    for module in ("torch", "torchvision", "torchaudio", "nvvfx", "tensorflow",
                   "pandas", "matplotlib", "jupyter", "scipy"):
        command.extend(["--exclude-module", module])
    command.extend(_data_files())
    command.append(LAUNCHER)
    build_env = dict(os.environ)
    build_env["PYTHONNOUSERSITE"] = "1"
    print("构建目标: %s | Python %s | PyInstaller %s | 项目 %s" % (
        "Windows 7" if target == "win7" else "Windows 10/11",
        platform.python_version(), PyInstaller.__version__, version))
    subprocess.run(command, check=True, cwd=PROJECT_DIR, env=build_env)
    executable = os.path.join(OUTPUT_DIR, executable_name + ".exe")
    if not os.path.isfile(executable):
        raise SystemExit("构建结束但未生成可执行文件。")
    print("完成: %s (%.1f MB)" % (executable, os.path.getsize(executable) / 1024 / 1024))


if __name__ == "__main__":
    main()

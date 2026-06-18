"""
自动检测系统上的 Python 环境及 torch/CUDA 可用性。

用于 PyInstaller 打包后（不包含 torch）查找系统 Python 的 torch 环境，
以支持 RIFE 等需要 PyTorch 的插帧引擎。
"""

import os
import sys
import subprocess
import json
from typing import Optional, Dict, Any, List


from ._logging import get_logger

_log = get_logger(__name__)


def _find_python_candidates() -> List[str]:
    """收集系统上可能的 Python 可执行文件路径。"""
    candidates = []

    # 1. 当前运行的 Python（非 frozen）
    if not getattr(sys, "frozen", False):
        candidates.append(sys.executable)

    if sys.platform != "win32":
        return candidates

    # 2. py.exe 启动器
    try:
        r = subprocess.run(
            ["py", "--list-paths"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = line.split()
                for p in parts:
                    if os.path.basename(p).lower().startswith("python") and os.path.isfile(p):
                        if p not in candidates:
                            candidates.append(p)
                        break
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # 3. PATH
    for name in ["python.exe", "python3.exe"]:
        for d in os.environ.get("PATH", "").split(os.pathsep):
            p = os.path.join(d, name)
            if os.path.isfile(p) and p not in candidates:
                candidates.append(p)

    # 4. 常见安装路径（仅当环境变量非空时搜索）
    local = os.environ.get("LOCALAPPDATA", "")
    prog = os.environ.get("PROGRAMDATA", "")
    home = os.path.expanduser("~")

    for prefix in [local, prog, home]:
        if not prefix or not os.path.isabs(prefix):
            continue
        for suffix in [
            "miniconda3", "Miniconda3", "anaconda3", "Anaconda3",
            "miniforge3", "Miniforge3",
        ]:
            base = os.path.join(prefix, suffix)
            exe = os.path.join(base, "python.exe")
            if os.path.isfile(exe) and exe not in candidates:
                candidates.append(exe)

        py_base = os.path.join(prefix, "Programs", "Python")
        if os.path.isdir(py_base):
            for d in sorted(os.listdir(py_base), reverse=True):
                if d.lower().startswith("python3"):
                    exe = os.path.join(py_base, d, "python.exe")
                    if os.path.isfile(exe) and exe not in candidates:
                        candidates.append(exe)

    return candidates


def check_python_env(python_exe: str, timeout: float = 10.0) -> Dict[str, Any]:
    """
    检测指定 Python 环境的信息。

    返回:
        {"exe": str, "version": str, "torch": bool, "cuda": bool,
         "gpu_name": str, "torch_version": str, "nvvfx": bool}
    """
    script = """\
import sys, json
r = {}
r['version'] = '.'.join(str(x) for x in sys.version_info[:3])
try:
    import torch
    r['torch'] = True
    r['torch_version'] = torch.__version__
    r['cuda'] = torch.cuda.is_available()
    r['gpu_name'] = torch.cuda.get_device_name(0) if r['cuda'] else ''
except ImportError:
    r['torch'] = False
    r['cuda'] = False
    r['torch_version'] = ''
    r['gpu_name'] = ''
try:
    import nvvfx
    r['nvvfx'] = True
except ImportError:
    r['nvvfx'] = False
print(json.dumps(r))
"""
    try:
        r = subprocess.run(
            [python_exe, "-c", script],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        if r.returncode == 0 and r.stdout.strip():
            lines = r.stdout.strip().splitlines()
            info = None
            for line in reversed(lines):
                try:
                    info = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
            if info is None:
                info = json.loads(lines[-1])
            info["exe"] = python_exe
            return info
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        pass

    return {
        "exe": python_exe, "version": "?",
        "torch": False, "cuda": False,
        "torch_version": "", "gpu_name": "",
        "nvvfx": False,
    }


def find_torch_python(timeout: float = 10.0) -> Optional[str]:
    """
    查找系统上可用的带 torch+CUDA 的 Python 路径。
    优先返回 CUDA 可用的环境，其次返回有 torch 的环境。
    """
    candidates = _find_python_candidates()
    if not candidates:
        return None

    torch_with_cuda = None
    torch_no_cuda = None

    for exe in candidates:
        if not os.path.isfile(exe):
            continue
        info = check_python_env(exe, timeout=timeout)
        if info.get("torch"):
            if info.get("cuda") and torch_with_cuda is None:
                torch_with_cuda = exe
                _log.info("[环境] 找到 torch+CUDA: %s (torch %s, %s)",
                         exe, info.get('torch_version', '?'),
                         info.get('gpu_name', '?'))
            elif torch_no_cuda is None:
                torch_no_cuda = exe

    result = torch_with_cuda or torch_no_cuda
    if result and not torch_with_cuda:
        _log.info("[环境] 找到 torch (无CUDA): %s", result)
    if not result:
        _log.info("[环境] 未找到可用的 torch 环境")
    return result


# 模块级缓存
_torch_python_cache = None
_checked = False


def get_torch_python(force_rescan: bool = False) -> Optional[str]:
    """获取带 torch+CUDA 的 Python 路径（带缓存）。"""
    global _torch_python_cache, _checked
    if _checked and not force_rescan:
        return _torch_python_cache
    _checked = True

    # 1. 尝试当前进程 import torch
    try:
        import torch
        if torch.cuda.is_available():
            _torch_python_cache = None  # None = 使用当前进程
            return None
    except ImportError:
        pass

    # 2. 扫描系统 Python
    _torch_python_cache = find_torch_python(timeout=15.0)
    return _torch_python_cache


def get_all_python_envs(timeout: float = 5.0) -> List[Dict[str, Any]]:
    """返回所有检测到的 Python 环境信息列表（供 GUI 显示）。"""
    candidates = _find_python_candidates()
    results = []
    seen = set()
    for exe in candidates:
        if exe in seen or not os.path.isfile(exe):
            continue
        seen.add(exe)
        results.append(check_python_env(exe, timeout=timeout))
    return results

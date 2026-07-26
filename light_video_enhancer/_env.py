"""On-demand, cached Python/PyTorch runtime discovery.

No function in this module scans environments at import time. GUI startup uses
only quick file/module checks; Python discovery and CUDA imports happen after
the user presses the explicit environment-scan button or selects an engine
that needs an external runtime.
"""

import concurrent.futures
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

from ._logging import get_logger

_log = get_logger(__name__)
_memory_result = None
_memory_checked = False


def _cache_path() -> str:
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(root, "LightVideoEnhancer", "environment-cache.json")


def _dedupe(paths: Iterable[str]) -> List[str]:
    """Return existing executables once and ignore Windows Store aliases."""
    result: List[str] = []
    seen = set()
    store_fragment = os.path.normcase(os.path.join("Microsoft", "WindowsApps"))
    for path in paths:
        if not path:
            continue
        path = os.path.abspath(os.path.expandvars(path.strip().strip('"')))
        normalised = os.path.normcase(path)
        # Starting an App Execution Alias can open Microsoft Store. It is not a
        # usable interpreter until its real installation is discoverable.
        if os.name == "nt" and store_fragment in normalised:
            continue
        if not os.path.isfile(path):
            continue
        key = os.path.normcase(os.path.realpath(path))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _parse_py_launcher_paths(output: str) -> List[str]:
    """Parse ``py -0p`` without breaking installation paths containing spaces."""
    result = []
    pattern = re.compile(
        r"((?:[A-Za-z]:[\\/]|\\\\).+?python(?:\d+(?:\.\d+)*)?(?:\.exe)?)\s*$",
        re.IGNORECASE,
    )
    for line in output.splitlines():
        match = pattern.search(line.strip())
        if match:
            result.append(match.group(1))
    return result


def _registry_python_candidates() -> List[str]:
    if os.name != "nt":
        return []
    try:
        import winreg
    except ImportError:
        return []
    candidates: List[str] = []
    views = [0]
    for name in ("KEY_WOW64_64KEY", "KEY_WOW64_32KEY"):
        value = getattr(winreg, name, 0)
        if value not in views:
            views.append(value)
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in views:
            try:
                root = winreg.OpenKey(
                    hive, r"SOFTWARE\Python\PythonCore", 0, winreg.KEY_READ | view)
            except OSError:
                continue
            try:
                index = 0
                while True:
                    try:
                        version = winreg.EnumKey(root, index)
                    except OSError:
                        break
                    index += 1
                    try:
                        install = winreg.OpenKey(root, version + r"\InstallPath")
                    except OSError:
                        continue
                    try:
                        try:
                            candidates.append(winreg.QueryValueEx(install, "ExecutablePath")[0])
                        except OSError:
                            try:
                                candidates.append(os.path.join(winreg.QueryValue(install, None),
                                                               "python.exe"))
                            except OSError:
                                pass
                    finally:
                        winreg.CloseKey(install)
            finally:
                winreg.CloseKey(root)
    return candidates


def _glob_python_candidates() -> List[str]:
    """Find common CPython, Conda, uv, pyenv and virtualenv layouts."""
    env = os.environ
    home = env.get("USERPROFILE") or os.path.expanduser("~")
    local = env.get("LOCALAPPDATA", "")
    roaming = env.get("APPDATA", "")
    program_data = env.get("PROGRAMDATA", r"C:\ProgramData" if os.name == "nt" else "")
    program_files = [env.get("ProgramFiles", ""), env.get("ProgramFiles(x86)", "")]
    patterns = [
        os.path.join(local, "Programs", "Python", "Python*", "python.exe"),
        os.path.join(local, "Python", "bin", "python.exe"),
        os.path.join(local, "uv", "python", "*", "python.exe"),
        os.path.join(roaming, "uv", "python", "*", "python.exe"),
        os.path.join(home, ".pyenv", "pyenv-win", "versions", "*", "python.exe"),
        os.path.join(home, ".virtualenvs", "*", "Scripts", "python.exe"),
        os.path.join(local, "pypoetry", "Cache", "virtualenvs", "*", "Scripts", "python.exe"),
        os.path.join(home, ".conda", "envs", "*", "python.exe"),
    ]
    conda_roots = [
        env.get("CONDA_PREFIX", ""),
        os.path.join(home, "miniconda3"), os.path.join(home, "anaconda3"),
        os.path.join(local, "miniconda3"), os.path.join(local, "anaconda3"),
        os.path.join(program_data, "miniconda3"), os.path.join(program_data, "anaconda3"),
    ]
    for root in conda_roots:
        if root:
            patterns.extend([
                os.path.join(root, "python.exe"),
                os.path.join(root, "envs", "*", "python.exe"),
            ])
    for root in program_files:
        if root:
            patterns.append(os.path.join(root, "Python*", "python.exe"))
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    return candidates


def _conda_list_candidates() -> List[str]:
    """Ask Conda for named environments; failure is intentionally harmless."""
    if os.name != "nt":
        return []
    conda = shutil.which("conda.exe") or shutil.which("conda")
    if not conda:
        return []
    try:
        result = subprocess.run(
            [conda, "env", "list", "--json"], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True, timeout=4,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        data = json.loads(result.stdout)
        return [os.path.join(prefix, "python.exe") for prefix in data.get("envs", [])]
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, json.JSONDecodeError):
        return []


def _find_python_candidates() -> List[str]:
    candidates: List[str] = []
    explicit = os.environ.get("LVE_TORCH_PYTHON")
    if explicit:
        candidates.append(explicit)
    if not getattr(sys, "frozen", False):
        candidates.append(sys.executable)
    for variable in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        prefix = os.environ.get(variable)
        if prefix:
            candidates.append(os.path.join(prefix, "python.exe" if os.name == "nt" else "bin/python"))
    if os.name == "nt":
        # PATH entries are inspected directly so we can find more than the
        # first interpreter returned by shutil.which().
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            if directory:
                candidates.append(os.path.join(directory.strip('"'), "python.exe"))
        for command in ("python.exe", "python3.exe"):
            found = shutil.which(command)
            if found:
                candidates.append(found)
        try:
            result = subprocess.run(
                ["py", "-0p"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=2,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            candidates.extend(_parse_py_launcher_paths(result.stdout))
        except (OSError, subprocess.TimeoutExpired):
            pass
        candidates.extend(_registry_python_candidates())
        candidates.extend(_glob_python_candidates())
        candidates.extend(_conda_list_candidates())
    return _dedupe(candidates)


def check_python_env(python_exe: str, timeout: float = 12.0) -> Dict[str, Any]:
    script = r"""
import importlib.util, json, sys
r = {'version': '.'.join(map(str, sys.version_info[:3])), 'torch': False,
     'cuda': False, 'torch_version': '', 'gpu_name': '', 'nvvfx': False,
     'flashvsr': False, 'seedvr2': False}
find = importlib.util.find_spec
if importlib.util.find_spec('torch') is not None:
    try:
        import torch
        r['torch'] = True
        r['torch_version'] = str(torch.__version__)
        r['cuda'] = bool(torch.cuda.is_available())
        if r['cuda']:
            r['gpu_name'] = torch.cuda.get_device_name(0)
    except Exception as e:
        r['error'] = str(e)
r['nvvfx'] = importlib.util.find_spec('nvvfx') is not None
flash_modules = ('block_sparse_attn', 'einops', 'safetensors', 'PIL', 'tqdm')
seed_modules = ('torchvision', 'safetensors', 'psutil', 'einops',
                'omegaconf', 'diffusers', 'peft',
                'rotary_embedding_torch', 'cv2', 'gguf', 'matplotlib')
r['flashvsr'] = bool(r['cuda'] and sys.version_info[:2] == (3, 11)
                     and all(find(name) is not None for name in flash_modules))
r['seedvr2'] = bool(r['cuda'] and sys.version_info[:2] >= (3, 10)
                    and all(find(name) is not None for name in seed_modules))
print(json.dumps(r, ensure_ascii=True))
"""
    info: Dict[str, Any] = {"exe": python_exe, "version": "?", "torch": False,
                            "cuda": False, "torch_version": "", "gpu_name": "",
                            "nvvfx": False, "flashvsr": False,
                            "seedvr2": False}
    try:
        result = subprocess.run(
            [python_exe, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        for line in reversed(result.stdout.splitlines()):
            try:
                parsed = json.loads(line)
                info.update(parsed)
                break
            except json.JSONDecodeError:
                continue
        if result.returncode and not info.get("error"):
            info["error"] = result.stderr.strip()[-500:]
    except subprocess.TimeoutExpired:
        info["error"] = "检测超时"
    except OSError as exc:
        info["error"] = str(exc)
    return info


_ENV_CACHE_VERSION = 2


def _load_cache(max_age: float = 24 * 3600) -> List[Dict[str, Any]]:
    try:
        with open(_cache_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if int(data.get("version", 0)) != _ENV_CACHE_VERSION:
            return []
        if time.time() - float(data.get("created", 0)) > max_age:
            return []
        valid = []
        for item in data.get("environments", []):
            exe = item.get("exe", "")
            if os.path.isfile(exe) and abs(os.path.getmtime(exe) - item.get("mtime", -1)) < 1:
                valid.append(item)
        return valid
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return []


def get_cached_python_envs() -> List[Dict[str, Any]]:
    """Return fresh explicit-scan results without probing any interpreter."""
    return [dict(item) for item in _load_cache()]


def get_cached_python_env(python_exe: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return a previously scanned environment without starting a subprocess.

    Automatic engine selection calls this function on the processing hot path.
    A missing or expired entry is intentionally treated as unverified: the
    explicit GUI scan remains the only operation that probes Python installs.
    """
    if not python_exe:
        return None
    try:
        wanted = os.path.normcase(os.path.realpath(os.path.abspath(python_exe)))
    except (OSError, TypeError, ValueError):
        return None
    for item in get_cached_python_envs():
        try:
            candidate = os.path.normcase(
                os.path.realpath(os.path.abspath(str(item.get("exe", "")))))
        except (OSError, TypeError, ValueError):
            continue
        if candidate == wanted:
            return dict(item)
    return None


def _save_cache(items: List[Dict[str, Any]]) -> None:
    try:
        path = _cache_path()
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        serialised = []
        for item in items:
            value = dict(item)
            value["mtime"] = os.path.getmtime(value["exe"])
            serialised.append(value)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"version": _ENV_CACHE_VERSION, "created": time.time(),
                       "environments": serialised}, handle,
                      ensure_ascii=False, indent=2)
    except OSError:
        _log.debug("无法写入环境缓存", exc_info=True)


def get_all_python_envs(timeout: float = 12.0, force_rescan: bool = False) -> List[Dict[str, Any]]:
    if not force_rescan:
        cached = _load_cache()
        if cached:
            return cached
    candidates = _find_python_candidates()
    if not candidates:
        return []
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(candidates))) as pool:
        futures = [pool.submit(check_python_env, exe, timeout) for exe in candidates]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                _log.debug("Python 环境检测失败: %s", exc)
    results.sort(key=lambda item: (
        not (item.get("cuda") and item.get("nvvfx")),
        not item.get("cuda"), not item.get("torch"), item.get("exe", "")))
    _save_cache(results)
    return results


def get_torch_python(force_rescan: bool = False) -> Optional[str]:
    """Return an external CUDA-Python path; ``None`` means use current process."""
    global _memory_result, _memory_checked
    if _memory_checked and not force_rescan:
        return _memory_result
    try:
        import torch
        if torch.cuda.is_available():
            _memory_result = None
            _memory_checked = True
            return None
    except (ImportError, OSError):
        pass
    for info in get_all_python_envs(force_rescan=force_rescan):
        if info.get("torch") and info.get("cuda"):
            exe = info.get("exe")
            if os.path.normcase(exe) == os.path.normcase(sys.executable):
                _memory_result = None
            else:
                _memory_result = exe
            _memory_checked = True
            return _memory_result
    _memory_result = None
    _memory_checked = True
    return None

def get_python_for_feature(feature: str,
                           force_rescan: bool = False) -> Optional[str]:
    """Return a scanned Python executable that provides an optional feature."""
    if feature not in {"flashvsr", "seedvr2"}:
        raise ValueError("Unknown Python feature: %s" % feature)
    for info in get_all_python_envs(force_rescan=force_rescan):
        if (info.get("torch") and info.get("cuda")
                and bool(info.get(feature))):
            return str(info["exe"])
    return None

import os
import sys


PACKAGE_NAME = "light_video_enhancer"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_bundle_dir() -> str:
    if is_frozen():
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_pkg_dir() -> str:
    if is_frozen():
        return os.path.join(sys._MEIPASS, PACKAGE_NAME)
    return os.path.dirname(os.path.abspath(__file__))


def get_pkg_file(*parts: str) -> str:
    return os.path.join(get_pkg_dir(), *parts)


def pkg_file_exists(*parts: str) -> bool:
    return os.path.exists(get_pkg_file(*parts))


def get_model_root() -> str:
    """Return the writable per-user model directory."""
    overridden = os.environ.get("LVE_MODEL_DIR", "").strip()
    if overridden:
        return os.path.abspath(os.path.expandvars(os.path.expanduser(overridden)))
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return os.path.join(local_app_data, "LightVideoEnhancer", "models")
    return os.path.join(os.path.expanduser("~"), ".light_video_enhancer", "models")


def get_model_file(*parts: str) -> str:
    """Resolve a weight file, preferring a downloaded per-user copy."""
    external = os.path.join(get_model_root(), *parts)
    if os.path.isfile(external):
        return external
    return get_pkg_file(*parts)


def get_model_dir(*parts: str) -> str:
    """Resolve a model directory, preferring downloaded resources."""
    external = os.path.join(get_model_root(), *parts)
    if os.path.isdir(external):
        return external
    return get_pkg_file(*parts)


def model_file_exists(*parts: str) -> bool:
    return os.path.isfile(get_model_file(*parts))


def get_data_file(*parts: str) -> str:
    package_path = get_pkg_file(*parts)
    if os.path.exists(package_path) or not is_frozen():
        return package_path
    return os.path.join(get_bundle_dir(), *parts)


def data_file_exists(*parts: str) -> bool:
    return os.path.exists(get_data_file(*parts))

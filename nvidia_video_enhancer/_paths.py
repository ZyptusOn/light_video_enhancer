import os
import sys


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def get_bundle_dir() -> str:
    if is_frozen():
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_pkg_dir() -> str:
    if is_frozen():
        return os.path.join(sys._MEIPASS, "nvidia_video_enhancer")
    return os.path.dirname(os.path.abspath(__file__))


def get_pkg_file(*parts: str) -> str:
    return os.path.join(get_pkg_dir(), *parts)


def pkg_file_exists(*parts: str) -> bool:
    return os.path.exists(get_pkg_file(*parts))


def get_data_file(*parts: str) -> str:
    return os.path.join(get_bundle_dir(), *parts)


def data_file_exists(*parts: str) -> bool:
    return os.path.exists(get_data_file(*parts))

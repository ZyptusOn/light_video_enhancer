"""
统一日志模块。

用于替代分散的 print() 调用，支持:
  - 日志级别控制 (DEBUG/INFO/WARNING/ERROR)
  - GUI 自定义 Handler 注入
  - 子模块通过 get_logger(__name__) 获取专属 logger
"""

import logging
import sys
from typing import Optional

_ROOT_LOGGER_NAME = "nve"

_logger: Optional[logging.Logger] = None


def _init_root_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger(_ROOT_LOGGER_NAME)
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False

    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "[%(levelname)-7s] %(message)s"
        ))
        _logger.addHandler(handler)

    return _logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    root = _init_root_logger()
    if name:
        return root.getChild(name.split(".")[-1] if "." in name else name)
    return root


def set_gui_handler(handler: Optional[logging.Handler]) -> None:
    root = _init_root_logger()
    root.handlers.clear()
    if handler is not None:
        root.addHandler(handler)
    else:
        default_handler = logging.StreamHandler(sys.stdout)
        default_handler.setLevel(logging.DEBUG)
        default_handler.setFormatter(logging.Formatter(
            "[%(levelname)-7s] %(message)s"
        ))
        root.addHandler(default_handler)

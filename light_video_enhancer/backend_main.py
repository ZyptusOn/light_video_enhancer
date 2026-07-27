"""GUI-free command-line entry point used by the standalone backend EXE."""

import ctypes
import json
import os
import sys
import threading
from typing import Iterable, List, Optional

from ._logging import get_logger


_log = get_logger("backend")
_PROGRESS_PREFIX = "__LVE_PROGRESS__"


def _remove_flag(argv: Iterable[str], flag: str):
    values = list(argv)
    present = flag in values
    return present, [value for value in values if value != flag]


def _progress_json(stage: str, current: int, total: int) -> None:
    payload = {"stage": stage, "current": int(current), "total": int(total)}
    print(_PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=True), flush=True)


def _configure_stdio() -> None:
    """Use UTF-8 for redirected output and both Windows console languages."""
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except (AttributeError, OSError, ValueError):
            pass
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def _listen_for_control(enhancer) -> None:
    """Accept lightweight commands from a parent GUI without extra IPC deps."""
    try:
        for line in sys.stdin:
            if line.strip().lower() == "cancel":
                enhancer.cancel()
                return
    except (OSError, ValueError):
        return


def _wait_for_interactive_close() -> None:
    """Keep an Explorer-launched console open long enough to read an error."""
    isatty = getattr(sys.stdin, "isatty", None)
    if not callable(isatty) or not isatty():
        return
    from .i18n import tr
    try:
        input(tr(
            "\n处理未能开始。按 Enter 关闭窗口。",
            "\nProcessing could not start. Press Enter to close this window."))
    except (EOFError, KeyboardInterrupt, OSError):
        pass


def main(argv: Optional[List[str]] = None) -> None:
    """Run the complete CLI without importing Tkinter or any GUI module."""
    _configure_stdio()
    values = list(sys.argv[1:] if argv is None else argv)

    from .i18n import extract_language
    try:
        _, values = extract_language(values)
    except ValueError as exc:
        _log.error("%s", exc)
        raise SystemExit(2)

    interactive_session = not values or values == ["--interactive"]
    from .frontend_protocol import handle_frontend_command
    try:
        if handle_frontend_command(values, _progress_json):
            return
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _log.error("%s", exc)
        raise SystemExit(1)

    if not values or values == ["--interactive"]:
        from .cli_app import interactive_arguments
        values = interactive_arguments()
        if not values:
            return

    if "--system-info" in values:
        from .utils import print_system_info
        print_system_info(deep="--deep" in values)
        return
    if values == ["--version"]:
        from . import __version__
        print(__version__)
        return

    progress_json, values = _remove_flag(values, "--progress-json")
    control_stdin, values = _remove_flag(values, "--control-stdin")

    # Parse first so --help and syntax errors stay fast and dependency-light.
    from .cli import parse_args
    config = parse_args(values)

    from .pipeline import ProcessingCancelled, VideoEnhancer
    try:
        enhancer = VideoEnhancer(
            config, progress_callback=_progress_json if progress_json else None)
        if control_stdin:
            threading.Thread(
                target=_listen_for_control, args=(enhancer,),
                name="lve-control", daemon=True).start()
        enhancer.run()
    except ProcessingCancelled as exc:
        _log.warning("%s", exc)
        raise SystemExit(130)
    except (FileNotFoundError, FileExistsError, ImportError,
            RuntimeError, ValueError) as exc:
        _log.error("%s", exc)
        if interactive_session:
            _wait_for_interactive_close()
        raise SystemExit(1)
    except KeyboardInterrupt:
        from .i18n import tr
        _log.warning(tr("用户取消", "Cancelled by user"))
        raise SystemExit(130)
    except Exception as exc:
        _log.exception("未预期的 CLI 错误: %s", exc)
        if interactive_session:
            _wait_for_interactive_close()
        raise SystemExit(1)

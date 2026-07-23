import json
import sys
import threading

from ._logging import get_logger

_log = get_logger("main")
_PROGRESS_PREFIX = "__LVE_PROGRESS__"


def _remove_flag(argv, flag):
    present = flag in argv
    return present, [value for value in argv if value != flag]


def _progress_json(stage: str, current: int, total: int) -> None:
    payload = {"stage": stage, "current": int(current), "total": int(total)}
    print(_PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=True), flush=True)


def _configure_stdio() -> None:
    """Keep the JSON-line protocol stable in frozen and legacy consoles."""
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




def main() -> None:
    _configure_stdio()
    argv = sys.argv[1:]
    from .i18n import extract_language
    try:
        _, argv = extract_language(argv)
    except ValueError as exc:
        _log.error("%s", exc)
        raise SystemExit(2)
    from .frontend_protocol import handle_frontend_command
    try:
        if handle_frontend_command(argv, _progress_json):
            return
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _log.error("%s", exc)
        raise SystemExit(1)

    if not argv or argv == ["--gui"] or argv == ["-g"]:
        from .gui import main as gui_main
        gui_main()
        return
    if "--system-info" in argv:
        from .utils import print_system_info
        print_system_info(deep="--deep" in argv)
        return
    if argv == ["--version"]:
        from . import __version__
        print(__version__)
        return
    progress_json, argv = _remove_flag(argv, "--progress-json")
    control_stdin, argv = _remove_flag(argv, "--control-stdin")
    from .cli import parse_args
    from .pipeline import ProcessingCancelled, VideoEnhancer
    try:
        enhancer = VideoEnhancer(
            parse_args(argv), progress_callback=_progress_json if progress_json else None)
        if control_stdin:
            threading.Thread(
                target=_listen_for_control, args=(enhancer,),
                name="lve-control", daemon=True).start()
        enhancer.run()
    except ProcessingCancelled as exc:
        _log.warning("%s", exc)
        raise SystemExit(130)
    except (FileNotFoundError, FileExistsError, ImportError, RuntimeError, ValueError) as exc:
        _log.error("%s", exc)
        raise SystemExit(1)
    except KeyboardInterrupt:
        from .i18n import tr
        _log.warning(tr("用户取消", "Cancelled by user"))
        raise SystemExit(130)


if __name__ == "__main__":
    main()

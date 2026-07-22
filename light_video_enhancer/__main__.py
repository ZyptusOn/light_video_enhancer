import sys

from ._logging import get_logger

_log = get_logger("main")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv == ["--gui"] or argv == ["-g"]:
        from .gui import main as gui_main
        gui_main()
        return
    if "--system-info" in argv:
        from .utils import print_system_info
        print_system_info(deep="--deep" in argv)
        return
    from .cli import parse_args
    from .pipeline import ProcessingCancelled, VideoEnhancer
    try:
        VideoEnhancer(parse_args(argv)).run()
    except ProcessingCancelled as exc:
        _log.warning("%s", exc)
        raise SystemExit(130)
    except (FileNotFoundError, FileExistsError, ImportError, RuntimeError, ValueError) as exc:
        _log.error("%s", exc)
        raise SystemExit(1)
    except KeyboardInterrupt:
        _log.warning("用户取消")
        raise SystemExit(130)


if __name__ == "__main__":
    main()

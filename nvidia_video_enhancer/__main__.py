import sys
from ._logging import get_logger

_log = get_logger("main")


def main():
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ("--gui", "-g")):
        from .gui import main as gui_main
        gui_main()
        return

    from .cli import parse_args
    from .pipeline import VideoEnhancer
    from .utils import print_system_info

    print_system_info()
    config = parse_args()
    enhancer = VideoEnhancer(config)
    try:
        enhancer.run()
    except FileNotFoundError as e:
        _log.error("%s", e)
        sys.exit(1)
    except ImportError as e:
        _log.error("依赖缺失: %s", e)
        sys.exit(1)
    except RuntimeError as e:
        _log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        _log.warning("用户取消")
        sys.exit(130)


if __name__ == "__main__":
    main()

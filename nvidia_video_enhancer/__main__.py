import sys


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
        print(f"[错误] {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"[依赖缺失] {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"[运行错误] {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[中断] 用户取消", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()

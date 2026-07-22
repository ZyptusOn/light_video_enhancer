"""Versioned, UI-neutral JSON-line protocol for graphical frontends."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Callable, List


PROTOCOL_VERSION = 1


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def capabilities_payload() -> dict:
    from . import __version__
    from .capabilities import quick_capabilities

    payload = quick_capabilities()
    payload["protocol_version"] = PROTOCOL_VERSION
    payload["version"] = __version__
    payload["gpus"] = [asdict(gpu) for gpu in payload.get("gpus", [])]
    payload["vendors"] = sorted(payload.get("vendors", ()))
    payload["encoders"] = list(payload.get("encoders", ()))
    return payload


def _model_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--download-model", metavar="PACK")
    action.add_argument("--install-model-pack", nargs=2, metavar=("PACK", "ZIP"))
    action.add_argument("--remove-model", metavar="PACK")
    parser.add_argument("--model-source", default="github",
                        choices=("github", "mirror", "custom"))
    parser.add_argument("--source-base")
    return parser


def handle_frontend_command(argv: List[str],
                            progress: Callable[[str, int, int], None]) -> bool:
    """Handle a frontend-only command and return whether argv was consumed."""
    if argv == ["--capabilities-json"]:
        _print(capabilities_payload())
        return True
    if "--environments-json" in argv:
        from ._env import get_all_python_envs
        _print(get_all_python_envs(force_rescan="--force" in argv))
        return True
    if argv == ["--models-json"]:
        from .model_manager import list_model_packs
        _print(list_model_packs())
        return True
    if not any(flag in argv for flag in (
            "--download-model", "--install-model-pack", "--remove-model")):
        return False

    from .model_manager import (
        download_model_pack, install_model_archive, list_model_packs,
        remove_downloaded_pack,
    )
    args = _model_parser().parse_args(argv)
    if args.download_model:
        download_model_pack(args.download_model, args.model_source,
                            args.source_base, progress)
    elif args.install_model_pack:
        pack_id, archive = args.install_model_pack
        install_model_archive(pack_id, archive, progress)
    elif args.remove_model:
        remove_downloaded_pack(args.remove_model)
    _print({"ok": True, "models": list_model_packs()})
    return True

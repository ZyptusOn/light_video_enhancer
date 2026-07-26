"""Build the small pure-Python runtime used by the optional FlashVSR adapter."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


FIXED_TIME = (2025, 11, 1, 0, 0, 0)
MINIMAL_INIT = (
    '"""FlashVSR runtime subset bundled by Light Video Enhancer."""\n'
)


def _write_bytes(bundle: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name.replace("\\", "/"), FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    bundle.writestr(info, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    required = (
        source / "diffsynth",
        source / "examples" / "WanVSR" / "utils" / "utils.py",
        source / "examples" / "WanVSR" / "utils" / "TCDecoder.py",
        source / "examples" / "WanVSR" / "prompt_tensor" / "posi_prompt.pth",
        source / "LICENSE",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing FlashVSR source assets: " + ", ".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", allowZip64=True) as bundle:
        for path in sorted((source / "diffsynth").rglob("*.py")):
            relative = path.relative_to(source).as_posix()
            data = (MINIMAL_INIT.encode("utf-8") if
                    relative == "diffsynth/__init__.py" else path.read_bytes())
            _write_bytes(bundle, relative, data)
        _write_bytes(bundle, "flashvsr_utils/__init__.py", b"")
        for name in ("utils.py", "TCDecoder.py"):
            path = source / "examples" / "WanVSR" / "utils" / name
            _write_bytes(bundle, "flashvsr_utils/" + name, path.read_bytes())
        _write_bytes(
            bundle, "flashvsr_assets/posi_prompt.pth",
            (source / "examples" / "WanVSR" / "prompt_tensor" /
             "posi_prompt.pth").read_bytes())
        _write_bytes(bundle, "FLASHVSR-LICENSE.txt",
                     (source / "LICENSE").read_bytes())
    print("%s (%d bytes)" % (output, output.stat().st_size))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the optional SeedVR2 low-VRAM runtime from a pinned upstream tree."""

import argparse
import os
import zipfile
from pathlib import Path


PINNED_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"
ROOT_FILES = (
    "inference_cli.py",
    "neg_emb.pt",
    "pos_emb.pt",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
)


def _files(source: Path):
    for name in ROOT_FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        yield path, name
    config = source / "configs_3b" / "main.yaml"
    if not config.is_file():
        raise FileNotFoundError(config)
    yield config, "configs_3b/main.yaml"
    for path in sorted((source / "src").rglob("*.py")):
        yield path, path.relative_to(source).as_posix()


def build(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=9) as archive:
        for path, relative in _files(source):
            info = zipfile.ZipInfo(relative, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
        info = zipfile.ZipInfo("UPSTREAM_COMMIT.txt", (2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(
            info,
            ("ComfyUI-SeedVR2_VideoUpscaler\n" + PINNED_COMMIT + "\n").encode(),
            compresslevel=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="build/upstream/ComfyUI-SeedVR2_VideoUpscaler")
    parser.add_argument(
        "--output",
        default="light_video_enhancer/external/seedvr2_runtime.zip")
    args = parser.parse_args()
    build(Path(args.source).resolve(), Path(args.output).resolve())
    print(os.path.abspath(args.output))


if __name__ == "__main__":
    main()

"""Build the optional SparkVSR runtime from a pinned upstream checkout."""

import argparse
import os
import zipfile
from pathlib import Path


PINNED_COMMIT = "a082284b80005bb5615c0f5f5f5ed66650b1b1e7"


def _writestr(bundle, name, data):
    info = zipfile.ZipInfo(name, (2026, 4, 4, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    bundle.writestr(info, data)


def build(source: Path, output: Path) -> None:
    wrapper = source / "ComfyUI-Spark" / "sparkvsr_wrapper"
    required = {
        "sparkvsr_wrapper/__init__.py": wrapper / "__init__.py",
        "sparkvsr_wrapper/infer.py": wrapper / "infer.py",
        "sparkvsr_wrapper/model_loader.py": wrapper / "model_loader.py",
        "sparkvsr_wrapper/preprocess.py": wrapper / "preprocess.py",
        "SPARKVSR_LICENSE.txt": source / "LICENSE",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing SparkVSR source assets: " + ", ".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(str(temporary), "w") as bundle:
        for name, path in required.items():
            _writestr(bundle, name, path.read_bytes())
        _writestr(bundle, "UPSTREAM_COMMIT.txt", (PINNED_COMMIT + "\n").encode())
    os.replace(str(temporary), str(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="build/upstream/SparkVSR",
        help="Pinned SparkVSR checkout")
    parser.add_argument(
        "--output",
        default="light_video_enhancer/external/sparkvsr_runtime.zip")
    args = parser.parse_args()
    build(Path(args.source), Path(args.output))


if __name__ == "__main__":
    main()

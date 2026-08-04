"""Build the optional OSDEnhancer runtime from a pinned upstream checkout."""

import argparse
import os
import zipfile
from pathlib import Path


PINNED_COMMIT = "64dd6e56331cf7ed44e987859d47fa26b57fa662"


def _writestr(bundle, name, data):
    info = zipfile.ZipInfo(name, (2026, 6, 25, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    bundle.writestr(info, data)


def build(source: Path, output: Path) -> None:
    required = {
        "pipeline/OSDEnhancer_pipeline.py":
            source / "pipeline" / "OSDEnhancer_pipeline.py",
        "transformer/CogVideoXTransformer3D_STVSR.py":
            source / "transformer" / "CogVideoXTransformer3D_STVSR.py",
        "transformer/modules.py": source / "transformer" / "modules.py",
        "vae/AutoencoderKLCogVideoX_STVSR.py":
            source / "vae" / "AutoencoderKLCogVideoX_STVSR.py",
        "vae/modules.py": source / "vae" / "modules.py",
        "OSDENHANCER_LICENSE.txt": source / "LICENSE",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing OSDEnhancer source assets: " + ", ".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(str(temporary), "w") as bundle:
        for package in ("pipeline", "transformer", "vae"):
            _writestr(bundle, package + "/__init__.py", b"")
        for name, path in required.items():
            _writestr(bundle, name, path.read_bytes())
        _writestr(bundle, "UPSTREAM_COMMIT.txt", (PINNED_COMMIT + "\n").encode())
    os.replace(str(temporary), str(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="build/upstream/OSDEnhancer",
        help="Pinned OSDEnhancer checkout")
    parser.add_argument(
        "--output",
        default="light_video_enhancer/external/osdenhancer_runtime.zip")
    args = parser.parse_args()
    build(Path(args.source), Path(args.output))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build deterministic downloadable model archives and their runtime manifest."""

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from light_video_enhancer.model_manager import MODEL_PACKS, MODEL_PROTOCOL_VERSION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "dist" / "model-packs"))
    args = parser.parse_args()
    package = ROOT / "light_video_enhancer"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol_version": MODEL_PROTOCOL_VERSION, "packs": {}}

    for pack in MODEL_PACKS:
        if pack.get("downloads"):
            download_size = int(pack.get("remote_download_size", 0))
            manifest["packs"][pack["id"]] = {
                "archive": pack["archive"],
                "archive_size": download_size,
                "installed_size": 0,
                "sha256": "",
                "files": dict(pack.get("remote_hashes", {})),
            }
            print("%s: remote %.1f MiB" % (
                pack["id"], download_size / 1048576))
            continue
        archive = output / pack["archive"]
        file_hashes = {}
        installed_size = 0
        with zipfile.ZipFile(str(archive), "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as bundle:
            for relative in sorted(pack["files"]):
                source = package.joinpath(*relative.split("/"))
                if not source.is_file():
                    raise SystemExit("Missing model source: %s" % source)
                data = source.read_bytes()
                installed_size += len(data)
                file_hashes[relative] = hashlib.sha256(data).hexdigest()
                info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                bundle.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                                compresslevel=9)
        manifest["packs"][pack["id"]] = {
            "archive": pack["archive"],
            "archive_size": archive.stat().st_size,
            "installed_size": installed_size,
            "sha256": sha256(archive),
            "files": file_hashes,
        }
        print("%s: %.1f MiB -> %.1f MiB" % (
            pack["id"], installed_size / 1048576, archive.stat().st_size / 1048576))

    manifest_path = package / "model_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("Manifest: %s" % manifest_path)


if __name__ == "__main__":
    main()

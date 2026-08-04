#!/usr/bin/env python3
"""Build the pinned, auditable VFIMamba runtime used by the FI worker."""
import argparse
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

VFIMAMBA_REPOSITORY = "https://github.com/MCG-NJU/VFIMamba.git"
VFIMAMBA_COMMIT = "8df805eb054cd423e188d509210731cffe9438af"
MAMBA_REPOSITORY = "https://github.com/state-spaces/mamba.git"
MAMBA_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"

def _checkout(repository: str, commit: str, directory: Path) -> None:
    subprocess.run(["git", "clone", "--filter=blob:none", repository, str(directory)], check=True)
    subprocess.run(["git", "checkout", commit], cwd=str(directory), check=True)

def _scan_fallback(source: str) -> str:
    match = re.search(r"def selective_scan_ref\(.*?(?=\n\nclass MambaInnerFn)", source, flags=re.DOTALL)
    if not match:
        raise RuntimeError("official selective_scan_ref implementation was not found")
    return ('"""Official Mamba selective-scan reference fallback (Apache-2.0)."""\n\n'
            "import torch\nimport torch.nn.functional as F\n"
            "from einops import rearrange, repeat\n\n" + match.group(0).rstrip() + "\n")

def build(vfimamba: Path, mamba: Path, output: Path) -> None:
    expected = {"VFIMamba": VFIMAMBA_COMMIT, "mamba": MAMBA_COMMIT}
    for name, path in (("VFIMamba", vfimamba), ("mamba", mamba)):
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(path), text=True).strip()
        if revision != expected[name]:
            raise RuntimeError("%s checkout is %s, expected %s" % (name, revision, expected[name]))
    files = {
        "config.py": vfimamba / "config.py",
        "Trainer_finetune.py": vfimamba / "Trainer_finetune.py",
        "model/__init__.py": vfimamba / "model" / "__init__.py",
        "model/feature_extractor.py": vfimamba / "model" / "feature_extractor.py",
        "model/flow_estimation.py": vfimamba / "model" / "flow_estimation.py",
        "model/refine.py": vfimamba / "model" / "refine.py",
        "model/warplayer.py": vfimamba / "model" / "warplayer.py",
        "LICENSE-VFIMamba.txt": vfimamba / "LICENSE",
        "LICENSE-Mamba.txt": mamba / "LICENSE",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing runtime inputs: " + ", ".join(missing))
    feature = files["model/feature_extractor.py"].read_text(encoding="utf-8")
    old_import = ("from mamba_ssm.ops.selective_scan_interface import "
                  "selective_scan_fn, selective_scan_ref")
    replacement = '''try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    VFIMAMBA_SCAN_BACKEND = "mamba_ssm CUDA"
except (ImportError, OSError, RuntimeError):
    from vfimamba_selective_scan import selective_scan_ref
    selective_scan_fn = selective_scan_ref
    VFIMAMBA_SCAN_BACKEND = "PyTorch reference"'''
    if feature.count(old_import) != 1:
        raise RuntimeError("unexpected VFIMamba mamba_ssm import")
    feature = feature.replace(old_import, replacement)
    scan_source = (mamba / "mamba_ssm" / "ops" / "selective_scan_interface.py").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, source in files.items():
            data = feature.encode("utf-8") if name == "model/feature_extractor.py" else source.read_bytes()
            archive.writestr(name, data)
        archive.writestr("vfimamba_selective_scan.py", _scan_fallback(scan_source).encode("utf-8"))
        archive.writestr("VFIMAMBA_COMMIT.txt", VFIMAMBA_COMMIT + "\n")
        archive.writestr("MAMBA_COMMIT.txt", MAMBA_COMMIT + "\n")
    print("built %s (%d bytes)" % (output, output.stat().st_size))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vfimamba-source", type=Path)
    parser.add_argument("--mamba-source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("light_video_enhancer/external/vfimamba_runtime.zip"))
    args = parser.parse_args()
    if args.vfimamba_source and args.mamba_source:
        build(args.vfimamba_source.resolve(), args.mamba_source.resolve(), args.output.resolve())
        return
    with tempfile.TemporaryDirectory(prefix="lve_vfimamba_") as temp:
        root = Path(temp)
        vf, ma = root / "VFIMamba", root / "mamba"
        _checkout(VFIMAMBA_REPOSITORY, VFIMAMBA_COMMIT, vf)
        _checkout(MAMBA_REPOSITORY, MAMBA_COMMIT, ma)
        build(vf, ma, args.output.resolve())

if __name__ == "__main__":
    main()

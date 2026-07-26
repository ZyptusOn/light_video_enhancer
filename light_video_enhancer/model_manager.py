"""Shared model catalog and safe per-user model installation.

Both graphical frontends and the CLI query this module through the versioned
JSON protocol in ``python -m light_video_enhancer``.  Processing engines only
use ``_paths.get_model_*`` and therefore do not know whether a weight was
bundled or downloaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional

from ._paths import get_model_root, get_pkg_file


MODEL_PROTOCOL_VERSION = 1
MODEL_RELEASE_TAG = "models-v1"
MODEL_RELEASE_BASE = (
    "https://github.com/ZyptusOn/light_video_enhancer/releases/download/"
    + MODEL_RELEASE_TAG
)
MODEL_SOURCES = {
    "github": MODEL_RELEASE_BASE,
    "mirror": "https://ghproxy.net/" + MODEL_RELEASE_BASE,
}


def _pack(pack_id: str, archive: str, name_zh: str, name_en: str,
          description_zh: str, description_en: str,
          files: Iterable[str]) -> dict:
    return {
        "id": pack_id,
        "archive": archive,
        "name": {"zh-CN": name_zh, "en-US": name_en},
        "description": {"zh-CN": description_zh, "en-US": description_en},
        "files": tuple(path.replace("\\", "/") for path in files),
    }


def _remote_pack(*args, downloads: Dict[str, str],
                 official_base: str, mirror_base: str,
                 download_size: int = 0,
                 hashes: Optional[Dict[str, str]] = None) -> dict:
    pack = _pack(*args)
    pack.update({
        "downloads": dict(downloads),
        "remote_bases": {
            "official": official_base,
            "mirror": mirror_base,
        },
        "remote_download_size": int(download_size),
        "remote_hashes": dict(hashes or {}),
    })
    return pack


MODEL_PACKS = (
    _pack(
        "rife-pytorch", "lve-model-rife-pytorch.zip",
        "RIFE PyTorch 插帧", "RIFE PyTorch interpolation",
        "质量最高的 CUDA 插帧模型，需要外部 PyTorch 环境。",
        "High-quality CUDA interpolation; requires an external PyTorch environment.",
        ["fi/flownet.pkl"],
    ),
    _pack(
        "ema-vfi-small", "lve-model-ema-vfi-small.zip",
        "EMA-VFI Small 插帧", "EMA-VFI Small interpolation",
        "高效 CUDA 任意时刻插帧模型，支持 2x 至 4x 特征复用。",
        "Efficient arbitrary-timestep CUDA interpolation with feature reuse from 2x to 4x.",
        ["fi/ema_vfi/ours_small_t.pkl"],
    ),
    _remote_pack(
        "flashvsr-v1.1", "flashvsr-v1.1",
        "FlashVSR v1.1（可选，约 6.5 GiB）",
        "FlashVSR v1.1 (optional, about 6.5 GiB)",
        "Win10/11 实验性扩散视频超分；需要独立 Python 3.11 CUDA 与 Block-Sparse Attention。",
        "Experimental diffusion VSR for Windows 10/11; requires a separate Python 3.11 CUDA environment with Block-Sparse Attention.",
        [
            "flashvsr-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors",
            "flashvsr-v1.1/LQ_proj_in.ckpt",
            "flashvsr-v1.1/TCDecoder.ckpt",
        ],
        downloads={
            "flashvsr-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors": "diffusion_pytorch_model_streaming_dmd.safetensors",
            "flashvsr-v1.1/LQ_proj_in.ckpt": "LQ_proj_in.ckpt",
            "flashvsr-v1.1/TCDecoder.ckpt": "TCDecoder.ckpt",
        },
        official_base="https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/main",
        mirror_base="https://hf-mirror.com/JunhaoZhuang/FlashVSR-v1.1/resolve/main",
        download_size=6925634764,
        hashes={
            "flashvsr-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors": "bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be",
            "flashvsr-v1.1/LQ_proj_in.ckpt": "d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2",
            "flashvsr-v1.1/TCDecoder.ckpt": "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161",
        },
    ),
    _remote_pack(
        "seedvr2-3b-fp8", "seedvr2-3b-fp8",
        "SeedVR2 3B FP8（可选，约 3.6 GiB）",
        "SeedVR2 3B FP8 (optional, about 3.6 GiB)",
        "Win10/11 重型视频修复；为 8-16 GB 显存启用分块 VAE 与模型交换。",
        "Heavy Win10/11 video restoration with tiled VAE and model swapping for 8-16 GB VRAM.",
        [
            "seedvr2-3b-fp8/seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            "seedvr2-3b-fp8/ema_vae_fp16.safetensors",
        ],
        downloads={
            "seedvr2-3b-fp8/seedvr2_ema_3b_fp8_e4m3fn.safetensors": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            "seedvr2-3b-fp8/ema_vae_fp16.safetensors": "ema_vae_fp16.safetensors",
        },
        official_base="https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main",
        mirror_base="https://hf-mirror.com/numz/SeedVR2_comfyUI/resolve/main",
        download_size=3892869510,
        hashes={
            "seedvr2-3b-fp8/seedvr2_ema_3b_fp8_e4m3fn.safetensors": "3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff",
            "seedvr2-3b-fp8/ema_vae_fp16.safetensors": "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1",
        },
    ),
    _pack(
        "rife-ncnn", "lve-model-rife-ncnn.zip",
        "RIFE NCNN 插帧", "RIFE NCNN interpolation",
        "便携 Vulkan 插帧模型，适用于 NVIDIA、AMD 与 Intel。",
        "Portable Vulkan interpolation for NVIDIA, AMD, and Intel GPUs.",
        ["ncnn/rife/rife-v4.6/flownet.param", "ncnn/rife/rife-v4.6/flownet.bin"],
    ),
    _pack(
        "ifrnet-ncnn", "lve-model-ifrnet-ncnn.zip",
        "IFRNet NCNN 插帧", "IFRNet NCNN interpolation",
        "轻量跨显卡 Vulkan 插帧，包含小型、标准与大型质量档位。",
        "Lightweight cross-vendor Vulkan interpolation with small, base, and large models.",
        [
            "ncnn/ifrnet/IFRNet_S_Vimeo90K/ifrnet.param",
            "ncnn/ifrnet/IFRNet_S_Vimeo90K/ifrnet.bin",
            "ncnn/ifrnet/IFRNet_Vimeo90K/ifrnet.param",
            "ncnn/ifrnet/IFRNet_Vimeo90K/ifrnet.bin",
            "ncnn/ifrnet/IFRNet_L_Vimeo90K/ifrnet.param",
            "ncnn/ifrnet/IFRNet_L_Vimeo90K/ifrnet.bin",
        ],
    ),
    _pack(
        "span-ncnn", "lve-model-span-ncnn.zip",
        "SPAN NCNN 超分", "SPAN NCNN super resolution",
        "轻量 Vulkan 超分，提供 2×/4× 与 48/52 通道模型。",
        "Lightweight Vulkan super resolution with 2x/4x and 48/52-channel models.",
        [
            "ncnn/span/spanx2_ch48.param",
            "ncnn/span/spanx2_ch48.bin",
            "ncnn/span/spanx2_ch52.param",
            "ncnn/span/spanx2_ch52.bin",
            "ncnn/span/spanx4_ch48.param",
            "ncnn/span/spanx4_ch48.bin",
            "ncnn/span/spanx4_ch52.param",
            "ncnn/span/spanx4_ch52.bin",
        ],
    ),
    _pack(
        "realcugan", "lve-model-realcugan.zip",
        "Real-CUGAN 超分", "Real-CUGAN super resolution",
        "动画与线稿友好的 Vulkan 模型，包含 2x、3x、4x 质量档位。",
        "Vulkan models for animation and line art, with 2x, 3x, and 4x variants.",
        [
            "ncnn/realcugan/models-se/up2x-conservative.param",
            "ncnn/realcugan/models-se/up2x-conservative.bin",
            "ncnn/realcugan/models-se/up2x-denoise1x.param",
            "ncnn/realcugan/models-se/up2x-denoise1x.bin",
            "ncnn/realcugan/models-se/up2x-denoise2x.param",
            "ncnn/realcugan/models-se/up2x-denoise2x.bin",
            "ncnn/realcugan/models-se/up2x-denoise3x.param",
            "ncnn/realcugan/models-se/up2x-denoise3x.bin",
            "ncnn/realcugan/models-se/up2x-no-denoise.param",
            "ncnn/realcugan/models-se/up2x-no-denoise.bin",
            "ncnn/realcugan/models-se/up3x-conservative.param",
            "ncnn/realcugan/models-se/up3x-conservative.bin",
            "ncnn/realcugan/models-se/up3x-denoise3x.param",
            "ncnn/realcugan/models-se/up3x-denoise3x.bin",
            "ncnn/realcugan/models-se/up3x-no-denoise.param",
            "ncnn/realcugan/models-se/up3x-no-denoise.bin",
            "ncnn/realcugan/models-se/up4x-conservative.param",
            "ncnn/realcugan/models-se/up4x-conservative.bin",
            "ncnn/realcugan/models-se/up4x-denoise3x.param",
            "ncnn/realcugan/models-se/up4x-denoise3x.bin",
            "ncnn/realcugan/models-se/up4x-no-denoise.param",
            "ncnn/realcugan/models-se/up4x-no-denoise.bin",
        ],
    ),
    _pack(
        "realesrgan-fast", "lve-model-realesrgan-fast.zip",
        "Real-ESRGAN 视频模型", "Real-ESRGAN video models",
        "速度优先的动漫视频 2x、3x、4x 模型。",
        "Speed-oriented 2x, 3x, and 4x animation video models.",
        [
            "ncnn/realesrgan/models/realesr-animevideov3-x2.param",
            "ncnn/realesrgan/models/realesr-animevideov3-x2.bin",
            "ncnn/realesrgan/models/realesr-animevideov3-x3.param",
            "ncnn/realesrgan/models/realesr-animevideov3-x3.bin",
            "ncnn/realesrgan/models/realesr-animevideov3-x4.param",
            "ncnn/realesrgan/models/realesr-animevideov3-x4.bin",
        ],
    ),
    _pack(
        "realesrgan-anime", "lve-model-realesrgan-anime.zip",
        "Real-ESRGAN 动画高质量模型", "Real-ESRGAN anime HQ model",
        "面向动画内容的高质量 4x 模型。",
        "High-quality 4x model for animation content.",
        [
            "ncnn/realesrgan/models/realesrgan-x4plus-anime.param",
            "ncnn/realesrgan/models/realesrgan-x4plus-anime.bin",
        ],
    ),
    _pack(
        "realesrgan-general", "lve-model-realesrgan-general.zip",
        "Real-ESRGAN 通用高质量模型", "Real-ESRGAN general HQ model",
        "面向真实影像的通用高质量 4x 模型。",
        "General-purpose high-quality 4x model for real-world footage.",
        [
            "ncnn/realesrgan/models/realesrgan-x4plus.param",
            "ncnn/realesrgan/models/realesrgan-x4plus.bin",
        ],
    ),
    _pack(
        "esrgan-classic", "lve-model-esrgan-classic.zip",
        "经典 ESRGAN 模型", "Classic ESRGAN model",
        "经典 ESRGAN 4x 模型，适合保留锐利纹理。",
        "Classic ESRGAN 4x model for crisp texture reconstruction.",
        [
            "ncnn/realesrgan/models/esrgan-x4.param",
            "ncnn/realesrgan/models/esrgan-x4.bin",
        ],
    ),
)

_BY_ID = {pack["id"]: pack for pack in MODEL_PACKS}
ProgressCallback = Callable[[str, int, int], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_generated_manifest() -> dict:
    path = Path(get_pkg_file("model_manifest.json"))
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _metadata(pack_id: str) -> dict:
    manifest = _load_generated_manifest()
    value = manifest.get("packs", {}).get(pack_id, {})
    return value if isinstance(value, dict) else {}


def _existing_kind(relative_path: str) -> Optional[str]:
    external = Path(get_model_root(), *PurePosixPath(relative_path).parts)
    if external.is_file():
        return "downloaded"
    bundled = Path(get_pkg_file(*PurePosixPath(relative_path).parts))
    if bundled.is_file():
        return "bundled"
    return None


def list_model_packs() -> dict:
    packs: List[dict] = []
    for definition in MODEL_PACKS:
        kinds = [_existing_kind(path) for path in definition["files"]]
        if kinds and all(kind == "downloaded" for kind in kinds):
            status = "downloaded"
        elif kinds and all(kind in {"downloaded", "bundled"} for kind in kinds):
            status = "bundled" if "bundled" in kinds else "downloaded"
        elif any(kinds):
            status = "partial"
        else:
            status = "missing"
        metadata = _metadata(definition["id"])
        size = int(metadata.get("installed_size", 0))
        if not size:
            for relative in definition["files"]:
                external = Path(get_model_root(), *PurePosixPath(relative).parts)
                path = external if external.is_file() else Path(get_pkg_file(*PurePosixPath(relative).parts))
                if path.is_file():
                    size += path.stat().st_size
        packs.append({
            **{key: value for key, value in definition.items()
               if key not in {"files", "downloads", "remote_bases",
                              "remote_download_size", "remote_hashes"}},
            "status": status,
            "installed": status in {"bundled", "downloaded"},
            "installed_size": size,
            "download_size": int(
                metadata.get("archive_size", 0) or
                definition.get("remote_download_size", 0)),
        })
    return {
        "protocol_version": MODEL_PROTOCOL_VERSION,
        "model_root": get_model_root(),
        "sources": [
            {"id": "github", "name": "GitHub", "base_url": MODEL_SOURCES["github"]},
            {"id": "mirror", "name": "GitHub Proxy", "base_url": MODEL_SOURCES["mirror"]},
            {"id": "custom", "name": "Custom / 自定义", "base_url": ""},
        ],
        "packs": packs,
    }


def _pack_by_id(pack_id: str) -> dict:
    try:
        return _BY_ID[pack_id]
    except KeyError:
        raise ValueError("Unknown model pack / 未知模型包: %s" % pack_id)


def _url_for(pack: dict, source: str, custom_base: Optional[str]) -> str:
    if source == "custom":
        base = (custom_base or "").strip()
        if not base:
            raise ValueError("A custom source URL is required / 请输入自定义下载源")
    else:
        try:
            base = MODEL_SOURCES[source]
        except KeyError:
            raise ValueError("Unknown model source / 未知下载源: %s" % source)
    if "{archive}" in base:
        return base.replace("{archive}", pack["archive"])
    return base.rstrip("/") + "/" + pack["archive"]


def _safe_member_name(raw: str) -> str:
    path = PurePosixPath(raw.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe model archive path / 模型包路径不安全: %s" % raw)
    return path.as_posix()


def install_model_archive(pack_id: str, archive: str,
                          progress: Optional[ProgressCallback] = None) -> None:
    pack = _pack_by_id(pack_id)
    expected = set(pack["files"])
    metadata = _metadata(pack_id)
    archive_path = Path(archive)
    expected_archive_hash = str(metadata.get("sha256", ""))
    if expected_archive_hash and _sha256(archive_path) != expected_archive_hash:
        raise ValueError("Model archive checksum mismatch / 模型包校验失败")

    root = Path(get_model_root())
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lve_model_", dir=str(root)) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(str(archive_path), "r") as bundle:
            members = {
                _safe_member_name(info.filename): info
                for info in bundle.infolist() if not info.is_dir()
            }
            if set(members) != expected:
                missing = sorted(expected - set(members))
                extra = sorted(set(members) - expected)
                raise ValueError(
                    "Unexpected model archive contents / 模型包内容不匹配; "
                    "missing=%s extra=%s" % (missing, extra))
            total = sum(info.file_size for info in members.values())
            current = 0
            for relative in sorted(expected):
                target = staging.joinpath(*PurePosixPath(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(members[relative], "r") as source, target.open("wb") as output:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        current += len(chunk)
                        if progress:
                            progress("install", current, total)
                expected_hash = str(metadata.get("files", {}).get(relative, ""))
                if expected_hash and _sha256(target) != expected_hash:
                    raise ValueError("Model file checksum mismatch / 模型文件校验失败: " + relative)
            for relative in sorted(expected):
                source = staging.joinpath(*PurePosixPath(relative).parts)
                target = root.joinpath(*PurePosixPath(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(source), str(target))


def _remote_file_url(pack: dict, relative: str, source: str,
                     custom_base: Optional[str]) -> str:
    filename = pack["downloads"][relative]
    if source == "custom":
        base = (custom_base or "").strip()
        if not base:
            raise ValueError("A custom source URL is required / 请输入自定义下载源")
    else:
        key = "mirror" if source == "mirror" else "official"
        base = pack["remote_bases"][key]
    if "{file}" in base:
        return base.replace("{file}", filename)
    return base.rstrip("/") + "/" + filename


def _download_remote_pack(pack: dict, source: str,
                          custom_base: Optional[str],
                          progress: Optional[ProgressCallback]) -> None:
    root = Path(get_model_root())
    download_dir = root / ".downloads" / pack["id"]
    download_dir.mkdir(parents=True, exist_ok=True)
    for relative in pack["files"]:
        target = root.joinpath(*PurePosixPath(relative).parts)
        if target.is_file():
            continue
        filename = pack["downloads"][relative]
        part = download_dir / (filename + ".part")
        existing = part.stat().st_size if part.is_file() else 0
        headers = {"User-Agent": "LightVideoEnhancer/1"}
        if existing:
            headers["Range"] = "bytes=%d-" % existing
        request = urllib.request.Request(
            _remote_file_url(pack, relative, source, custom_base),
            headers=headers)
        with urllib.request.urlopen(request, timeout=90) as response:
            partial = int(getattr(response, "status", 200) or 200) == 206
            if existing and not partial:
                existing = 0
            mode = "ab" if existing and partial else "wb"
            remaining = int(response.headers.get("Content-Length") or 0)
            total = existing + remaining if remaining else 0
            current = existing
            with part.open(mode) as output:
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    current += len(chunk)
                    if progress:
                        progress("download", current, total)
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_hash = str(
            pack.get("remote_hashes", {}).get(relative, "") or
            _metadata(pack["id"]).get("files", {}).get(relative, ""))
        if expected_hash and _sha256(part) != expected_hash:
            raise ValueError(
                "Model file checksum mismatch / 模型文件校验失败: " +
                relative)
        os.replace(str(part), str(target))


def download_model_pack(pack_id: str, source: str = "github",
                        custom_base: Optional[str] = None,
                        progress: Optional[ProgressCallback] = None) -> None:
    pack = _pack_by_id(pack_id)
    if pack.get("downloads"):
        _download_remote_pack(pack, source, custom_base, progress)
        return
    url = _url_for(pack, source, custom_base)
    root = Path(get_model_root())
    download_dir = root / ".downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    archive = download_dir / pack["archive"]
    part = archive.with_suffix(archive.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "LightVideoEnhancer/1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response, part.open("wb") as output:
            total = int(response.headers.get("Content-Length") or 0)
            current = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                current += len(chunk)
                if progress:
                    progress("download", current, total)
        os.replace(str(part), str(archive))
        install_model_archive(pack_id, str(archive), progress)
    finally:
        try:
            part.unlink()
        except FileNotFoundError:
            pass


def remove_downloaded_pack(pack_id: str) -> None:
    pack = _pack_by_id(pack_id)
    root = Path(get_model_root())
    for relative in pack["files"]:
        path = root.joinpath(*PurePosixPath(relative).parts)
        try:
            path.unlink()
        except FileNotFoundError:
            continue
    for directory in sorted(root.rglob("*"), reverse=True) if root.exists() else []:
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                pass


def model_weight_paths() -> set:
    """Return package-relative weight paths omitted by the light build."""
    return {path.replace("/", os.sep) for pack in MODEL_PACKS for path in pack["files"]}

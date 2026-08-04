"""Shared model catalog and safe per-user model installation.

Both graphical frontends and the CLI query this module through the versioned
JSON protocol in ``python -m light_video_enhancer``.  Processing engines only
use ``_paths.get_model_*`` and therefore do not know whether a weight was
bundled or downloaded.
"""

from __future__ import annotations

import hashlib
import html.parser
import http.cookiejar
import json
import os
import re
import shutil
import tempfile
import urllib.parse
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
    _remote_pack(
        "seedvr2-7b-q4", "seedvr2-7b-q4",
        "SeedVR2 7B Q4（可选，约 4.4 GiB）",
        "SeedVR2 7B Q4 (optional, about 4.4 GiB)",
        "消费级显卡质量档；复用 SeedVR2 3B 模型包中的 VAE，需要至少约 11 GiB 显存。",
        "Consumer-GPU quality profile; reuses the VAE from the SeedVR2 3B pack and requires about 11 GiB VRAM.",
        ["seedvr2-7b-q4/seedvr2_ema_7b-Q4_K_M.gguf"],
        downloads={
            "seedvr2-7b-q4/seedvr2_ema_7b-Q4_K_M.gguf":
                "seedvr2_ema_7b-Q4_K_M.gguf",
        },
        official_base="https://huggingface.co/cmeka/SeedVR2-GGUF/resolve/main",
        mirror_base="https://hf-mirror.com/cmeka/SeedVR2-GGUF/resolve/main",
        download_size=4758306592,
        hashes={
            "seedvr2-7b-q4/seedvr2_ema_7b-Q4_K_M.gguf":
                "db9cb2ad90ebd40d2e8c29da2b3fc6fd03ba87cd58cbadceccca13ad27162789",
        },
    ),
    _remote_pack(
        "seedvr2-7b-sharp-q4", "seedvr2-7b-sharp-q4",
        "SeedVR2 7B Sharp Q4（可选，约 4.4 GiB）",
        "SeedVR2 7B Sharp Q4 (optional, about 4.4 GiB)",
        "消费级显卡极致细节档；可能生成更锐利细节，复用 3B 模型包 VAE。",
        "Maximum-detail consumer-GPU profile; may generate sharper detail and reuses the VAE from the 3B pack.",
        ["seedvr2-7b-sharp-q4/seedvr2_ema_7b_sharp-Q4_K_M.gguf"],
        downloads={
            "seedvr2-7b-sharp-q4/seedvr2_ema_7b_sharp-Q4_K_M.gguf":
                "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
        },
        official_base="https://huggingface.co/cmeka/SeedVR2-GGUF/resolve/main",
        mirror_base="https://hf-mirror.com/cmeka/SeedVR2-GGUF/resolve/main",
        download_size=4758306592,
        hashes={
            "seedvr2-7b-sharp-q4/seedvr2_ema_7b_sharp-Q4_K_M.gguf":
                "7aed800ac4eb8e0d18569a954c0ff35f5a1caa3ed5d920e66cc31405f75b6e69",
        },
    ),
    _remote_pack(
        "dloral-core", "dloral-core",
        "DLoRAL 核心模型（可选，约 8.1 GiB）",
        "DLoRAL core models (optional, about 8.1 GiB)",
        "一阶段 4× 扩散视频超分；包含作者公开 checkpoint 与固定 SD 2.1 基座，不参与自动选择。",
        "One-step 4x diffusion VSR with the public author checkpoint and pinned SD 2.1 base; never auto-selected.",
        [
            "dloral-core/model.pkl",
            "dloral-core/spynet_20210409-c6c1bd09.pth",
            "dloral-core/stable-diffusion-2-1-base/scheduler/scheduler_config.json",
            "dloral-core/stable-diffusion-2-1-base/tokenizer/merges.txt",
            "dloral-core/stable-diffusion-2-1-base/tokenizer/special_tokens_map.json",
            "dloral-core/stable-diffusion-2-1-base/tokenizer/tokenizer_config.json",
            "dloral-core/stable-diffusion-2-1-base/tokenizer/vocab.json",
            "dloral-core/stable-diffusion-2-1-base/text_encoder/config.json",
            "dloral-core/stable-diffusion-2-1-base/text_encoder/model.safetensors",
            "dloral-core/stable-diffusion-2-1-base/unet/config.json",
            "dloral-core/stable-diffusion-2-1-base/unet/diffusion_pytorch_model.safetensors",
            "dloral-core/stable-diffusion-2-1-base/vae/config.json",
            "dloral-core/stable-diffusion-2-1-base/vae/diffusion_pytorch_model.safetensors",
        ],
        downloads={
            "dloral-core/model.pkl":
                "https://www.dropbox.com/scl/fi/gmw50778y1h51crghhp38/model.pkl?rlkey=bk42rqnc7aqryxf8suupr46jg&dl=1",
            "dloral-core/spynet_20210409-c6c1bd09.pth":
                "https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth",
            **{
                "dloral-core/stable-diffusion-2-1-base/" + path:
                    "https://huggingface.co/yujingsun/stable-diffusion-2-1-base/resolve/main/" + path
                for path in (
                    "scheduler/scheduler_config.json",
                    "tokenizer/merges.txt",
                    "tokenizer/special_tokens_map.json",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/vocab.json",
                    "text_encoder/config.json",
                    "text_encoder/model.safetensors",
                    "unet/config.json",
                    "unet/diffusion_pytorch_model.safetensors",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.safetensors",
                )
            },
        },
        official_base="https://github.com/yjsunnn/DLoRAL",
        mirror_base="https://github.com/yjsunnn/DLoRAL",
        download_size=8739659355,
        hashes={
            "dloral-core/model.pkl": "89ca79785c99fac07f59ad6876d2d707fcce87a088c15930956cb393e517d0df",
            "dloral-core/spynet_20210409-c6c1bd09.pth": "c6c1bd09b52d05ba17f3e701f549d6faf5e314aabce8ae462c1c171a8d6c4914",
            "dloral-core/stable-diffusion-2-1-base/scheduler/scheduler_config.json": "11ac5627d7df0fa344b875c4b5722b1767a8a2aa1684c2cf8b4d614300127234",
            "dloral-core/stable-diffusion-2-1-base/tokenizer/merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
            "dloral-core/stable-diffusion-2-1-base/tokenizer/special_tokens_map.json": "f118ab3a983206e4f32583448de6bd6aae4ee21869135cef1f5848a753cdaab6",
            "dloral-core/stable-diffusion-2-1-base/tokenizer/tokenizer_config.json": "d562b18a0c6d32a168fce3d7a342fce64579ebacffb98fb71b0affc494fc32d8",
            "dloral-core/stable-diffusion-2-1-base/tokenizer/vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
            "dloral-core/stable-diffusion-2-1-base/text_encoder/config.json": "7e33bfc475f6cf76e69b28b101993ca834d1f22ff4d35caefde32d63d4f42fba",
            "dloral-core/stable-diffusion-2-1-base/text_encoder/model.safetensors": "cce6febb0b6d876ee5eb24af35e27e764eb4f9b1d0b7c026c8c3333d4cfc916c",
            "dloral-core/stable-diffusion-2-1-base/unet/config.json": "ce0c6d379e3b1d3e1f79338de70c80a1e36f17a6372439a8249b6fb0dfa1b608",
            "dloral-core/stable-diffusion-2-1-base/unet/diffusion_pytorch_model.safetensors": "6dfae3e5f7d459b50f4b0850ead945972c75bb0e1897628933e169eb43974214",
            "dloral-core/stable-diffusion-2-1-base/vae/config.json": "424117cb534ce03497c41305ed868980123917b2b6abba4bbaa615e968772903",
            "dloral-core/stable-diffusion-2-1-base/vae/diffusion_pytorch_model.safetensors": "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815",
        },
    ),
    _remote_pack(
        "dloral-prompt", "dloral-prompt",
        "DLoRAL 内容提示模型（可选，约 5.2 GiB）",
        "DLoRAL content-prompt models (optional, about 5.2 GiB)",
        "RAM + DAPE 内容标签器；未安装时 DLoRAL 使用固定中性提示词。",
        "RAM + DAPE content tagger; DLoRAL uses a fixed neutral prompt when omitted.",
        [
            "dloral-prompt/ram_swin_large_14m.pth",
            "dloral-prompt/DAPE.pth",
        ],
        downloads={
            "dloral-prompt/ram_swin_large_14m.pth":
                "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth",
            "dloral-prompt/DAPE.pth":
                "https://drive.google.com/uc?export=download&id=1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm",
        },
        official_base="https://github.com/yjsunnn/DLoRAL",
        mirror_base="https://github.com/yjsunnn/DLoRAL",
        download_size=5632829366,
        hashes={
            "dloral-prompt/ram_swin_large_14m.pth": "15c729c793af28b9d107c69f85836a1356d76ea830d4714699fb62e55fcc08ed",
            "dloral-prompt/DAPE.pth": "a7028be2edcbe9ab0bd1c4ab6f2a2a86f4b44d32261a4faa50ae10fdd9b2feba",
        },
    ),
    _remote_pack(
        "osdenhancer-v1", "osdenhancer-v1",
        "OSDEnhancer 联合超分插帧（可选，约 12.0 GiB）",
        "OSDEnhancer joint STVSR (optional, about 12.0 GiB)",
        "一次完成 4× 空间超分和 2× 时间插帧；官方要求至少 80 GB 显存，不参与自动选择。",
        "Joint 4x spatial and 2x temporal enhancement; the author requires at least 80 GB VRAM; never auto-selected.",
        [
            "osdenhancer-v1/prompt_embeddings/empty.safetensors",
            "osdenhancer-v1/scheduler/scheduler_config.json",
            "osdenhancer-v1/transformer/config.json",
            "osdenhancer-v1/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "osdenhancer-v1/transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "osdenhancer-v1/transformer/diffusion_pytorch_model.safetensors.index.json",
            "osdenhancer-v1/vae/config.json",
            "osdenhancer-v1/vae/diffusion_pytorch_model.safetensors",
        ],
        downloads={
            **{
                "osdenhancer-v1/" + path:
                    "https://huggingface.co/W-Shuoyan/OSDEnhancer/resolve/main/" + path
                for path in (
                    "prompt_embeddings/empty.safetensors",
                    "scheduler/scheduler_config.json",
                    "transformer/config.json",
                    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
                    "transformer/diffusion_pytorch_model.safetensors.index.json",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.safetensors",
                )
            },
        },
        official_base="https://huggingface.co/W-Shuoyan/OSDEnhancer/resolve/main",
        mirror_base="https://hf-mirror.com/W-Shuoyan/OSDEnhancer/resolve/main",
        download_size=12846839231,
        hashes={
            "osdenhancer-v1/prompt_embeddings/empty.safetensors": "49738b5f634bc7c7ebad8e0ba01bf8c4eb5930b84c38d00f78dc0d5bc0a417cc",
            "osdenhancer-v1/scheduler/scheduler_config.json": "8c821359b9f0d625d87d0c9f095343e2c790acdf89b622dd020a2241647090ae",
            "osdenhancer-v1/transformer/config.json": "0eaff819b4b628a087b83427e528ce43b848aa0eb3e737b98f0689e272e241ba",
            "osdenhancer-v1/transformer/diffusion_pytorch_model-00001-of-00002.safetensors": "4ffe768fb2a9a384f666662c3ff62d37a861e90c58154f93cd65a30c94bca7c6",
            "osdenhancer-v1/transformer/diffusion_pytorch_model-00002-of-00002.safetensors": "9a8f8768f700fd51cc3fe15f8cc756478d1f1e63de51c7e3bbaed78f9b4dc82f",
            "osdenhancer-v1/transformer/diffusion_pytorch_model.safetensors.index.json": "b6208590d63174931273a705a56a0f2ac4dae057ea1cef33d1f9e7442bee7ef3",
            "osdenhancer-v1/vae/config.json": "1ccbe603fdd170cddc16b678e3cbf0e736de396edf00c4b66e32b3a18b060350",
            "osdenhancer-v1/vae/diffusion_pytorch_model.safetensors": "329fe3b7c3c45bf09697de2903cadbc65b8b999f7b612d972484d1061b47aa69",
        },
    ),
    _remote_pack(
        "sparkvsr-stage2", "sparkvsr-stage2",
        "SparkVSR Stage-2 关键帧视频超分（可选，约 39.3 GiB）",
        "SparkVSR Stage-2 keyframe VSR (optional, about 39.3 GiB)",
        "原生 4×、支持本地高质量关键帧传播；42.2 GB 权重，需要高端 CUDA 主机且不参与自动选择。",
        "Native 4x with local HQ-keyframe propagation; 42.2 GB of weights, requires a high-end CUDA host, and is never auto-selected.",
        ["sparkvsr-stage2/" + path for path in (
            "model_index.json", "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model-00001-of-00004.safetensors",
            "text_encoder/model-00002-of-00004.safetensors",
            "text_encoder/model-00003-of-00004.safetensors",
            "text_encoder/model-00004-of-00004.safetensors",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/added_tokens.json", "tokenizer/special_tokens_map.json",
            "tokenizer/spiece.model", "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
            "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
            "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
        )],
        downloads={"sparkvsr-stage2/" + path: path for path in (
            "model_index.json", "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model-00001-of-00004.safetensors",
            "text_encoder/model-00002-of-00004.safetensors",
            "text_encoder/model-00003-of-00004.safetensors",
            "text_encoder/model-00004-of-00004.safetensors",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/added_tokens.json", "tokenizer/special_tokens_map.json",
            "tokenizer/spiece.model", "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
            "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
            "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
        )},
        official_base="https://huggingface.co/JiongzeYu/SparkVSR/resolve/ec23be78e9f28c4433f61ec737ce6e17559cda90",
        mirror_base="https://hf-mirror.com/JiongzeYu/SparkVSR/resolve/ec23be78e9f28c4433f61ec737ce6e17559cda90",
        download_size=42199097809,
        hashes={
            "sparkvsr-stage2/model_index.json": "cf8378a2d8f65b1484764dd4d9e65ef650f0fb58d9abf674be11b9d26e77fa94",
            "sparkvsr-stage2/scheduler/scheduler_config.json": "8c821359b9f0d625d87d0c9f095343e2c790acdf89b622dd020a2241647090ae",
            "sparkvsr-stage2/text_encoder/config.json": "2b64ff3ea7f74ae9b2019daa8d48fc7b89685bdaad548ce541378e22934c3b07",
            "sparkvsr-stage2/text_encoder/model.safetensors.index.json": "a545bb25dc0f423d84be7b577311bba8bb7c6931f1eefcea65fc8b0a61a60a76",
            "sparkvsr-stage2/tokenizer/added_tokens.json": "ea5a91a3234f66ea642c8e672d67f0f493759a9bee6910ae304ea9b9492118b5",
            "sparkvsr-stage2/tokenizer/special_tokens_map.json": "7a1985a994c41886db38c719d2a3d2f40606663cc19d7c5d6a85d349320e06d2",
            "sparkvsr-stage2/tokenizer/tokenizer_config.json": "9c4f8a92630f7c6cc082831b291d6f46f18455d69d1a2a52c9b5f5303ab01469",
            "sparkvsr-stage2/transformer/config.json": "d12248f8e683cb4b619c6444e0d9d798c4a6e2fecf6b4e3d5b619ad4e93c2936",
            "sparkvsr-stage2/transformer/diffusion_pytorch_model.safetensors.index.json": "37903d14f3bfdfbc667787059a74890e62bfad2094b43215c0081537c1e40b88",
            "sparkvsr-stage2/vae/config.json": "cdc0a3c6c363878ad18920842b001fd16e7d65dceb1810d30a694be062e77e91",
            "sparkvsr-stage2/text_encoder/model-00001-of-00004.safetensors": "8e0e98242fed1de128a73ee6fee174aa30186c66f7d70811c8ebf67d72067f38",
            "sparkvsr-stage2/text_encoder/model-00002-of-00004.safetensors": "7309a33cd1780b20e3d84b2e14585d7a64ccea45079aecc86c507e381e00ed14",
            "sparkvsr-stage2/text_encoder/model-00003-of-00004.safetensors": "d278bf7cb4de9957cb67d8a92dc9273f88186f0c90e0f2a3de13882217d9a95c",
            "sparkvsr-stage2/text_encoder/model-00004-of-00004.safetensors": "69afce9a62402b82095baa1dfef572be92dfd9a4062693d49a5948e4d27585a5",
            "sparkvsr-stage2/tokenizer/spiece.model": "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
            "sparkvsr-stage2/transformer/diffusion_pytorch_model-00001-of-00005.safetensors": "94f98d9d81a66822b1f70add0b57a651df73726071d5f7d832361e5478061490",
            "sparkvsr-stage2/transformer/diffusion_pytorch_model-00002-of-00005.safetensors": "6a11deeb4f2b874410a8e8399a9a56da53704058f7afec38fc82f94202c7d1fb",
            "sparkvsr-stage2/transformer/diffusion_pytorch_model-00003-of-00005.safetensors": "5e23bdb33982887f21197504624dd9b026cd7adc2e2775c5f6b51ac1b93751ac",
            "sparkvsr-stage2/transformer/diffusion_pytorch_model-00004-of-00005.safetensors": "275ef2857f6106b4eb1d06f1da28889a5a686144893f3a0c1237d14d2b4629bd",
            "sparkvsr-stage2/transformer/diffusion_pytorch_model-00005-of-00005.safetensors": "8ab0b18ca0bb6f4166a2421ba04b8830fb29363075b7cadf35f4825f4f1fc0be",
            "sparkvsr-stage2/vae/diffusion_pytorch_model.safetensors": "a410e48d988c8224cef392b68db0654485cfd41f345f4a3a81d3e6b765bb995e",
        },
    ),
    _remote_pack(
        "vfimamba", "vfimamba",
        "VFIMamba S / Full 插帧（可选，约 316 MiB）",
        "VFIMamba S / Full interpolation (optional, about 316 MiB)",
        "状态空间视频插帧，包含轻量与完整质量模型；CUDA 扩展不可用时安全回退到较慢的 PyTorch 实现。",
        "State-space video interpolation with small and full models; safely falls back to a slower PyTorch implementation when its CUDA extension is unavailable.",
        ["vfimamba/VFIMamba_S.pkl", "vfimamba/VFIMamba.pkl"],
        downloads={
            "vfimamba/VFIMamba_S.pkl": "ckpt/VFIMamba_S.pkl",
            "vfimamba/VFIMamba.pkl": "ckpt/VFIMamba.pkl",
        },
        official_base="https://huggingface.co/MCG-NJU/VFIMamba_ckpts/resolve/7c383874883191d240bcc9435590eecc573f1055",
        mirror_base="https://hf-mirror.com/MCG-NJU/VFIMamba_ckpts/resolve/7c383874883191d240bcc9435590eecc573f1055",
        download_size=331712554,
        hashes={
            "vfimamba/VFIMamba_S.pkl":
                "ddc1e07e5917f1bbd254ca77e077354cf822c9af6cacd1434136e86b9961acc7",
            "vfimamba/VFIMamba.pkl":
                "c1dac5b08f4c41e95452f7a41b35347409fc58b0e25b3ba8de899144ff28a350",
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
    parsed_file = urllib.parse.urlparse(filename)
    if parsed_file.scheme == "https" and parsed_file.netloc:
        return filename
    return base.rstrip("/") + "/" + filename


def _remote_part_filename(download: str, relative: str) -> str:
    parsed = urllib.parse.urlparse(download)
    candidate = (urllib.parse.unquote(parsed.path).replace("\\", "/")
                 if parsed.scheme else download.replace("\\", "/"))
    filename = PurePosixPath(candidate).name
    if not filename or filename in {".", ".."}:
        filename = PurePosixPath(relative).name
    return filename


class _DriveDownloadForm(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = ""
        self.fields: Dict[str, str] = {}
        self._inside = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "form" and values.get("id") == "download-form":
            self._inside = True
            self.action = values.get("action", "")
        elif self._inside and tag == "input" and values.get("name"):
            self.fields[str(values["name"])] = str(values.get("value", ""))

    def handle_endtag(self, tag):
        if tag == "form":
            self._inside = False


def _drive_confirmation_url(document: str) -> str:
    parser = _DriveDownloadForm()
    parser.feed(document)
    parsed = urllib.parse.urlparse(parser.action)
    if (parsed.scheme != "https" or
            parsed.hostname != "drive.usercontent.google.com"):
        raise ValueError("Invalid Google Drive download confirmation form")
    required = {"id", "export", "confirm", "uuid"}
    if not required.issubset(parser.fields):
        raise ValueError("Incomplete Google Drive download confirmation form")
    return parser.action + "?" + urllib.parse.urlencode(parser.fields)


def _open_remote(url: str, headers: dict, timeout: float):
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname not in {"drive.google.com", "drive.usercontent.google.com"}:
        return urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=timeout)
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    response = opener.open(
        urllib.request.Request(url, headers=headers), timeout=timeout)
    content_type = str(response.headers.get("Content-Type") or "").lower()
    if "text/html" not in content_type:
        return response
    try:
        document = response.read(1024 * 1024).decode("utf-8", "replace")
    finally:
        response.close()
    confirmed = _drive_confirmation_url(document)
    result = opener.open(
        urllib.request.Request(confirmed, headers=headers), timeout=timeout)
    confirmed_type = str(result.headers.get("Content-Type") or "").lower()
    if "text/html" not in confirmed_type:
        return result
    try:
        error_page = result.read(1024 * 1024).decode("utf-8", "replace")
    finally:
        result.close()
    if "Quota exceeded" in error_page or "Too many users" in error_page:
        raise IOError(
            "Google Drive download quota exceeded; use a mirror or try later")
    raise IOError(
        "Google Drive returned an HTML page instead of the model file")


def _download_remote_pack(pack: dict, source: str,
                          custom_base: Optional[str],
                          progress: Optional[ProgressCallback]) -> None:
    root = Path(get_model_root())
    download_dir = root / ".downloads" / pack["id"]
    download_dir.mkdir(parents=True, exist_ok=True)
    pack_total = int(pack.get("remote_download_size", 0))
    completed = 0
    for relative in pack["files"]:
        target = root.joinpath(*PurePosixPath(relative).parts)
        if target.is_file():
            completed += target.stat().st_size
            continue
        filename = _remote_part_filename(
            pack["downloads"][relative], relative)
        part = download_dir / (filename + ".part")
        expected_hash = str(
            pack.get("remote_hashes", {}).get(relative, "") or
            _metadata(pack["id"]).get("files", {}).get(relative, ""))
        # A previous run may have completed the bytes but been interrupted
        # before the atomic move. Avoid another network request in that case.
        if part.is_file() and expected_hash and _sha256(part) == expected_hash:
            part_size = part.stat().st_size
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(part), str(target))
            completed += part_size
            if progress:
                progress("download", completed, pack_total or completed)
            continue

        existing = part.stat().st_size if part.is_file() else 0
        headers = {
            "User-Agent": "LightVideoEnhancer/1",
            "Accept-Encoding": "identity",
        }
        if existing:
            headers["Range"] = "bytes=%d-" % existing
        request = urllib.request.Request(
            _remote_file_url(pack, relative, source, custom_base),
            headers=headers)
        with _open_remote(request.full_url, headers, timeout=90) as response:
            status = int(getattr(response, "status", 200) or 200)
            content_range = str(response.headers.get("Content-Range") or "")
            match = re.fullmatch(
                r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range.strip())
            partial = (
                status == 206 and match is not None and
                int(match.group(1)) == existing)
            if existing and not partial:
                # A source that ignores Range sends the entire object. Replace
                # the partial file instead of appending duplicate bytes.
                existing = 0
            mode = "ab" if existing and partial else "wb"
            remaining = int(response.headers.get("Content-Length") or 0)
            declared_total = (
                int(match.group(3)) if match and match.group(3) != "*" else 0)
            total = declared_total or (
                existing + remaining if remaining else 0)
            current = existing
            with part.open(mode) as output:
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    current += len(chunk)
                    if progress:
                        file_current = min(current, total) if total else current
                        progress(
                            "download", completed + file_current,
                            pack_total or (completed + total if total else 0))
                # Some mirrors return a valid Content-Range but stream bytes
                # past its declared object length. Keep only the declared
                # object; the cryptographic hash below still decides validity.
                if declared_total and current > declared_total:
                    output.truncate(declared_total)
                    current = declared_total
            if declared_total and current != declared_total:
                raise IOError(
                    "Incomplete ranged model download: %d/%d bytes" %
                    (current, declared_total))
        target.parent.mkdir(parents=True, exist_ok=True)
        if expected_hash and _sha256(part) != expected_hash:
            raise ValueError(
                "Model file checksum mismatch / 模型文件校验失败: " +
                relative)
        part_size = part.stat().st_size
        os.replace(str(part), str(target))
        completed += part_size
        if progress:
            progress("download", completed, pack_total or completed)


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

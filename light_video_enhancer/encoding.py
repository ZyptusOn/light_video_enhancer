"""Encoder names, aliases and format-preserving fallback order."""

from typing import List


CODEC_CHOICES = (
    "auto",
    "h264_nvenc", "h264_amf", "h264_mf", "libx264",
    "hevc_nvenc", "hevc_amf", "hevc_mf", "libx265",
    "av1_nvenc", "av1_amf", "libsvtav1", "libaom-av1",
    "mpeg4",
)

CODEC_ALIASES = {
    "x264": "libx264",
    "h264": "libx264",
    "x265": "libx265",
    "h265": "libx265",
    "av1": "libsvtav1",
    "aom": "libaom-av1",
    "svt-av1": "libsvtav1",
}

CLI_CODEC_CHOICES = CODEC_CHOICES + tuple(CODEC_ALIASES)

_FAMILY_ENCODERS = {
    "h264": ("h264_nvenc", "h264_amf", "h264_mf", "libx264"),
    "hevc": ("hevc_nvenc", "hevc_amf", "hevc_mf", "libx265"),
    "av1": ("av1_nvenc", "av1_amf", "libsvtav1", "libaom-av1"),
}


def canonical_codec(codec: str) -> str:
    """Return the FFmpeg encoder name for a user-facing name or alias."""
    value = (codec or "auto").strip().lower()
    return CODEC_ALIASES.get(value, value)


def codec_family(codec: str) -> str:
    value = canonical_codec(codec)
    for family, encoders in _FAMILY_ENCODERS.items():
        if value in encoders:
            return family
    return "mpeg4" if value == "mpeg4" else "unknown"


def codec_candidates(requested: str) -> List[str]:
    """Prefer the requested format, then degrade AV1 -> HEVC -> H.264."""
    codec = canonical_codec(requested)
    if codec == "auto":
        codec = "h264_mf"
    family = codec_family(codec)
    families = {
        "av1": ("av1", "hevc", "h264"),
        "hevc": ("hevc", "h264"),
        "h264": ("h264",),
    }.get(family, ())
    result = [codec]
    for name in families:
        for candidate in _FAMILY_ENCODERS[name]:
            if candidate not in result:
                result.append(candidate)
    if "mpeg4" not in result:
        result.append("mpeg4")
    return result

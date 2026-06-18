from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EncodeConfig:
    codec: str = "h264_nvenc"
    preset: str = "p7"
    crf: int = 23
    pixel_format: str = "yuv420p"
    container: str = "mp4"


@dataclass
class ProcessConfig:
    input_path: str = ""
    output_path: str = ""
    width: int = 0
    height: int = 0
    scale: float = 2.0
    sr_engine: str = "nvvfx"
    fi_engine: str = "optical_flow"
    fi_multiplier: int = 2
    fi_quality: str = "balanced"
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    fps: Optional[float] = None
    start_time: Optional[float] = None
    duration: Optional[float] = None
    device: str = "cuda"
    torch_python: Optional[str] = None
    sr_first: bool = False

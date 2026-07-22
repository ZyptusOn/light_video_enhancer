from dataclasses import dataclass, field
from typing import Optional


QUALITY_CHOICES = ("ultra", "fast", "balanced", "quality")


@dataclass
class EncodeConfig:
    codec: str = "auto"
    preset: str = "balanced"
    crf: int = 23
    pixel_format: str = "yuv420p"
    container: str = "mp4"
    copy_audio: bool = True
    overwrite: bool = False


@dataclass
class ProcessConfig:
    input_path: str = ""
    output_path: str = ""
    width: int = 0
    height: int = 0
    scale: float = 2.0
    sr_engine: str = "auto"
    fi_engine: str = "auto"
    sr_quality: str = "quality"
    fi_multiplier: int = 2
    fi_quality: str = "balanced"
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    fps: Optional[float] = None
    start_time: Optional[float] = None
    duration: Optional[float] = None
    device: str = "auto"
    torch_python: Optional[str] = None
    sr_first: bool = False
    ncnn_gpu: Optional[int] = None
    keep_partial: bool = False

    def validate(self) -> None:
        if not self.input_path:
            raise ValueError("未指定输入文件")
        if not self.output_path:
            raise ValueError("未指定输出文件")
        if self.scale <= 0:
            raise ValueError("超分倍率必须大于 0")
        if self.width < 0 or self.height < 0:
            raise ValueError("输出宽高不能为负数")
        if not 1 <= self.fi_multiplier <= 8:
            raise ValueError("插帧倍率必须在 1 到 8 之间")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("输出帧率必须大于 0")
        if self.start_time is not None and self.start_time < 0:
            raise ValueError("起始时间不能为负数")
        if self.duration is not None and self.duration <= 0:
            raise ValueError("持续时长必须大于 0")
        if not 0 <= self.encode.crf <= 63:
            raise ValueError("质量值必须在 0 到 63 之间")
        if self.sr_quality not in QUALITY_CHOICES:
            raise ValueError("未知的超分质量: %s" % self.sr_quality)
        if self.fi_quality not in QUALITY_CHOICES:
            raise ValueError("未知的插帧质量: %s" % self.fi_quality)

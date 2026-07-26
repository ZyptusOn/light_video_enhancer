"""Public model contracts shared by NCNN engines and native executors."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NcnnInterpolationStage:
    model_dir: str
    kind: str = "rife"
    tta: bool = False
    uhd: bool = False


@dataclass(frozen=True)
class NcnnSuperResolutionStage:
    kind: str
    param_path: str
    model_path: str
    scale: int
    tta: bool = False
    noise: int = -1
    syncgap: int = 3
    tile: int = 0

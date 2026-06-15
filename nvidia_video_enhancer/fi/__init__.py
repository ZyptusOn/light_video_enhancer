from .base import FrameInterpolationEngine
from .blend import BlendFIEngine
from .optical_flow import OpticalFlowEngine


def create_fi_engine(engine_name: str, device: str = "cuda",
                     quality: str = "balanced") -> FrameInterpolationEngine:
    if engine_name == "rife":
        from .rife import RIFEEngine
        return RIFEEngine(device=device)
    elif engine_name == "optical_flow":
        return OpticalFlowEngine(quality=quality)
    elif engine_name == "dis":
        from .dis_flow import DISFlowEngine
        return DISFlowEngine(quality=quality)
    elif engine_name == "torch_flow":
        from .torch_flow import TorchFlowEngine
        return TorchFlowEngine(quality=quality)
    elif engine_name == "blend":
        return BlendFIEngine(device=device)
    else:
        raise ValueError(f"未知的插帧引擎: {engine_name}")


__all__ = [
    "FrameInterpolationEngine",
    "BlendFIEngine",
    "OpticalFlowEngine",
    "create_fi_engine",
]

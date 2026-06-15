from .base import SuperResolutionEngine
from .dxva_vsr import DXVA_VSR_Engine
from .fallback import ESRGANEngine, BicubicEngine, LanczosEngine


def create_sr_engine(engine_name: str, device: str = "cuda") -> SuperResolutionEngine:
    if engine_name == "dxva_vsr":
        return DXVA_VSR_Engine()
    elif engine_name == "nvvfx":
        from .nvvfx_sr import NVVFX_SR_Engine
        return NVVFX_SR_Engine(quality="HIGH")
    elif engine_name == "esrgan":
        return ESRGANEngine(device=device)
    elif engine_name == "bicubic":
        return BicubicEngine()
    elif engine_name == "lanczos":
        return LanczosEngine()
    else:
        raise ValueError(f"未知的超分引擎: {engine_name}")


__all__ = [
    "SuperResolutionEngine",
    "DXVA_VSR_Engine",
    "ESRGANEngine",
    "BicubicEngine",
    "LanczosEngine",
    "create_sr_engine",
]

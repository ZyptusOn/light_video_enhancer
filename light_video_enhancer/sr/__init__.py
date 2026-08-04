from typing import Optional

from .base import SuperResolutionEngine
from .fallback import BicubicEngine, LanczosEngine


def create_sr_engine(engine_name: str, device: str = "auto",
                     torch_python: Optional[str] = None,
                     ncnn_gpu: Optional[int] = None,
                     quality: str = "quality",
                     spark_reference_path: Optional[str] = None,
                     spark_reference_indices=None,
                     spark_reference_guidance: float = 1.0) -> SuperResolutionEngine:
    if engine_name == "dxva_vsr":
        from .dxva_vsr import DXVA_VSR_Engine
        return DXVA_VSR_Engine()
    if engine_name == "nvvfx":
        from .nvvfx_sr import NVVFX_SR_Engine
        return NVVFX_SR_Engine(quality=quality, torch_python=torch_python)
    if engine_name == "realcugan":
        from .realcugan_ncnn import RealCUGANEngine
        return RealCUGANEngine(device=device, gpu_id=ncnn_gpu, quality=quality)
    if engine_name == "span":
        from .span_ncnn import SPANNcnnEngine
        return SPANNcnnEngine(device=device, gpu_id=ncnn_gpu, quality=quality)
    if engine_name == "seedvr2":
        from .seedvr2 import SeedVR2Engine
        return SeedVR2Engine(device=device, torch_python=torch_python, quality=quality)
    if engine_name == "dloral":
        from .dloral import DLoRALEngine
        return DLoRALEngine(device=device, torch_python=torch_python, quality=quality)
    if engine_name == "osdenhancer":
        from .osdenhancer import OSDEnhancerEngine
        return OSDEnhancerEngine(
            device=device, torch_python=torch_python, quality=quality)
    if engine_name == "sparkvsr":
        from .sparkvsr import SparkVSREngine
        return SparkVSREngine(
            device=device, torch_python=torch_python, quality=quality,
            reference_path=spark_reference_path,
            reference_indices=spark_reference_indices,
            reference_guidance=spark_reference_guidance)
    if engine_name == "flashvsr":
        from .flashvsr import FlashVSREngine
        return FlashVSREngine(device=device, torch_python=torch_python, quality=quality)
    if engine_name in {"realesrgan", "esrgan"}:
        from .realesrgan_ncnn import ESRGANEngine, RealESRGANEngine
        engine_class = ESRGANEngine if engine_name == "esrgan" else RealESRGANEngine
        return engine_class(device=device, gpu_id=ncnn_gpu, quality=quality)
    if engine_name == "bicubic":
        return BicubicEngine()
    if engine_name == "lanczos":
        return LanczosEngine()
    raise ValueError("未知的超分引擎: %s" % engine_name)


__all__ = ["SuperResolutionEngine", "BicubicEngine", "LanczosEngine",
           "create_sr_engine"]

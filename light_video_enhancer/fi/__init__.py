from typing import Optional

from .base import FrameInterpolationEngine


def create_fi_engine(engine_name: str, device: str = "auto",
                     quality: str = "balanced",
                     torch_python: Optional[str] = None,
                     ncnn_gpu: Optional[int] = None) -> FrameInterpolationEngine:
    if engine_name == "rife":
        from .rife import RIFEEngine
        return RIFEEngine(device=device, torch_python=torch_python)
    if engine_name == "ema_vfi":
        from .ema_vfi import EMAVFIEngine
        return EMAVFIEngine(device=device, quality=quality, torch_python=torch_python)
    if engine_name == "rife_ncnn":
        from .rife_ncnn import RIFENcnnEngine
        return RIFENcnnEngine(quality=quality, gpu_id=ncnn_gpu)
    if engine_name == "ifrnet_ncnn":
        from .ifrnet_ncnn import IFRNetNcnnEngine
        return IFRNetNcnnEngine(quality=quality, gpu_id=ncnn_gpu)
    if engine_name == "dis":
        from .dis_flow import DISFlowEngine
        return DISFlowEngine(quality=quality)
    if engine_name == "optical_flow":
        from .optical_flow import OpticalFlowEngine
        return OpticalFlowEngine(quality=quality)
    if engine_name == "torch_flow":
        from .torch_flow import TorchFlowEngine
        return TorchFlowEngine(quality=quality)
    if engine_name == "blend":
        from .blend import BlendFIEngine
        return BlendFIEngine(device=device, quality=quality)
    raise ValueError("未知的插帧引擎: %s" % engine_name)


__all__ = ["FrameInterpolationEngine", "create_fi_engine"]

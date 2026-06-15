import os
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FrameInterpolationEngine


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=True),
        nn.PReLU(out_planes)
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super().__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class IFBlock(nn.Module):
    def __init__(self, in_planes, scale=1, c=64):
        super().__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear",
                              align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear",
                                 align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=self.scale / 2, mode="bilinear",
                            align_corners=False)
        flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        return flow, mask


class RIFE_IFI(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(6, scale=4, c=192)
        self.block1 = IFBlock(10, scale=2, c=128)
        self.block2 = IFBlock(10, scale=1, c=64)

    def forward(self, x, timestep=0.5):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.full((x.size(0), 1, 1, 1), timestep,
                                  dtype=torch.float32, device=x.device)
        img0 = x[:, :3]
        img1 = x[:, 3:6]

        x_cat = torch.cat((img0, img1), 1)
        blk_scales = [4, 2, 1]
        flow = None
        mask = None
        merged = None

        for i in range(3):
            flow, mask = self._run_block(i, x_cat, flow, scale=blk_scales[i])
            warped_img0 = self._warp(img0, flow[:, :2], timestep)
            warped_img1 = self._warp(img1, flow[:, 2:4], 1 - timestep)
            merged = warped_img0 * mask + warped_img1 * (1 - mask)

        return merged

    def _run_block(self, idx, x, flow, scale):
        if idx == 0:
            return self.block0(x, flow, scale=scale)
        elif idx == 1:
            return self.block1(x, flow, scale=scale)
        else:
            return self.block2(x, flow, scale=scale)

    @staticmethod
    def _warp(img, flow, timestep):
        B, C, H, W = img.shape
        flow = flow * timestep
        xx = torch.arange(0, W, device=img.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=img.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W) + flow[:, 0:1, :, :]
        yy = yy.view(1, 1, H, W) + flow[:, 1:2, :, :]
        xx = 2.0 * xx / max(W - 1, 1) - 1.0
        yy = 2.0 * yy / max(H - 1, 1) - 1.0
        grid = torch.cat((xx, yy), 1).permute(0, 2, 3, 1)
        return F.grid_sample(img, grid, align_corners=True,
                             padding_mode='border')


class RIFEEngine(FrameInterpolationEngine):
    """
    基于 RIFE (Real-time Intermediate Flow Estimation) 的 AI 插帧引擎。
    使用 PyTorch 实现，支持 CUDA 加速。

    工作原理：
    1. 输入相邻两帧 frame0 和 frame1
    2. RIFE 估算两帧之间的光流
    3. 根据时间步长 (timestep) 合成中间帧
    4. 对于 2x 插帧，生成 timestep=0.5 的中间帧
    """

    def __init__(self, device: str = "cuda"):
        self._device = device if torch.cuda.is_available() else "cpu"
        self._model: RIFE_IFI = None
        self._multiplier = 2
        self._width = 0
        self._height = 0
        self._pad_w = 0
        self._pad_h = 0
        self._model_path = ""

    @property
    def name(self) -> str:
        return "RIFE (PyTorch IFI)"

    def _pad_to_multiple(self, w: int, h: int, base: int = 64):
        pw = ((w + base - 1) // base) * base - w
        ph = ((h + base - 1) // base) * base - h
        return pw, ph

    def initialize(self, width: int, height: int,
                   multiplier: int = 2) -> None:
        self._multiplier = multiplier
        self._width = width
        self._height = height
        self._pad_w, self._pad_h = self._pad_to_multiple(width, height, 32)

        self._model = RIFE_IFI().to(self._device).eval()
        self._load_weights()

    def _load_weights(self):
        weight_paths = [
            "rife_v4.25.pth",
            os.path.join(os.path.dirname(__file__), "rife_v4.25.pth"),
        ]
        for p in weight_paths:
            if os.path.exists(p):
                state = torch.load(p, map_location=self._device)
                self._model.load_state_dict(state, strict=False)
                return

    def interpolate(self, frame0: np.ndarray,
                    frame1: np.ndarray) -> List[np.ndarray]:
        if self._model is None:
            raise RuntimeError("RIFE 引擎未初始化")

        t0 = self._preprocess(frame0)
        t1 = self._preprocess(frame1)
        x = torch.cat((t0, t1), 1)
        del t0, t1

        with torch.no_grad():
            results = []
            for i in range(1, self._multiplier):
                t = i / self._multiplier
                pred = self._model(x, t)
                pred = pred[:, :, :self._height, :self._width]
                out = self._postprocess(pred)
                results.append(out)
                del pred

        del x
        torch.cuda.empty_cache()

        if self._multiplier == 2:
            return results

        return results

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        import cv2
        if self._pad_h > 0 or self._pad_w > 0:
            frame = cv2.copyMakeBorder(frame, 0, self._pad_h, 0, self._pad_w,
                                       cv2.BORDER_REPLICATE)
        t = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
        t = t / 255.0
        return t.to(self._device)

    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        t = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        t = (t * 255.0).clip(0, 255).astype(np.uint8)
        return t

    def release(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

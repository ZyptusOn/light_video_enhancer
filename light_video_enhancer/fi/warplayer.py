"""
Practical-RIFE warplayer — 后向光流 warp 实现。

来自 Practical-RIFE model/warplayer.py，被 IFNet_HDv3 引用。
"""

from collections import OrderedDict

import torch


_MAX_GRID_CACHE = 8
backwarp_tenGrid = OrderedDict()


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(
            -1.0, 1.0, tenFlow.shape[3],
            device=tenFlow.device, dtype=tenFlow.dtype,
        ).view(1, 1, 1, tenFlow.shape[3]).expand(
            tenFlow.shape[0], -1, tenFlow.shape[2], -1
        )
        tenVertical = torch.linspace(
            -1.0, 1.0, tenFlow.shape[2],
            device=tenFlow.device, dtype=tenFlow.dtype,
        ).view(1, 1, tenFlow.shape[2], 1).expand(
            tenFlow.shape[0], -1, -1, tenFlow.shape[3]
        )
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1)
        while len(backwarp_tenGrid) > _MAX_GRID_CACHE:
            backwarp_tenGrid.popitem(last=False)
    else:
        backwarp_tenGrid.move_to_end(k)

    normalised_flow = torch.cat([
        tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
        tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
    ], 1)

    grid = backwarp_tenGrid[k]
    if grid.dtype != tenFlow.dtype:
        grid = grid.to(dtype=tenFlow.dtype)
    g = (grid + normalised_flow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput, grid=g, mode='bilinear',
        padding_mode='border', align_corners=True
    )

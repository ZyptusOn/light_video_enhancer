"""
RIFE v4.25 (RIFE4.25) 模型定义。

架构来源: https://github.com/hzwer/Practical-RIFE (RIFE4.25 tag)
对应权重: train_log/flownet.pkl

结构: FlownetCas (5-block cascade + teacher + caltime)
  - Flownet: 2x 下采样 → ResConv x8 → PixelShuffle 上采样
  - FlownetCas: 5 级级联 + Head 特征编码 + 教师蒸馏
  - 训练时 teacher/caltime 参与，推理时仅使用级联光流估计

被 fi/rife.py 和 fi/_rife_infer.py 共享使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .warplayer import warp
except ImportError:
    from warplayer import warp


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True)
    )


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True)
    )


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class Flownet(nn.Module):
    def __init__(self, in_planes, c=64):
        super(Flownet, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear",
                          align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear",
                                 align_corners=False) * (1. / scale)
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear",
                            align_corners=False)
        flow_out = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        conf = tmp[:, 5:]
        return flow_out, mask, conf


class FlownetCas(nn.Module):
    def __init__(self):
        super(FlownetCas, self).__init__()
        self.block0 = Flownet(7 + 8, c=192)
        self.block1 = Flownet(8 + 4 + 8 + 8, c=128)
        self.block2 = Flownet(8 + 4 + 8 + 8, c=96)
        self.block3 = Flownet(8 + 4 + 8 + 8, c=64)
        self.block4 = Flownet(8 + 4 + 8 + 8, c=32)
        self.encode = Head()
        self.teacher = Flownet(8 + 4 + 8 + 3 + 8, c=64)
        self.caltime = nn.Sequential(
            nn.Conv2d(8 + 9, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, timestep=0.5, scale=None, training=False, distill=True):
        if scale is None:
            scale = [8, 4, 2, 1]

        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] if x.shape[1] > 6 else None

        f0 = self.encode(img0)
        f1 = self.encode(img1)

        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        stu = [self.block0, self.block1, self.block2, self.block3, self.block4]

        for i in range(5):
            if flow is not None:
                flow_d, mask, feat = stu[i](
                    torch.cat((warped_img0, warped_img1, warped_f0, warped_f1,
                               timestep, mask, feat), 1),
                    flow, scale=scale[i])
                flow = flow + flow_d
            else:
                flow, mask, feat = stu[i](
                    torch.cat((img0, img1, f0, f1, timestep), 1),
                    None, scale=scale[i])
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            warped_f0 = warp(f0, flow[:, :2])
            warped_f1 = warp(f1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))

        for i in range(5):
            mask_list[i] = torch.sigmoid(mask_list[i])
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])

        return flow_list, mask_list[4], merged

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16.0 / scale, 8.0 / scale, 4.0 / scale,
                       2.0 / scale, 1.0 / scale]
        _, _, merged = self.forward(imgs, timestep, scale_list, training=False)
        return merged[-1]

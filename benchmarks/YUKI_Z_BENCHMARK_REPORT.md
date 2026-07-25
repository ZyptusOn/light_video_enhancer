# YUKI_Z 视频管线性能测试

测试日期：2026-07-25

## 测试对象与方法

- 输入：`YUKI_Z.mp4`，1280×720，29.999 fps，共 483 帧（约 16.1 秒）。
- 统一输出：2× 分辨率、2× 帧率、HEVC NVENC `p5`、CQ 23、无音频。
- 硬件：NVIDIA GeForce RTX 5070 Ti Laptop GPU，12 GB，驱动 610.62。
- 外部环境：Python 3.13.12、PyTorch 2.11.0+cu128、CUDA 与 NV-VFX 可用。
- 每条管线由与 WinUI 相同的冻结后端启动；记录墙钟时间、初始化时间、输入/输出吞吐、系统 CPU、NVIDIA GPU、显存和功耗。
- 快速矩阵使用 4.0–8.0 秒片段；最终基线额外处理完整 16.1 秒视频。

复现工具：[`benchmark_video_pipelines.py`](benchmark_video_pipelines.py)。

## 完整视频基线

| 版本 | 墙钟时间 | 处理阶段输入 fps | 输出 fps | GPU 平均/峰值 | 输出 |
|---|---:|---:|---:|---:|---|
| v0.5.2 发布版 | 25.02 s | 24.42 | 48.78 | 40.9/70% | 2560×1440、59.999 fps、965 帧 |
| 最终优化版 | 21.24 s | 31.37 | 62.68 | 56.8/79% | 2560×1440、59.999 fps、965 帧 |

最终版的处理阶段吞吐提高 28.5%，整段墙钟时间减少 15.1%，GPU 平均利用率从 40.9% 提高到 56.8%。

## 4 秒热状态 A/B

| 管线 | v0.5.2 输入 fps | 优化版输入 fps | 提升 |
|---|---:|---:|---:|
| RIFE PyTorch + NVIDIA Video Effects VSR | 11.40 | 21.88 | 1.92× |
| RIFE PyTorch only | 12.76 | 30.80 | 2.41× |
| RIFE PyTorch + D3D11 driver VSR | 5.93 | 8.32 | 1.40× |
| NVIDIA Video Effects VSR only | 35.57 | 约 33–36 | 无稳定变化 |

短片最受首次卷积算法搜索影响，所以收益大于完整视频；长视频中该固定成本会逐渐摊薄。

## 其他管线矩阵

以下数据来自相同 4 秒片段，均正确输出 2560×1440、59.999 fps、239 帧：

| 管线 | 输入 fps | GPU 平均/峰值 | CPU 平均/峰值 | 结论 |
|---|---:|---:|---:|---|
| RIFE PyTorch + Real-CUGAN | 1.15 | 29/100% | 44.7/89.3% | PNG/进程阶段切换严重 |
| RIFE PyTorch + Real-ESRGAN AnimeVideo-v3 | 1.57 | 23.9/100% | 25.6/58.4% | 明显慢于 NV-VFX |
| RIFE NCNN + Real-CUGAN | 1.47 | 33/100% | 19.2/37.4% | 批处理后可用，但吞吐仍低 |
| RIFE NCNN + Real-ESRGAN AnimeVideo-v3 | 1.61 | 25.3/100% | 42/75.1% | 此组 NCNN 中最快 |

这些 NCNN/ESRGAN 管线的 GPU 锯齿不是监控错误，而是“CPU 写 PNG → NCNN GPU 推理 → CPU 读 PNG → 编码”的阶段性交替。峰值能到 100%，但平均利用率低，主要瓶颈是文件中转和多个进程之间无法并行。

## A/B 优化过程

1. 移除 NV-VFX DLPack 输出的整帧 GPU clone：没有稳定可复现的收益，且缓冲区所有权风险高，已回退。
2. 将两个相邻 RIFE 帧对合并为 batch=2：显存峰值从约 1.5 GB 增至约 2.1 GB；配合 `cudnn.benchmark=True` 时还会为 batch=1/2 分别搜索算法，4 秒测试降至 7.11 输入 fps，已回退。
3. 关闭 RIFE 的 `torch.backends.cudnn.benchmark`：消除了每种 batch 形状首次出现时约数秒的算法搜索；稳态卷积速度没有观察到回退。这是最终保留的优化。
4. 同一设置已同步到融合 Worker、普通外部 Python RIFE Worker和进程内 RIFE，避免不同前端或运行方式表现不一致。

## 结论与后续方向

- 推荐默认管线仍是 RIFE PyTorch + NVIDIA Video Effects VSR。它在本机同时具有最高综合质量和远高于 NCNN/ESRGAN 组合的吞吐。
- RIFE + D3D11 driver VSR 可作为无 NV-VFX 环境的 NVIDIA 备用路径，但当前 D3D11 逐帧桥接仍明显慢于融合 CUDA 路径。
- NCNN 已从“几乎不可运行”改善到能完成任务，但当前文件型接口决定了其 GPU 锯齿和低平均利用率。下一阶段真正有效的工作应是内存/管道帧接口或常驻 NCNN Worker，而不是继续增大目录 batch。
- `nvidia-smi` 的 GPU 百分比采样较粗；最终判断同时依据处理吞吐、首帧日志时间、显存、功耗和完整输出校验。

## 原始结果

- [`YUKI_Z_full_final`](results/YUKI_Z_full_final/RESULTS.md)
- [`YUKI_Z_full_v052_control`](results/YUKI_Z_full_v052_control/RESULTS.md)
- [`YUKI_Z_final_candidate`](results/YUKI_Z_final_candidate/RESULTS.md)
- [`YUKI_Z_v052_warm_control`](results/YUKI_Z_v052_warm_control/RESULTS.md)
- [`YUKI_Z_v052_warm_components_control`](results/YUKI_Z_v052_warm_components_control/RESULTS.md)
- [`YUKI_Z_final_dxva_control`](results/YUKI_Z_final_dxva_control/RESULTS.md)
- [`YUKI_Z_v052_warm_dxva_control`](results/YUKI_Z_v052_warm_dxva_control/RESULTS.md)
- [`YUKI_Z_v052_comparison`](results/YUKI_Z_v052_comparison/RESULTS.md)

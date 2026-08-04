# NVE 性能优化备忘

## 1. NV12 对齐拷贝：Python 循环 → numpy 切片

**位置**: `sr/dxva_vsr.py` 的 `_bgr_to_nv12` 函数

**问题**: 当 `align_w != w` 时（如奇数宽度对齐到偶数），Y 平面和 UV 平面用 Python `for` 循环逐行拷贝。1080p 下每帧多出上千次 Python 循环，4K 更严重。

**修复**: 用 numpy reshape + 切片批量拷贝替代逐行循环——

```python
# 旧: for row in range(h): aligned[row * align_w: ...] = ...
# 新: y_plane = y.reshape(h, w); aligned_y[:h, :w] = y_plane
```

## 2. BGR→YUV 转换应在编码线程完成

**位置**: `pipeline.py` 的 `_process_with_fi` 方法

**问题**: 主线程在完成 SR 处理后立即调用 `cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)` 做 BGR→YUV 转换，然后把 YUV 数据放入编码队列。这导致 GPU 在等待 CPU 做色彩空间转换时闲置。

**修复**: 主线程直接把 BGR 帧放入编码队列，转换移到 `_enc_worker` 线程内部完成。这样主线程能更快回到下一个 SR 任务，GPU 利用率更高。

## 3. ffmpeg / ffprobe 路径查找应加缓存

**位置**: `pipeline.py` 的 `_find_ffmpeg` / `_find_ffprobe` 函数

**问题**: 每次调用都执行 subprocess 检测 `ffmpeg -version`，或对文件系统做多次 `os.path.isfile` 检查。管线中 `_probe_input` 和 `_decode_frames_subprocess` 等多处调用，重复开销累积。

**修复**: 添加模块级缓存 `_ffmpeg_cache` / `_ffprobe_cache`，首次查找后缓存结果。

## 4. RIFE warp 坐标缓存无限增长

**位置**: `fi/rife.py` 的 `_WARP_CACHE` 字典

**问题**: 缓存 key 是 `(device, tensor_size)` 元组。不同分辨率的帧（或不同 batch size，虽然当前固定为1）会产生新的缓存条目，且永不清除，导致 GPU 显存泄漏。

**修复**: 添加 `_WARP_CACHE_MAX = 8` 限制，超限时清空重建（`_WARP_CACHE.clear()`）。

## 5. RIFE subprocess 模式缺少 SSIM 静态帧跳过

**位置**: `fi/rife.py` 的 `_interpolate_subprocess` 方法

**问题**: in-process 模式通过 `_calc_ssim` 跳过 SSIM > 0.996 的静态帧（直接复制），但 subprocess 模式完全没有这个优化。对于有长时间静止画面的视频，subprocess 模式会做大量无用推理和 pickle 序列化/反序列化开销。

**修复**: subprocess 端添加 numpy 版 SSIM 快速计算（降采样到 32×32），跳过静态帧。


## 总结

| 优化 | 影响范围 | 预期收益 |
|------|----------|----------|
| NV12 numpy 切片 | DXVA VSR 引擎每帧 | 4K 下显著减少 CPU 时间 |
| BGR→YUV 移入编码线程 | FI+SR 模式主循环 | GPU 利用率提升，吞吐提升 |
| ffmpeg 路径缓存 | 启动/探针阶段 | 减少不必要的子进程启动 |
| RIFE warp 缓存限制 | RIFE 引擎长期运行 | 避免 GPU 显存持续增长 |
| subprocess SSIM 跳过 | RIFE subprocess 模式 | 静止场景大幅减少 IPC 开销 |

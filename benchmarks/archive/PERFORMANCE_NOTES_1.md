# 性能优化思路备忘

> 生成时间: 2026-06-17
> 项目: nvidia_video_enhancer (light_video_enhancer)

---

## 1. 流水线并行化

### 1.1 纯超分路径：主线程 + 编码线程分离

- **问题**：GPU 做 VSR 超分后，主线程还要做 BGR→YUV + NVENC 编码，GPU 空闲等待
- **方案**：`pipeline.py` `_process_sr_only()` 把编码移到独立线程
  ```
  主线程: [decoder] → [do_sr(GPU)] → enc_q.put(bgr)
  编码线程:                    enc_q.get() → BGR→YUV(CPU) → encoder.encode_yuv()
  ```
- **效果**：GPU 不等 CPU 编码，GPU 利用率提升约 20%

### 1.2 FI+SR+Enc 三线程并行

- **问题**：FI（光流/RIFE）是 CPU/GPU 密集操作，串行时 GPU 大量空闲
- **方案**：`pipeline.py` `_process_with_fi()` 三重流水线
  ```
  FI 线程(CPU/GPU):  f0,f1 → do_fi → fi_out_q   // 处理下一对帧
  主线程(GPU):       raw0 → do_sr → enc_q.put    // 永不等 FI，GPU 持续跑
  编码线程:           enc_q.get → YUV → encode   // 独立编码
  ```
- **关键设计**：主线程"永不等待"FI 结果——使用上一轮的 FI 输出覆盖当前轮，GPU 无空闲周期

### 1.3 编码线程错误传递与防死锁

- **问题**：`enc_q.put(sr_bgr)` 是阻塞调用，如果编码线程已崩溃，主线程永久阻塞
- **方案**：改用 `enc_q.put(sr_bgr, timeout=1.0)` + `_check_enc()` 循环
  - 每 1 秒超时检查编码线程是否异常
  - 避免死锁，异常能及时向上传播

---

## 2. GPU 推理加速

### 2.1 RIFE FP16 推理

- **问题**：RIFE v4.x 网络 FP32 推理占 ~4GB 显存，RTX 卡 Tensor Core 未利用
- **方案**：`model.half()` + 输入 `.half()` 显式转换
  - 显存减半（4GB → 2GB）
  - RTX 30/40/50 系列 Tensor Core 推速翻倍
  - **不修改** `torch.set_default_tensor_type`（全局副作用，影响其他引擎）

### 2.2 RIFE SSIM 静态帧检测

- **问题**：静止画面（如谈话节目、PPT 演示）仍触发完整网络推理，浪费计算
- **方案**：32×32 缩略图 SSIM 快速判定
  - SSIM > 0.996 → 直接复制前一帧，跳过网络
  - SSIM < 0.2 → 场景切换，复制前一帧（避免跨场景伪影）
  - 仅在 SSIM 正常范围才跑完整 RIFE 推理
- **效果**：静态内容视频处理速度可提升 50-90%

### 2.3 RIFE Padding 对齐

- **问题**：v4.x 要求 128 像素对齐，输入尺寸不对齐导致边缘伪影或错误
- **方案**：`_calc_pad(dim, 128)` 计算填充量，`F.pad()` 零填充，推理后裁剪回原尺寸

### 2.4 BlendFI 光流降分辨率

- **问题**：`blend.py` 在全分辨率计算两遍 Farneback 光流（前向+后向），1080p 下极慢
- **方案**：仿照 `optical_flow.py` 的 `scale_div` 设计
  - `quality=balanced` → 1/4 分辨率计算光流 → upscale
  - `quality=ultra` → 1/8 分辨率
  - 全分辨率只做 `cv2.remap` warp（硬件双线性采样，极快）

---

## 3. 内存与数据传输

### 3.1 BGR→YUV420 转换在编码线程

- **问题**：BGR→YUV420 是 CPU 密集操作，在主线程做则 GPU 空闲
- **方案**：编码线程内部用 `cv2.cvtColor(bgr, COLOR_BGR2YUV_I420).ravel()` 转换
  - OpenCV SIMD 优化
  - 与 GPU 超分完全并行

### 3.2 NV12 缓冲区对齐

- **问题**：`_bgr_to_nv12` 用原始尺寸计算 NV12 布局，但 D3D11 Bridge 期望 `alignSrcW×alignSrcH` 对齐尺寸
  - 奇数宽高时 Y/UV 平面偏移错位 → 越界读取 / 花屏
- **方案**：`_bgr_to_nv12` 新增 `align_w/align_h` 参数
  - 非对齐尺寸：构造 `align_w*align_h*3/2` 零初始化缓冲区，逐行复制有效数据
  - 对齐尺寸：零开销，无额外操作

---

## 4. 编码器参数调优

### 4.1 NVENC 参数

- `tune=ll`（low latency）→ 减少编码延迟
- `async_depth=4` → NVENC 硬件流水线深度
- `no-scenecut=1` → 禁用场景检测（FI 已处理帧间连续性，scene cut 反而降低质量）
- `rc=vbr` + `cq` → 可变码率质量控制

### 4.2 FPS 精度（AVRational）

- **问题**：`(int)59.94 = 59` 截断 → 输出视频时长偏差 + 音画不同步
- **方案**：`av_d2q(fps, 1000000)` → `{60000, 1001}` 精确有理数
  - `time_base = av_inv_q(fr)` → 1/60000
  - `framerate = {60000, 1001}` 精确保持

---

## 5. RIFE 架构升级

### 5.1 v4.x 网络 vs 旧自生成网络

| 对比项 | 旧版（自生成） | 新版（v4.x） |
|---|---|---|
| Block 数 | 3 个 | **4 个** |
| Timestep 通道 | 无（固定 0.5） | **每层传入 timestep** |
| ContextNet + UNet | 无 | **有（残差精修减少伪影）** |
| 权重重合度 | `strict=False` 丢弃大量参数 | **完全匹配** |
| Padding | 32 对齐 | **128 对齐（v4.x 要求）** |
| 推理方式 | 递归二分 | **任意 timestep 一帧一推理** |

### 5.2 Subprocess 回退模式

- **问题**：PyInstaller 打包的 exe 不含 torch（4GB+），RIFE 只能在有 torch 的 Python 环境运行
- **方案**：`_env.py` 自动扫描系统 Python 环境
  - 搜索路径：py.exe、PATH、Miniconda/Anaconda、Python314 等
  - 检测到 torch+CUDA → `_rife_infer.py` 子进程通过 pickle stdin/stdout 通信
  - 对调用方透明：`interpolate(f0, f1)` 接口不变，自动选择 in-process 或 subprocess

---

## 6. GUI 响应性

### 6.1 Probe 缓存与合并

- **问题**：每次键盘输入触发 `_probe_input_width()` + `_probe_input_height()`，各创建一次 FFmpegVideoDecoder
- **方案**：合并为 `_probe_input_info()` 单次 probe，带字典缓存 `{path: {width, height}}`
  - 同一视频文件只 probe 一次
  - 路径变化时清除旧缓存

### 6.2 后台检测系统 Python

- **问题**：启动时扫描所有候选 Python 路径（subprocess 探测 torch）可能耗时数秒，阻塞 UI
- **方案**：后台线程异步检测,完成后 `after(0, callback)` 更新 UI

---

## 7. 潜在进一步优化方向（未实施）

### 7.1 RIFE Warmup 缓存

- RIFE 模型加载后在首帧推理前做一次空推理（warmup），CUDA kernel 编译缓存到后续帧复用
- 约节省首帧 2-3 秒编译延迟

### 7.2 逐帧批处理（Batch Inference）

- 当前 RIFE 每对帧单独推理，可改为 `[N,6,H,W]` batch 输入
- 提升 GPU 利用率（减少 kernel launch 次数）
- 注意：batch 越大显存占用越高，需权衡

### 7.3 编码线程 Pipeline 深度

- 当前 `enc_q maxsize=2`，可考虑 `maxsize=4` 使编码线程预取更深
- 前提：确保 `_check_enc()` 检查及时，不引入额外死锁风险

### 7.4 FPS × Scale 联合调优

- 高倍率插帧（4x）+ 高倍率超分（4x）时 GPU 压力倍增
- 可考虑：先插帧（小分辨率）→ 后超分（最终分辨率）
- vs 当前：先超分 → 后插帧 → 每帧更大

### 7.5 D3D11 Video Processor 共享纹理

- 当前每帧创建 staging → map → memcpy → unmap → CopySubresourceRegion
- 可改为 D3D11_MAP_WRITE_DISCARD + 内存池复用
- 约节省 1-2ms/帧 的内存分配开销

---

## 参考资料

- NVIDIA NVENC Programming Guide: `tune=ll`, `async_depth`
- FFmpeg `av_d2q`: 浮点帧率 → 精确 AVRational
- Practical-RIFE v4.25: 4-block IFNet_m + ContextNet + UNet
- OpenCV Farneback: `pyr_scale/levels/winsize` SVP 启发式预设
- D3D11 VideoProcessorBlt: RTX VSR 驱动拦截原理

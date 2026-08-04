# PERFORMANCE NOTES 3: ncnn 引擎性能优化思路

> 仅基于对话记忆整理，未重新阅读代码。

---

## 一、问题：ncnn 超分+插帧同时使用时速度极慢（< 0.5 fps）

### 根因分析

ncnn 引擎通过 `subprocess.run()` 调用 CLI exe（rife-ncnn-vulkan.exe / realcugan-ncnn-vulkan.exe），每帧都需要：

| 开销项 | 耗时 | 次数/对帧 | 合计 |
|--------|------|-----------|------|
| `subprocess.run` 进程启动 | ~50ms | 6 | 300ms |
| Vulkan GPU context 初始化 | ~30ms | 6 | 180ms |
| JPEG/PNG 编码+写入 | ~20ms | 10 | 200ms |
| JPEG/PNG 解码+读取 | ~15ms | 10 | 150ms |
| 实际推理 | ~30ms | 6 | 180ms |
| `tempfile.mkdtemp` + `rmtree` | ~10ms | 4 | 40ms |
| **合计** | | | **~1050ms/对帧** |

1680 帧 → 840 对帧 → 约 14 分钟，即 **~0.5 fps**。

---

## 二、方案 1：持久 tmpdir + 固定文件名

### 思路
- 在 `initialize()` 时创建 tmpdir，`release()` 时清理
- 所有帧复用同一组固定文件名（f0.png、f1.png、out.png）
- 消除 `mkdtemp`/`rmtree` 的 per-frame 开销

### 效果
- 省约 40ms/帧（mkdtemp+rmtree）
- 实际提速 **~10-15%**（从 0.5fps → 0.55~0.6fps）

### 局限
- 进程启动开销（~50ms/次）无法消除：ncnn CLI 工具为单次调用模式
- 这是 CLI 模式固有的性能天花板

### 状态：已实现

---

## 三、方案 2：ncnn Python binding（进程内推理）

### 发现
- PyPI 上的 `ncnn` 包（由 nihui 维护）提供了 **Python binding**
- Wheel 仅 **5.1 MB**（解压后 ~14 MB）
- 核心：`ncnn.cp313-win_amd64.pyd`（13.9 MB Python 扩展）
- 运行时依赖：仅 `msvcp140.dll` + `vcomp140.dll`（VS C++ 运行时）
- **不需要 Vulkan SDK**：ncnn 直接通过 GPU 驱动的 Vulkan 接口使用 GPU

### 实现
```python
import ncnn

net = ncnn.Net()
net.opt.use_vulkan_compute = True
net.load_param("model.param")
net.load_model("model.bin")

ex = net.create_extractor()
ex.input("in0", mat)
ret, out = ex.extract("out0")
```

- 模型只需加载一次（在 `initialize()` 中）
- 每帧推理直接在 Python 进程中执行
- 零文件 I/O、零进程启动

### 模型结构（探测结果）
- **RIFE v4.6**: inputs=`in0,in1,in2` → output=`out0`
  - in0: frame0 BGR [0,1] float32, padded to 32x multiple
  - in1: frame1 BGR [0,1] float32
  - in2: timestep, 1ch float32 broadcast to padded size
  - out0: interpolated frame BGR
- **Real-CUGAN models-se**: inputs=`in0` → output=`out0`
  - in0: input BGR [0,1] float32
  - out0: upscaled BGR (2×/3×/4× baked into model)

### 性能预期
- 消除所有 subprocess 启动 + 文件 I/O 开销
- 预计提速 **10~50×**（从 0.5fps → 5~25fps）

### 回退策略
- 如果 `import ncnn` 失败（包未安装），自动回退到 CLI 模式
- 如果 ncnn 模型加载/推理失败，自动回退到 CLI 模式
- CLI 回退路径保留原有持久 tmpdir 优化

### 跨设备可用性
- `pip install ncnn` 直接可用，PyPI 有预编译的 cp311/cp312/cp313 Windows 轮子
- 需要支持 Vulkan 的 GPU（NVIDIA/AMD/Intel 均支持，非 NVIDIA 也可用）
- 完全不需要 Vulkan SDK

### 状态：已实现

---

## 四、可行的替代引擎组合（无需 ncnn）

当 ncnn 性能不满足需求时的替代方案：

| 组合 | 预计速度 | 质量 | 依赖 |
|------|---------|------|------|
| **DXVA VSR + RIFE PyTorch (子进程)** | ~15 fps | 最佳 | torch + DXVA |
| **DXVA VSR + 光流法 (Farneback)** | ~60 fps | 一般 | 仅 dxva_vsr_bridge.dll |
| **bicubic + RIFE PyTorch (子进程)** | ~20 fps | 很好 | torch |
| Real-CUGAN ncnn + RIFE ncnn (CLI) | ~0.5 fps | 好 | 仅包含的 exe+模型 |

---

## 五、RIFE 性能相关的其他发现

### RIFE PyTorch 无手动挡位
- `--fi-quality` 参数对 RIFE (PyTorch/ncnn) **不生效**
- RIFE PyTorch 只有基于分辨率的**自动缩放**：
  - ≤ 2K: scale=1.0（全分辨率）
  - 2K~4K: scale=0.5（半分辨率，~4× 快）
  - > 4K: scale=0.25（1/4 分辨率，~16× 快）
- 光流法引擎（dis/optical_flow/torch_flow）才有 ultra/fast/balanced/quality 四档

### ncnn 常驻子进程方案（已放弃）
- 曾考虑通过 stdin/stdout 管道让 ncnn CLI exe 常驻
- 放弃原因：ncnn CLI 工具不支持流式输入/输出模式，为纯文件 I/O 的单次调用设计
- 该分析直接导向了方案 2（ncnn Python binding）

---

## 六、关键结论

1. **ncnn CLI 的性能天花板在 ~0.5-1fps**，由进程启动+文件 I/O 固有开销决定
2. **ncnn Python binding 是最佳解决方案**：5.1MB 依赖换来 10-50× 提速，且不引入 Vulkan SDK
3. **日常使用推荐 DXVA VSR + RIFE PyTorch** 组合，速度/质量平衡最好
4. **CLI 回退路径务必保留**，以应对 `ncnn` 包未安装或 Python 版本不匹配的情况

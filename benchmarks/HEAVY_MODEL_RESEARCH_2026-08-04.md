# 重型视频模型候选评估（2026-08-04）

本页只评估“主程序集成接口与算法适配器、权重由用户按需下载”的候选。排序同时考虑输出质量、代码/权重完整性、许可证、Windows 可落地性，以及 Light Video Enhancer 当前分离式超分/插帧管线能否安全承载。

## 建议顺序

### 1. SeedVR2 7B 量化档

这不是新算法，但复用现有 SeedVR2 worker、VAE、分块与 block-swap 路径，开发风险最低。现已集成 7B GGUF Q4 与 7B Sharp GGUF Q4 两个按需模型包；`quality` 优先普通 7B，`ultra` 优先 Sharp 7B，缺少可选权重时回退 3B FP8。官方项目采用 Apache-2.0，自动选择仍不得启用。

RTX 5070 Ti Laptop 12 GB 实测表明 7B Q4 可以在消费级显卡运行：720p→1440p、5 帧推理 45.253 秒，峰值显存 8,589 MiB、GPU 峰值 100%，相较同机 3B FP8 的 40.738 秒慢约 11%，但并未 OOM。适配器对 7B 固定 5 帧时序批次、384 像素 VAE tile 和最多 36 个 block swap，并设置 11 GiB 显存硬门槛。

下载测试还发现跨源续传缺陷：部分镜像会返回 `206` 和看似正确的 `Content-Range`，却继续发送超过声明对象长度的数据。旧下载器会把多余数据追加到 `.part`，最终只能在 SHA-256 阶段失败。现已增加 `Content-Range` 起点/总长校验、超过声明长度截断、忽略 Range 时覆盖重下，以及“完整 `.part` 校验后直接安装”路径；3 个回归测试覆盖上述行为。

### 2. DLoRAL

DLoRAL 是一阶段真实视频超分模型，官方推理代码采用 MIT。现已集成原生 4× 隔离 worker、约 8.14 GiB 核心模型包和可选内容提示包。运行时固定官方提交 `e8a5574...`，用保持 checkpoint 键名的 Torch/TorchVision 兼容层替代 `mmcv-full` / `mmengine`，避免在 Windows 安装旧二进制栈；主进程不会导入 CUDA。

RTX 5070 Ti Laptop 12 GB 实测：128×128→512×512 两帧在 fast/quality 档分别为 0.782/0.737 fps，峰值显存 9,390/9,354 MiB；320×180→1280×720 fast 为 0.273 fps、8,920 MiB；640×360→2560×1440 fast 为 0.0233 fps、11,822 MiB。1440p 输出已非常接近显存上限，因此保留“实验、手动选择、至少约 11 GiB”门控，不参与自动选择。真实测试还修复了运行时漏打包 `devices.py`、MMCV `ConvModule` 参数键名不一致，以及分块路径不确定性掩码维度错误。

官方仓库：https://github.com/yjsunnn/DLoRAL

### 3. OSDEnhancer（联合时空超分接口）

OSDEnhancer 一次完成 4× 空间超分和 2× 时间插帧，官方代码、修正后的权重和 Apache-2.0 许可证均已发布。现以带 `temporal_multiplier=2` 的联合 SR 适配器接入：管线会独占并禁用额外插帧阶段，按 `(N-1)×2+1` 计算输出、把自然帧率翻倍，并用 5 帧重叠批次保持跨批连续。模型页提供 12,846,839,231 字节的按需下载清单和八个逐文件 SHA-256，运行时固定官方提交 `64dd6e5...`。

官方明确建议不少于 80 GB 显存；worker 与环境扫描均执行硬门控。本机只验证到门控正确返回“检测到 11.9 GiB”，没有下载 12.0 GiB 权重或伪造推理成绩。它不会进入自动选择，只有模型、依赖和显存三项均通过时 WinUI 才显示可用。

官方仓库：https://github.com/W-Shuoyan/OSDEnhancer

### 4. SparkVSR（参考关键帧引导）

SparkVSR 基于 CogVideoX1.5-5B，代码和 Stage-2 权重完整，Apache-2.0。现已接入原生 4×、无参考和本地高质量关键帧两种模式，运行时固定官方提交 `a082284...`，Stage-2 的 21 个文件固定到 Hugging Face 提交 `ec23be7...`，总计 42,199,097,809 字节并逐文件校验。GUI 提供参考目录、源帧编号和引导强度；参考编号必须递增、唯一且至少间隔 4 帧。

该模型永不参与自动选择。安全门要求至少 40 GiB 显存，或至少 11 GiB 显存加 56 GiB 系统内存；当前 12 GB 显存/32 GB 内存主机只验证了禁用路径，没有下载 42.2 GB 权重或虚构推理成绩。

官方仓库：https://github.com/taco-group/SparkVSR

### 5. VFIMamba

VFIMamba 的 Small 与 Full 模型、任意时刻插帧和四档质量映射均已接入。运行时固定官方提交 `8df805e...`，Mamba 的 Apache-2.0 参考扫描固定到 `e9594ce...`；两份权重固定到 Hugging Face 提交 `7c38387...`，合计 331,712,554 字节并已下载、逐文件 SHA-256 校验。`fast/balanced` 使用 Small，`quality/ultra` 使用 Full；fast/quality 对流场做 0.5× 降采样，ultra 另启用 TTA。

Windows 没有可直接复用的兼容 `mamba_ssm` 原生扩展时会安全回退到官方纯 PyTorch selective-scan。RTX 5070 Ti Laptop 实测 64×64、2×：Small fast 初始化 3.399 秒、真实推理 23.650 秒；Full quality 初始化 3.767 秒、真实推理 22.732 秒。输出均为一帧 64×64 uint8，有效范围与非零均值正确。该回退路径只证明功能正确，速度不适合实际视频，因此 VFIMamba 不参与自动选择；安装兼容 CUDA 扩展后 worker 会自动切换原生快路径。

官方仓库：https://github.com/MCG-NJU/VFIMamba

## 暂不集成

- **GIMM-VFI**：质量和任意时间插帧能力有吸引力，但 S-Lab License 只允许非商业用途；在没有额外授权前不应随项目分发或作为默认功能。
- **RDVFI**：ICLR 2026 报告了 1024×576 下 17 FPS 和大运动优势，但截至本次核查没有可验证的官方代码/权重发布入口；只跟踪，不建立空壳下载项。
- **HiFI / HFD**：扩散插帧方向先进，HiFI 可扩到 8K，HFD 强调高效大运动；目前项目页未提供完整可用的官方推理代码与权重，等待发布。
- **FILM**：Windows 与大运动支持成熟、Apache-2.0，但官方仓库已在 2025 年归档，TensorFlow 运行时成本较高；相较现有 RIFE/EMA-VFI 和上述候选，新增维护价值不足。
- **STAR / DOVE**：都基于大型视频扩散底座且代码/权重已发布，但更适合 24–40 GB 以上显存。DOVE 与现有 SeedVR2/DLoRAL 的功能重叠，STAR 的官方示例约需 39 GB；暂列研究后备。
- **LucidFlux**：它是逐图像的生成式修复，默认约需 28 GB，并明确可能幻觉细节；在没有视频一致性层之前不应直接逐帧用于普通视频。

## 架构要求

后续重型算法必须继续遵守：主进程不导入 CUDA 框架；每个模型使用固定提交的运行时 ZIP 与独立 Python 环境；模型清单记录逐文件 SHA-256；stdout 仅用于 framed IPC；所有普通日志写 stderr；能力探测必须真实导入必要模块，但不得在扫描阶段导入已知可能使解释器崩溃的可选扩展；显存不满足时禁用而不是尝试后崩溃；大型/生成式模型永不参与默认自动选择。

"""English translations for legacy backend log and error messages.

The processing core predates the bilingual frontends and still contains a
number of Chinese diagnostic templates.  Keeping the translations here lets
the console, WinUI protocol process, and legacy GUI share one message policy
without importing any GUI toolkit.
"""

from typing import Any, Dict, Tuple

from .i18n import is_chinese


_LOG_TEMPLATES: Dict[str, str] = {
    "融合 Worker 错误输出:\n%s": "Fused worker stderr:\n%s",
    "融合 CUDA Worker 就绪: %dx%d -> %dx%d, %dx, batch=%d":
        "Fused CUDA worker ready: %dx%d -> %dx%d, %dx, batch=%d",
    "原生 NCNN Worker 错误输出:\n%s": "Native NCNN worker stderr:\n%s",
    "原生 NCNN Worker 就绪: %s, batch=%d":
        "Native NCNN worker ready: %s, batch=%d",
    "原生 NCNN 批次: %d -> %d 帧, %.1f ms":
        "Native NCNN batch: %d -> %d frames, %.1f ms",
    "输入: %dx%d @ %.3f fps, %s 帧": "Input: %dx%d @ %.3f fps, %s frames",
    "输出: %dx%d @ %.3f fps": "Output: %dx%d @ %.3f fps",
    "管线: %s -> %s -> %s": "Pipeline: %s -> %s -> %s",
    "完成: %s (%.1f MB)": "Completed: %s (%.1f MB)",
    "融合快速路径就绪: %s": "Fused fast path ready: %s",
    "融合 CUDA 快速路径不可用，回退到独立后端: %s":
        "Fused CUDA fast path unavailable; falling back to separate backends: %s",
    "正在按需扫描 CUDA PyTorch / NV-VFX Python 环境...":
        "Scanning for a CUDA PyTorch / NV-VFX Python environment on demand...",
    "自动选择 Python 环境: %s":
        "Automatically selected Python environment: %s",
    "旧 NCNN 引擎释放失败，常驻 Worker 仍可继续":
        "The old NCNN engine could not be released; the persistent worker can continue",
    "NCNN 常驻快速路径就绪: %s": "Persistent NCNN fast path ready: %s",
    "NCNN 常驻快速路径不可用，回退到 CLI 流水: %s":
        "Persistent NCNN fast path unavailable; falling back to the CLI pipeline: %s",
    "超分就绪: %s": "Super resolution ready: %s",
    "插帧就绪: %s": "Frame interpolation ready: %s",
    "批处理: 输入=%d, 编码队列=%d%s":
        "Batching: input=%d, encoder queue=%d%s",
    "输入 %d 帧，输出 %d 帧": "Read %d frames and wrote %d frames",
    "无法清理 NCNN 临时目录: %s": "Could not clean the NCNN temporary directory: %s",
    "进度回调失败": "Progress callback failed",
    "保留未完成输出: %s": "Kept incomplete output: %s",
    "引擎释放失败": "Engine cleanup failed",
    "无法写入环境缓存": "Could not write the environment cache",
    "无法写入环境缓存: %s": "Could not write the environment cache: %s",
    "Python 环境检测失败: %s": "Python environment probe failed: %s",
    "编码器 %s 不可用，已回退到 %s":
        "Encoder %s is unavailable; fell back to %s",
    "IFRNet ncnn 就绪: %dx%d, %dx, model=%s":
        "IFRNet ncnn ready: %dx%d, %dx, model=%s",
    "RIFE 就绪: %dx%d, %dx, scale=%.2f":
        "RIFE ready: %dx%d, %dx, scale=%.2f",
    "RIFE 权重缺少 %d 个键": "RIFE weights are missing %d keys",
    "RIFE 权重含 %d 个未使用键": "RIFE weights contain %d unused keys",
    "RIFE 共享内存不可用，回退到管道传输":
        "RIFE shared memory is unavailable; falling back to pipe transport",
    "RIFE ncnn 批处理就绪: %dx%d, %dx":
        "RIFE ncnn batch processing ready: %dx%d, %dx",
    "NV-VFX VSR 就绪: %dx%d -> %dx%d (%s, %s)":
        "NV-VFX VSR ready: %dx%d -> %dx%d (%s, %s)",
    "NV-VFX 共享内存不可用，回退到管道传输":
        "NV-VFX shared memory is unavailable; falling back to pipe transport",
    "Real-CUGAN 批处理就绪: %dx%d -> %dx%d (%s)":
        "Real-CUGAN batch processing ready: %dx%d -> %dx%d (%s)",
    "%s 就绪: %dx%d -> %dx%d, model=%s, native=%dx":
        "%s ready: %dx%d -> %dx%d, model=%s, native=%dx",
    "SPAN NCNN 就绪: %dx%d -> %dx%d, model=%s":
        "SPAN NCNN ready: %dx%d -> %dx%d, model=%s",
}


_ERROR_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("未知的超分引擎", "Unknown super-resolution engine"),
    ("未知的插帧引擎", "Unknown interpolation engine"),
    ("原生 NCNN Worker 初始化失败", "Native NCNN worker initialization failed"),
    ("原生 NCNN Worker 尚未初始化", "Native NCNN worker is not initialized"),
    ("原生 NCNN Worker 不可用", "Native NCNN worker is unavailable"),
    ("原生 NCNN Worker 只支持 Vulkan GPU", "Native NCNN worker requires a Vulkan GPU"),
    ("原生 NCNN Worker 未运行", "Native NCNN worker is not running"),
    ("原生 NCNN Worker 已退出", "Native NCNN worker exited"),
    ("原生 NCNN 批次过大", "Native NCNN batch is too large"),
    ("原生 NCNN 输入尺寸不一致", "Native NCNN input dimensions do not match"),
    ("无法向原生 NCNN Worker 发送数据", "Could not send data to the native NCNN worker"),
    ("原生 NCNN 推理失败", "Native NCNN inference failed"),
    ("原生 NCNN 返回帧数错误", "Native NCNN returned an incorrect frame count"),
    ("融合 CUDA 快速路径只在 Windows 10/11 启用",
     "The fused CUDA fast path is available only on Windows 10/11"),
    ("融合 CUDA Worker 尚未初始化", "Fused CUDA worker is not initialized"),
    ("融合 CUDA Worker 未运行", "Fused CUDA worker is not running"),
    ("融合 CUDA Worker 已退出", "Fused CUDA worker exited"),
    ("融合 CUDA 推理失败", "Fused CUDA inference failed"),
    ("融合 Worker 输入尺寸不一致", "Fused worker input dimensions do not match"),
    ("融合 Worker 返回帧数错误", "Fused worker returned an incorrect frame count"),
    ("融合批次过大", "Fused batch is too large"),
    ("没有找到同时支持 CUDA PyTorch 与 nvvfx 的 Python 环境",
     "No Python environment with both CUDA PyTorch and nvvfx was found"),
    ("RIFE 插帧倍率至少为 2", "RIFE interpolation multiplier must be at least 2"),
    ("IFRNet 插帧倍率至少为 2", "IFRNet interpolation multiplier must be at least 2"),
    ("缺少 RIFE 权重", "RIFE weights are missing"),
    ("RIFE ncnn 资源不完整", "RIFE ncnn resources are incomplete"),
    ("RIFE ncnn 目录批处理至少需要 2 帧",
     "RIFE ncnn directory processing requires at least two frames"),
    ("RIFE ncnn 批处理失败", "RIFE ncnn batch processing failed"),
    ("RIFE 需要 CUDA PyTorch；也可选择 RIFE ncnn-vulkan",
     "RIFE requires CUDA PyTorch; RIFE ncnn-vulkan is also available"),
    ("RIFE 子进程启动失败", "RIFE subprocess failed to start"),
    ("RIFE 输入帧尺寸不一致", "RIFE input frame dimensions do not match"),
    ("RIFE 子进程已退出", "RIFE subprocess exited"),
    ("RIFE 子进程通信失败", "RIFE subprocess communication failed"),
    ("RIFE 推理失败", "RIFE inference failed"),
    ("RIFE 子进程返回了无效数据", "RIFE subprocess returned invalid data"),
    ("IFRNet NCNN 仅支持 Vulkan GPU", "IFRNet NCNN requires a Vulkan GPU"),
    ("IFRNet ncnn 资源不完整", "IFRNet ncnn resources are incomplete"),
    ("IFRNet 必须通过原生 NCNN 常驻 Worker 运行",
     "IFRNet requires the persistent native NCNN worker"),
    ("请确认 lve-ncnn-worker.exe 可用且未禁用快速路径",
     "Make sure lve-ncnn-worker.exe is available and the fast path is enabled"),
    ("SPAN NCNN 仅支持 Vulkan GPU", "SPAN NCNN requires a Vulkan GPU"),
    ("SPAN 模型资源不完整", "SPAN model resources are incomplete"),
    ("SPAN 需要原生 NCNN Worker；请恢复 lve-ncnn-worker.exe",
     "SPAN requires the native NCNN worker; restore lve-ncnn-worker.exe"),
    ("Real-CUGAN 资源不完整", "Real-CUGAN resources are incomplete"),
    ("Real-CUGAN 批处理失败", "Real-CUGAN batch processing failed"),
    ("未找到 realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan.exe was not found"),
    ("ESRGAN 模型资源不完整", "ESRGAN model resources are incomplete"),
    ("批处理失败", "batch processing failed"),
    ("GPU 光流需要当前 Python 环境中的 CUDA PyTorch",
     "GPU optical flow requires CUDA PyTorch in the current Python environment"),
    ("外部 Python 的 PyTorch CUDA 不可用",
     "PyTorch CUDA is unavailable in the external Python environment"),
    ("模型权重不存在", "Model weights do not exist"),
    ("打包版使用 NV-VFX 时需要先选择外部 CUDA Python 环境",
     "The packaged NV-VFX backend requires a selected external CUDA Python environment"),
    ("NV-VFX 初始化失败", "NV-VFX initialization failed"),
    ("NV-VFX 尚未初始化", "NV-VFX is not initialized"),
    ("NV-VFX 推理失败", "NV-VFX inference failed"),
    ("NV-VFX 返回了无效数据", "NV-VFX returned invalid data"),
    ("NV-VFX 子进程未运行", "NV-VFX subprocess is not running"),
    ("NV-VFX 子进程已退出", "NV-VFX subprocess exited"),
    ("NV-VFX 子进程通信失败", "NV-VFX subprocess communication failed"),
    ("在 %.0f 秒内没有响应", "did not respond within %.0f seconds"),
    ("在 ", "within "),
    (" 秒内没有响应", " seconds"),
    ("预期", "expected"),
)


def translate_log_template(value: Any) -> Any:
    """Translate a logging template or textual argument when English is active."""
    if is_chinese() or not isinstance(value, str):
        return value
    translated = _LOG_TEMPLATES.get(value, value)
    if translated == value:
        for chinese, english in _ERROR_PHRASES:
            translated = translated.replace(chinese, english)
    return translated

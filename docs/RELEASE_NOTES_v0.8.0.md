# Light Video Enhancer v0.8.0

v0.8.0 将上一轮重型模型研究整理为可维护的正式接口，并使用 2026 年 7 月的稳定 Windows 组件重新构建全部发行包。大型权重仍由用户按需安装，标准包只携带运行时桥接与必要的内置模型。

## 新增算法与模型

- DLoRAL：一阶段 4× 扩散视频超分，可选内容提示模型。
- OSDEnhancer：联合 4× 空间超分与 2× 时间插帧，面向超大显存工作站。
- SparkVSR Stage-2：支持高质量关键帧传播的 4× 视频超分。
- VFIMamba S / Full：状态空间视频插帧；CUDA 扩展不可用时提供安全但较慢的 PyTorch 回退。
- SeedVR2：增加 7B Q4 与 7B Sharp Q4 档位，并复用 3B 模型包中的 VAE。

这些重型模型均要求用户明确选择，不参与自动选择。Full 包同样不内置其数百 MiB 至数十 GiB 的权重。

## 下载、环境与界面

- 模型下载支持断点续传、Google Drive、大文件镜像、自定义源和逐文件 SHA-256 校验。
- 环境扫描会分别报告 DLoRAL、OSDEnhancer、SparkVSR、SeedVR2 与 VFIMamba 的真实可用状态。
- WinUI、独立 CLI 和模型页共用版本化后端协议及中英文名称。
- WinUI 升级至稳定版 Windows App SDK 2.3.1；使用 .NET SDK 10.0.302 与 Windows SDK BuildTools 10.0.28000.2270 构建，最低系统仍为 Windows 10 1809 x64。

## 发行包

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `LightVideoEnhancer-WinUI3-Full-Win10-11-x64.zip` | 389.18 MiB | `676610FF33B382362551DD58F554FB0861AA6552FD10FB58C913EE4C18A93166` |
| `LightVideoEnhancer-WinUI3-Lite-Win10-11-x64.zip` | 152.56 MiB | `1B5F108AD406C8D8B2FF6924B748BBE9A9D43BD587C83476F0F06B08A3739E3D` |
| `LightVideoEnhancer-Win7-x64.exe` | 318.97 MiB | `8782F7A5580793417884E0B1A831476BD46AB1A25EDC2A19F220CA83F539FEEA` |

Full 与 Lite ZIP 均只有 `LightVideoEnhancer.WinUI.exe`、`LightVideoEnhancer-Backend.exe` 和 `CLI_GUIDE.md` 三个根目录文件。两者前端 SHA-256 都是 `C01EE09D36B373CA8BFB62379F7F09E1551D388BC6EA5F88FCFCB5B4554C4E66`。

## 验证

- Python 单元及集成测试：89 项通过，1 项真实 Vulkan 环境测试按条件跳过。
- WinUI Release x64 编译：0 警告、0 错误；自包含裁剪发布保留 2 项 Windows SDK/WinRT IL2104 分析警告。
- Full/Lite 后端与 Win7 GUI 的文件版本均为 0.8.0.0。
- 隔离空模型目录：Full 识别 19 个模型包并内置 10 个；Lite 识别 19 个且内置 0 个。
- WinUI Full 与 Windows 7 Tk LTS 均通过启动存活测试。

## English summary

v0.8.0 adds isolated integrations for DLoRAL, OSDEnhancer, SparkVSR, and VFIMamba, plus SeedVR2 7B Q4 profiles. Heavy weights remain optional downloads and are never auto-selected. Model downloads now support resuming, Google Drive, mirrors, custom sources, and per-file SHA-256 checks. The WinUI frontend is rebuilt with the stable Windows App SDK 2.3.1 while retaining Windows 10 version 1809 as its minimum supported OS. Full, Lite, and Windows 7 LTS packages were rebuilt and smoke-tested from the same versioned processing core.

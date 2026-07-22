# Light Video Enhancer v0.5.0

v0.5.0 brings a modern WinUI 3 interface and separates the Windows 10/11 release into Full and Lite packages. Both packages use the exact same GUI: Full contains every supported model, while Lite keeps the application core and lets users install only the models they need.

## Highlights

- Modern WinUI 3 frontend for Windows 10 1809 and later.
- Full and Lite packages with one shared frontend and a stable versioned backend protocol.
- Model manager with GitHub, proxy, custom source, and verified local ZIP import.
- Per-user model storage that survives application upgrades.
- Chinese and English UI/CLI language selection.
- New Light Video Enhancer logo and complete Windows icon assets.
- Windows 7 Full package retained as a frozen Tk-based LTS build.

## Model safety

Every official model pack has an archive SHA-256 and per-file SHA-256 values in the bundled manifest. The installer rejects unknown files, missing files, hash mismatches, and paths that would escape the model directory.

## Upgrade notes

- Windows 10/11 users should remove the old Tk executable and choose either the Full or Lite WinUI ZIP.
- Lite users can install models from the **Models & Downloads** page. Models are stored under `%LOCALAPPDATA%\LightVideoEnhancer\models` by default.
- Windows 7 users should continue using the Full Tk package. It intentionally has no model download page.

---

# Light Video Enhancer v0.5.0 中文说明

v0.5.0 为 Windows 10/11 引入现代 WinUI 3 界面，并将发行包拆分为 Full 与 Lite。两个包使用完全相同的 GUI：Full 内置全部模型，Lite 只保留核心文件，用户可按需安装模型。

## 主要变化

- 支持 Windows 10 1809 及以上版本的 WinUI 3 前端。
- Full/Lite 共用前端，通过稳定、带版本号的协议调用后端。
- 模型下载页支持 GitHub、代理、自定义源及经过完整校验的本地 ZIP。
- 模型保存在用户目录，更新程序时无需重复下载。
- WinUI 与 CLI 支持中文和英文选择。
- 新应用 Logo 与完整 Windows 图标资源。
- Windows 7 Full Tk 版作为冻结 LTS 继续提供。

## 升级提示

- Windows 10/11 用户应停止使用旧 Tk 版本，改用 Full 或 Lite WinUI ZIP。
- Lite 用户可在“模型与下载”页按需安装；默认目录为 `%LOCALAPPDATA%\LightVideoEnhancer\models`。
- Windows 7 用户继续使用内含全部模型的 Full Tk 包；该版本不会增加下载页面。

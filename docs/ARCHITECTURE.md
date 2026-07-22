# Frontend/backend architecture

Light Video Enhancer keeps processing code independent from its user interfaces.

## Supported frontends

- **WinUI 3 (Windows 10 version 1809 and newer)** is the actively developed GUI.
- **CLI** is a first-class frontend and exposes every processing option.
- **Tk (Windows 7 SP1)** is a frozen compatibility frontend. It receives critical backend compatibility fixes, but no new Windows 10/11 UI features or model-download page.

No frontend imports CUDA, Vulkan, NCNN, NVIDIA VFX, or FFmpeg worker code. The WinUI process starts `LightVideoEnhancer-Backend.exe` and exchanges UTF-8 lines over standard input/output. Tk remains an in-process legacy adapter because Python 3.8 and Windows 7 make the extra deployment layer disproportionately expensive; it only constructs `ProcessConfig` and calls the same `VideoEnhancer` facade as the CLI.

## Stable frontend protocol

`light_video_enhancer/frontend_protocol.py` owns the UI-neutral protocol. The current `protocol_version` is `1`.

| Command | Output |
|---|---|
| `--capabilities-json` | one JSON object containing version, protocol version, GPU, encoder and engine availability |
| `--environments-json [--force]` | one JSON array of discovered Python/PyTorch environments |
| `--models-json` | model directory, source list, pack metadata and installed state |
| `--download-model PACK --model-source SOURCE` | progress lines followed by a result JSON object |
| `--install-model-pack PACK FILE.zip` | verified local install |
| `--remove-model PACK` | removes only the per-user copy |
| normal processing with `--progress-json --control-stdin` | progress lines; accepts `cancel` on stdin |

Progress lines begin with `__LVE_PROGRESS__` followed by a JSON object. New fields may be added without changing the protocol version. Removing/renaming fields or changing command semantics requires a protocol-version increment and a compatibility adapter in the frontend.

## Model ownership

Processing engines resolve weights through `_paths.get_model_file()` or `_paths.get_model_dir()` only:

1. `LVE_MODEL_DIR`, when explicitly set;
2. `%LOCALAPPDATA%\LightVideoEnhancer\models`;
3. weights bundled inside the executable.

The Full and Lite packages therefore use identical source code. Full reports bundled weights as `bundled`; Lite reports them as `missing` until the model manager verifies and installs a pack. Downloads never write into the application folder and do not require elevation.

`tools/build_model_packs.py` builds deterministic ZIP files and `model_manifest.json`. The installer validates the archive SHA-256, the SHA-256 of every extracted file, the exact file list, and safe relative paths before atomically replacing installed files.

## Maintenance boundary

- `pipeline.py`, engine packages, `encoding.py`, `_paths.py`, `model_manager.py`: processing/backend owners.
- `frontend_protocol.py`: stable contract owner; contains no UI code.
- `windows/LightVideoEnhancer.WinUI`: modern frontend owner; contains no processing implementation.
- `gui.py`: Windows 7 compatibility adapter only.

A backend change is complete only when CLI tests and protocol tests pass. A GUI change must consume the protocol instead of reading package files or probing CUDA/PyTorch directly.

## Windows 7 policy recommendation

Keep one Full, offline-capable Windows 7 LTS package for now, but freeze its feature set. Windows 7 and Python 3.8 no longer receive upstream security maintenance, and active feature parity would permanently constrain the modern backend and GUI. Reconsider removal after at least one release cycle with download statistics or user feedback; until then, a frozen compatibility package costs less and avoids removing a working path without evidence.

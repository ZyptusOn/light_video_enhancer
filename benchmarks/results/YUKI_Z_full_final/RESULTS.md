# Video pipeline benchmark

- Input: `C:\Users\24645\Downloads\YUKI_Z.mp4`
- Segment: 0.000–16.100 s
- Source: 1280x720 @ 29.999 fps, 483 frames total
- Output target: 2× resolution, 2× frame rate, `hevc_nvenc` / `p5` / CQ 23

| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RIFE PyTorch + NVIDIA Video Effects VSR | 21.24 | 5.85 | 31.37 | 62.68 | 56.8/79.0% | 32.6/48.0% | 1527 | 89.4 | 1.00x | OK |

Raw samples and paths are stored in `results.json`.

# Video pipeline benchmark

- Input: `C:\Users\24645\Downloads\YUKI_Z.mp4`
- Segment: 0.000–16.100 s
- Source: 1280x720 @ 29.999 fps, 483 frames total
- Output target: 2× resolution, 2× frame rate, `hevc_nvenc` / `p5` / CQ 23

| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RIFE PyTorch + NVIDIA Video Effects VSR | 25.02 | 5.24 | 24.42 | 48.78 | 40.9/70.0% | 17.6/21.9% | 1513 | 75.6 | 1.00x | OK |

Raw samples and paths are stored in `results.json`.

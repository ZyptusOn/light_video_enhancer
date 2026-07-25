# Video pipeline benchmark

- Input: `C:\Users\24645\Downloads\YUKI_Z.mp4`
- Segment: 4.000–8.000 s
- Source: 1280x720 @ 29.999 fps, 483 frames total
- Output target: 2× resolution, 2× frame rate, `hevc_nvenc` / `p5` / CQ 23

| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RIFE PyTorch only | 14.24 | 4.84 | 12.76 | 25.42 | 14.5/54.0% | 24.9/34.0% | 1130 | 38.1 | 0.00x | OK |
| NVIDIA Video Effects VSR only | 8.54 | 5.17 | 35.57 | 35.57 | 24.7/48.0% | 27.3/38.8% | 1067 | 53.4 | 0.00x | OK |

Raw samples and paths are stored in `results.json`.

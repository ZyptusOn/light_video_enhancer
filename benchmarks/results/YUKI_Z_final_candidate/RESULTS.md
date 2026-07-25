# Video pipeline benchmark

- Input: `C:\Users\24645\Downloads\YUKI_Z.mp4`
- Segment: 4.000–8.000 s
- Source: 1280x720 @ 29.999 fps, 483 frames total
- Output target: 2× resolution, 2× frame rate, `hevc_nvenc` / `p5` / CQ 23

| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RIFE PyTorch + NVIDIA Video Effects VSR | 11.42 | 5.94 | 21.88 | 43.57 | 37.0/71.0% | 30.8/41.1% | 1507 | 80.6 | 1.00x | OK |
| RIFE PyTorch only | 8.58 | 4.68 | 30.80 | 61.35 | 32.1/57.0% | 29.2/35.4% | 1094 | 47.1 | 1.33x | OK |

Raw samples and paths are stored in `results.json`.

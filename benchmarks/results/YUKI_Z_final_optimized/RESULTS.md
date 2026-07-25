# Video pipeline benchmark

- Input: `C:\Users\24645\Downloads\YUKI_Z.mp4`
- Segment: 4.000–8.000 s
- Source: 1280x720 @ 29.999 fps, 483 frames total
- Output target: 2× resolution, 2× frame rate, `hevc_nvenc` / `p5` / CQ 23

| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RIFE PyTorch + NVIDIA Video Effects VSR | 11.47 | 5.98 | 21.87 | 43.55 | 40.7/78.0% | 40.4/53.0% | 1575 | 64.9 | 1.00x | OK |
| RIFE PyTorch only | 9.06 | 5.12 | 30.49 | 60.72 | 31.3/55.0% | 23.5/41.8% | 1094 | 51.0 | 1.27x | OK |
| NVIDIA Video Effects VSR only | 25.08 | 21.42 | 32.82 | 32.82 | 20.7/49.0% | 28.7/42.3% | 1023 | 44.8 | 0.46x | OK |

Raw samples and paths are stored in `results.json`.

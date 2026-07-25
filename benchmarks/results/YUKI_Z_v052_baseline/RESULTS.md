# Video pipeline benchmark

- Input: `C:\Users\24645\Downloads\YUKI_Z.mp4`
- Segment: 4.000–8.000 s
- Source: 1280x720 @ 29.999 fps, 483 frames total
- Output target: 2× resolution, 2× frame rate, `hevc_nvenc` / `p5` / CQ 23

| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RIFE PyTorch + NVIDIA Video Effects VSR | 31.09 | 15.51 | 7.70 | 15.34 | 14.3/66.0% | 15.7/23.5% | 1499 | 35.1 | 1.00x | OK |
| RIFE PyTorch + D3D11 driver VSR | 23.92 | 4.42 | 6.15 | 12.25 | 7.7/37.0% | 32.0/57.6% | 1239 | 23.6 | 1.30x | OK |
| RIFE PyTorch only | 14.30 | 4.67 | 12.46 | 24.82 | 13.5/57.0% | 26.7/52.3% | 1130 | 32.2 | 2.17x | OK |
| NVIDIA Video Effects VSR only | 22.35 | 16.12 | 19.27 | 19.27 | 12.6/44.0% | 25.5/36.7% | 1078 | 34.1 | 1.39x | OK |
| D3D11 driver VSR only | 10.34 | 2.87 | 16.06 | 16.06 | 1.2/4.0% | 24.3/36.3% | 491 | 20.1 | 3.01x | OK |

Raw samples and paths are stored in `results.json`.

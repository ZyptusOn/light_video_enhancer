# Video pipeline benchmark

- Input: `C:\Users\24645\Downloads\YUKI_Z.mp4`
- Segment: 4.000–8.000 s
- Source: 1280x720 @ 29.999 fps, 483 frames total
- Output target: 2× resolution, 2× frame rate, `hevc_nvenc` / `p5` / CQ 23

| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RIFE PyTorch + NVIDIA Video Effects VSR | 22.73 | 5.84 | 7.11 | 14.15 | 14.7/76.0% | 38.0/50.2% | 2061 | 31.7 | 1.00x | OK |

Raw samples and paths are stored in `results.json`.

# Video pipeline benchmark

- Input: `C:\Users\24645\Downloads\YUKI_Z.mp4`
- Segment: 4.000–4.200 s
- Source: 1280x720 @ 29.999 fps, 483 frames total
- Output target: 2× resolution, 2× frame rate, `hevc_nvenc` / `p5` / CQ 23

| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D3D11 driver VSR only | 4.57 | 3.12 | 4.13 | 4.13 | 2.0/4.0% | 26.8/34.3% | 477 | 19.9 | 0.00x | OK |

Raw samples and paths are stored in `results.json`.

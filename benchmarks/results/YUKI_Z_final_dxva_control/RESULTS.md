# Video pipeline benchmark

- Input: `C:\Users\24645\Downloads\YUKI_Z.mp4`
- Segment: 4.000–8.000 s
- Source: 1280x720 @ 29.999 fps, 483 frames total
- Output target: 2× resolution, 2× frame rate, `hevc_nvenc` / `p5` / CQ 23

| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RIFE PyTorch + D3D11 driver VSR | 19.41 | 4.98 | 8.32 | 16.57 | 9.5/26.0% | 19.2/27.1% | 1203 | 29.0 | 0.00x | OK |

Raw samples and paths are stored in `results.json`.

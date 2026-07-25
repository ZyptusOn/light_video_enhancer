# Video pipeline benchmark

- Input: `C:\Users\24645\Downloads\YUKI_Z.mp4`
- Segment: 0.000–2.000 s
- Source: 1280x720 @ 29.999 fps, 483 frames total
- Output target: 2× resolution, 2× frame rate, `hevc_nvenc` / `p5` / CQ 23

| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RIFE NCNN + Real-CUGAN (persistent native worker) | 21.59 | 9.57 | 4.99 | 9.90 | 60.8/79.0% | 25.3/32.3% | 1568 | 104.4 | 0.00x | OK |
| RIFE NCNN + Real-ESRGAN AnimeVideo-v3 (native) | 15.84 | 8.50 | 8.18 | 16.22 | 58.6/81.0% | 24.0/28.9% | 782 | 107.0 | 0.00x | OK |

Raw samples and paths are stored in `results.json`.

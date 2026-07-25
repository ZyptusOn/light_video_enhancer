# Video pipeline benchmark

- Input: `C:\Users\24645\Downloads\YUKI_Z.mp4`
- Segment: 4.000–8.000 s
- Source: 1280x720 @ 29.999 fps, 483 frames total
- Output target: 2× resolution, 2× frame rate, `hevc_nvenc` / `p5` / CQ 23

| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RIFE PyTorch + Real-CUGAN | 109.00 | 4.44 | 1.15 | 2.29 | 29.0/100.0% | 44.7/89.3% | 4251 | 44.1 | 0.00x | OK |
| RIFE PyTorch + Real-ESRGAN AnimeVideo-v3 | 81.27 | 4.67 | 1.57 | 3.12 | 23.9/100.0% | 25.6/58.4% | 1555 | 45.3 | 0.00x | OK |
| RIFE NCNN + Real-CUGAN | 84.54 | 2.74 | 1.47 | 2.92 | 33.0/100.0% | 19.2/37.4% | 3492 | 56.1 | 0.00x | OK |
| RIFE NCNN + Real-ESRGAN AnimeVideo-v3 | 77.28 | 2.73 | 1.61 | 3.21 | 25.3/100.0% | 42.0/75.1% | 1384 | 40.9 | 0.00x | OK |

Raw samples and paths are stored in `results.json`.

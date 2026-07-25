# Video pipeline benchmark

- Input: `C:\Users\24645\Downloads\YUKI_Z.mp4`
- Segment: 0.000–2.000 s
- Source: 1280x720 @ 29.999 fps, 483 frames total
- Output target: 2× resolution, 2× frame rate, `hevc_nvenc` / `p5` / CQ 23

| Pipeline | Wall s | Init s | Input fps | Output fps | GPU avg/peak | CPU avg/peak | VRAM peak MiB | Power W | vs baseline | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RIFE PyTorch + NVIDIA Video Effects VSR | 23.84 | 16.35 | 8.01 | 15.88 | 12.4/59.0% | 12.4/17.7% | 1527 | 31.4 | 1.00x | OK |
| RIFE NCNN + Real-CUGAN (persistent native worker) | 20.98 | 9.50 | 5.23 | 10.36 | 58.3/75.0% | 20.9/33.5% | 1568 | 100.4 | 1.14x | OK |
| RIFE NCNN + Real-CUGAN (legacy CLI/PNG) | 42.91 | 3.02 | 1.50 | 2.98 | 39.0/100.0% | 32.4/57.7% | 3492 | 53.8 | 0.56x | OK |
| RIFE NCNN + Real-ESRGAN AnimeVideo-v3 (native) | 15.99 | 8.62 | 8.14 | 16.14 | 58.9/80.0% | 25.4/32.5% | 782 | 117.4 | 1.49x | OK |
| RIFE NCNN + Real-ESRGAN AnimeVideo-v3 (CLI/PNG) | 35.38 | 3.00 | 1.85 | 3.67 | 26.5/100.0% | 26.8/59.9% | 972 | 46.7 | 0.67x | OK |
| RIFE NCNN + ESRGAN classic (persistent native worker) | 192.34 | 10.66 | 0.33 | 0.65 | 90.7/95.0% | 18.8/53.6% | 1794 | 136.0 | 0.12x | OK |
| RIFE NCNN + ESRGAN classic (legacy CLI/PNG) | 426.22 | 3.16 | 0.14 | 0.28 | 70.5/100.0% | 31.5/100.0% | 2838 | 104.6 | 0.06x | OK |

Raw samples and paths are stored in `results.json`.

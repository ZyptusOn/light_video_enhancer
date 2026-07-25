# Persistent NCNN Vulkan worker

This worker embeds the upstream RIFE, Real-CUGAN, and Real-ESRGAN NCNN
implementations in one process. Frames are exchanged through Windows named
shared memory; stdin/stdout carry only small versioned binary control packets.
Models and Vulkan pipelines remain resident for the entire video job.

The portable upstream command-line executables remain the compatibility
fallback. The worker is selected only on supported Windows/Vulkan systems and
any initialization failure falls back before video processing starts.

Upstream source snapshots:

- `nihui/rife-ncnn-vulkan`
- `nihui/realcugan-ncnn-vulkan`
- `xinntao/Real-ESRGAN-ncnn-vulkan`

Their license texts are preserved under `upstream/licenses`. Tencent NCNN is a
build-time dependency and is not vendored here; `build_worker.ps1` accepts a
checkout through `-NcnnSource`.

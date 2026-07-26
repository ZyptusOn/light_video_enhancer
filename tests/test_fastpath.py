import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from light_video_enhancer._shared_frames import SharedNDArray, close_process_pipes
from light_video_enhancer.config import ProcessConfig
from light_video_enhancer.fused_rife_nvvfx import (
    FusedRifeNvvfxEngine,
    YUV420Frame,
    modern_windows_available,
)
from light_video_enhancer.pipeline import VideoEnhancer, _AsyncDirectoryCleaner, _AsyncEncoder


class _RecordingEncoder:
    def __init__(self):
        self.calls = []

    def encode_yuv(self, data, width, height):
        self.calls.append((np.asarray(data).copy(), width, height))


class FastPathTests(unittest.TestCase):
    def test_named_shared_array_round_trip(self):
        owner = SharedNDArray.create((2, 3, 4), np.uint8)
        attached = None
        try:
            attached = SharedNDArray.attach(owner.descriptor())
            attached.array[:] = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
            np.testing.assert_array_equal(owner.array, attached.array)
        finally:
            if attached is not None:
                attached.close()
            owner.close()

    def test_process_pipe_cleanup_is_idempotent(self):
        process = type("Process", (), {
            "stdin": io.BytesIO(),
            "stdout": io.BytesIO(),
            "stderr": io.BytesIO(),
        })()
        close_process_pipes(process)
        close_process_pipes(process)
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)

    def test_fused_output_count_handles_chunk_overlap(self):
        self.assertEqual(FusedRifeNvvfxEngine.output_count(3, 2), 5)
        self.assertEqual(FusedRifeNvvfxEngine.output_count(3, 2, skip_first=True), 4)
        self.assertEqual(FusedRifeNvvfxEngine.output_count(2, 4, skip_first=True), 4)

    def test_async_encoder_accepts_preconverted_i420(self):
        recorder = _RecordingEncoder()
        encoder = _AsyncEncoder(recorder)
        data = np.arange(24, dtype=np.uint8)
        encoder.put(YUV420Frame(data, 4, 4))
        self.assertEqual(encoder.finish(), 1)
        self.assertEqual(recorder.calls[0][1:], (4, 4))
        np.testing.assert_array_equal(recorder.calls[0][0], data)

    def test_async_directory_cleaner_removes_workspace(self):
        path = tempfile.mkdtemp(prefix="lve-cleaner-test-")
        with open(os.path.join(path, "frame.tmp"), "wb") as handle:
            handle.write(b"frame")
        cleaner = _AsyncDirectoryCleaner()
        cleaner.submit(path)
        cleaner.finish()
        self.assertFalse(os.path.exists(path))

    def test_fusion_python_prefers_cached_nvvfx_without_scanning(self):
        preferred = r"C:\envs\torch-only\python.exe"
        combined = r"C:\envs\torch-nvvfx\python.exe"
        enhancer = VideoEnhancer(ProcessConfig(torch_python=preferred))
        environments = [
            {"exe": preferred, "torch": True, "cuda": True, "nvvfx": False},
            {"exe": combined, "torch": True, "cuda": True, "nvvfx": True},
        ]
        with mock.patch("light_video_enhancer._env.get_cached_python_envs",
                        return_value=environments), mock.patch(
                            "light_video_enhancer._env.get_all_python_envs") as scan:
            self.assertEqual(enhancer._fusion_python(), combined)
        scan.assert_not_called()

    def test_fused_path_can_be_disabled_for_compatibility(self):
        with mock.patch.dict(os.environ, {"LVE_DISABLE_FUSED_CUDA": "1"}):
            self.assertFalse(modern_windows_available())


if __name__ == "__main__":
    unittest.main()

import io
import os
import unittest
from unittest import mock

import numpy as np

from light_video_enhancer.fi.ifrnet_ncnn import IFRNetNcnnEngine
from light_video_enhancer.fi.rife_ncnn import RIFENcnnEngine
from light_video_enhancer.native_ncnn import (
    NativeNcnnEngine,
    _MAGIC,
    _REPLY,
    _ReplyReader,
    spec_from_engines,
)
from light_video_enhancer.sr.realcugan_ncnn import RealCUGANEngine
from light_video_enhancer.sr.realesrgan_ncnn import RealESRGANEngine
from light_video_enhancer.sr.span_ncnn import SPANNcnnEngine


class NativeNcnnTests(unittest.TestCase):
    def test_output_count_accounts_for_overlapped_chunks(self):
        self.assertEqual(NativeNcnnEngine.output_count(5, 2), 9)
        self.assertEqual(
            NativeNcnnEngine.output_count(5, 2, skip_first=True), 8)
        self.assertEqual(
            NativeNcnnEngine.output_count(2, 4, skip_first=True), 4)

    def test_binary_reply_reader_validates_and_decodes_protocol(self):
        message = "NVIDIA GPU".encode("utf-8")
        payload = _REPLY.pack(
            _MAGIC, 1, 0, 7, len(message), 12.5) + message
        reader = _ReplyReader(io.BytesIO(payload))
        status, count, decoded, elapsed = reader.read(1)
        self.assertEqual((status, count, decoded), (0, 7, "NVIDIA GPU"))
        self.assertAlmostEqual(elapsed, 12.5)

    def test_engine_adapters_produce_exact_native_model_paths(self):
        rife = RIFENcnnEngine(quality="ultra", gpu_id=1)
        rife._model_dir = os.path.join("models", "rife-v4.6")
        rife._width, rife._height = 2560, 1440

        cugan = RealCUGANEngine(quality="balanced", gpu_id=1)
        cugan._models = os.path.join("models", "cugan")
        cugan._scale = 2
        with mock.patch(
                "light_video_enhancer.sr.realcugan_ncnn.ncnn_tile",
                return_value=192):
            spec = spec_from_engines(cugan, rife, 1)
        self.assertEqual(spec.gpu_id, 1)
        self.assertTrue(spec.fi_tta)
        self.assertTrue(spec.fi_uhd)
        self.assertEqual(spec.fi_kind, "rife")
        self.assertEqual(spec.sr_kind, "realcugan")
        self.assertEqual(spec.sr_noise, 0)
        self.assertEqual(spec.sr_tile, 192)
        self.assertTrue(spec.sr_param.endswith(
            os.path.join("cugan", "up2x-conservative.param")))

        esrgan = RealESRGANEngine(quality="fast")
        esrgan._models_dir = os.path.join("models", "esrgan")
        esrgan._model_name = "realesr-animevideov3"
        esrgan._native_scale = 2
        spec = spec_from_engines(esrgan, None, None)
        self.assertEqual(spec.sr_kind, "realesrgan")
        self.assertEqual(spec.sr_scale, 2)
        self.assertTrue(spec.sr_param.endswith(
            "realesr-animevideov3-x2.param"))

        ifrnet = IFRNetNcnnEngine(quality="ultra", gpu_id=1)
        ifrnet._model_dir = os.path.join("models", "ifrnet-large")
        spec = spec_from_engines(None, ifrnet, 1)
        self.assertEqual(spec.fi_kind, "ifrnet")
        self.assertTrue(spec.fi_tta)
        self.assertTrue(spec.fi_model.endswith("ifrnet-large"))

        span = SPANNcnnEngine(quality="quality", gpu_id=1)
        span._param = os.path.join("models", "spanx2_ch52.param")
        span._model = os.path.join("models", "spanx2_ch52.bin")
        span._scale = 2
        spec = spec_from_engines(span, None, 1)
        self.assertEqual(spec.sr_kind, "span")
        self.assertEqual(spec.sr_scale, 2)
        self.assertTrue(spec.sr_param.endswith("spanx2_ch52.param"))
        self.assertTrue(spec.sr_model.endswith("spanx2_ch52.bin"))

    @unittest.skipUnless(
        os.environ.get("LVE_NATIVE_SMOKE", "") == "1",
        "set LVE_NATIVE_SMOKE=1 for the real Vulkan worker smoke test")
    def test_span_worker_preserves_byte_range_and_bgr_order(self):
        span = SPANNcnnEngine(quality="fast")
        span.initialize(64, 48, 128, 96)
        worker = NativeNcnnEngine(spec_from_engines(span, None, None))
        try:
            worker.initialize(64, 48, 128, 96, multiplier=1)
            frame = np.empty((48, 64, 3), dtype=np.uint8)
            frame[..., 0] = 20
            frame[..., 1] = np.arange(64, dtype=np.uint8)[None, :]
            frame[..., 2] = 200
            output = worker.process([frame])[0]
        finally:
            worker.release()
        self.assertEqual(output.shape, (96, 128, 3))
        self.assertGreater(float(output.mean()), 10.0)
        self.assertGreater(float(output.std()), 2.0)
        self.assertGreater(
            float(output[..., 2].mean()), float(output[..., 0].mean()) + 30.0)

    def test_incompatible_engine_combinations_keep_legacy_path(self):
        self.assertIsNone(spec_from_engines(object(), None, None))
        self.assertIsNone(spec_from_engines(None, object(), None))
        self.assertIsNone(spec_from_engines(None, None, None))


if __name__ == "__main__":
    unittest.main()

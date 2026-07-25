import os
import pickle
import sys
import unittest

import numpy as np

from light_video_enhancer._image_batch import ncnn_jobs, ncnn_tile
from light_video_enhancer.capabilities import GPUInfo, _vendor_from_text, choose_codec, quick_capabilities
from light_video_enhancer.cli import _auto_output
from light_video_enhancer.config import ProcessConfig
from light_video_enhancer.encoding import canonical_codec, codec_candidates
from light_video_enhancer.ffmpeg_bridge.encoder import _codec_preset
from light_video_enhancer.fi import create_fi_engine
from light_video_enhancer.fi._scene_detect import (
    PAIR_NORMAL, PAIR_SCENE_CUT, PAIR_STATIC, classify_pair,
    skipped_intermediates, thumbnail_ssim)
from light_video_enhancer.fi.rife import _pack_array as rife_pack, _unpack_array as rife_unpack
from light_video_enhancer.pipeline import VideoEnhancer
from light_video_enhancer.sr import create_sr_engine
from light_video_enhancer.sr._dxva_convert import bgr_to_nv12
from light_video_enhancer.sr.nvvfx_sr import (
    _pack_array as nvvfx_pack, _unpack_array as nvvfx_unpack)


class CoreTests(unittest.TestCase):
    def test_vendor_and_cross_vendor_codec_selection(self):
        self.assertEqual(_vendor_from_text("PCI\\VEN_10DE"), "nvidia")
        self.assertEqual(_vendor_from_text("Intel UHD Graphics"), "intel")
        self.assertEqual(_vendor_from_text("AMD Radeon"), "amd")
        self.assertEqual(
            choose_codec("auto", [GPUInfo("RTX", "nvidia")]),
            "h264_nvenc",
        )
        amd_expected = "h264_amf"
        intel_expected = "h264_mf" if sys.platform == "win32" else "mpeg4"
        self.assertEqual(choose_codec("auto", [GPUInfo("Radeon", "amd")]), amd_expected)
        self.assertEqual(choose_codec("auto", [GPUInfo("UHD", "intel")]), intel_expected)

    def test_codec_aliases_presets_and_format_fallbacks(self):
        self.assertEqual(canonical_codec("x264"), "libx264")
        self.assertEqual(canonical_codec("h265"), "libx265")
        self.assertEqual(canonical_codec("av1"), "libsvtav1")
        hevc = codec_candidates("libx265")
        self.assertEqual(hevc[0], "libx265")
        self.assertLess(hevc.index("hevc_nvenc"), hevc.index("h264_nvenc"))
        self.assertEqual(_codec_preset("libaom-av1", "balanced"), "5")
        self.assertEqual(_codec_preset("libsvtav1", "fast"), "9")
        self.assertEqual(_codec_preset("libx264", "p1"), "ultrafast")

    def test_config_validation(self):
        config = ProcessConfig(input_path="in.mp4", output_path="out.mp4")
        config.validate()
        config.encode.crf = 63
        config.validate()
        config.encode.crf = 64
        with self.assertRaises(ValueError):
            config.validate()
        config.encode.crf = 23
        config.duration = 0
        with self.assertRaises(ValueError):
            config.validate()
        config.duration = None
        config.sr_quality = "invalid"
        with self.assertRaises(ValueError):
            config.validate()
        config.sr_quality = "quality"
        config.validate()

    def test_output_name_is_sanitised(self):
        path = _auto_output("demo.mp4", 2.0, "blend", 2, "..MP4!", "lanczos")
        self.assertTrue(path.endswith("demo_x2_f2.mp4"), path)

    def test_nv12_conversion_supports_odd_dimensions(self):
        frame = np.zeros((63, 65, 3), dtype=np.uint8)
        data = bgr_to_nv12(frame)
        self.assertEqual(data.dtype, np.uint8)
        self.assertEqual(data.size, 66 * 64 * 3 // 2)

    def test_cpu_interpolators_return_expected_count_and_shape(self):
        first = np.zeros((48, 64, 3), dtype=np.uint8)
        second = first.copy()
        second[12:32, 20:40] = (20, 180, 250)
        for name in ("blend", "dis", "optical_flow"):
            with self.subTest(engine=name):
                engine = create_fi_engine(name, quality="fast")
                engine.initialize(64, 48, 3)
                frames = engine.interpolate(first, second)
                engine.release()
                self.assertEqual(len(frames), 2)
                self.assertTrue(all(frame.shape == first.shape for frame in frames))
                self.assertTrue(all(frame.dtype == np.uint8 for frame in frames))

    def test_rife_scene_detection_skips_static_and_avoids_crossfade(self):
        black = np.zeros((48, 64, 3), dtype=np.uint8)
        white = np.full_like(black, 255)
        textured = black.copy()
        textured[8:40, 12:52] = (60, 150, 220)
        shifted = np.roll(textured, 2, axis=1)
        self.assertEqual(classify_pair(textured, textured.copy()), PAIR_STATIC)
        self.assertEqual(classify_pair(black, white), PAIR_SCENE_CUT)
        self.assertEqual(classify_pair(textured, shifted), PAIR_NORMAL)
        self.assertGreater(thumbnail_ssim(textured, textured), 0.996)
        frames = skipped_intermediates(black, white, 4, PAIR_SCENE_CUT)
        self.assertEqual(len(frames), 3)
        np.testing.assert_array_equal(frames[0], black)
        np.testing.assert_array_equal(frames[1], black)
        np.testing.assert_array_equal(frames[2], white)

    def test_numpy_neutral_subprocess_array_protocol(self):
        source = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        for pack, unpack in ((rife_pack, rife_unpack), (nvvfx_pack, nvvfx_unpack)):
            with self.subTest(pack=pack.__module__):
                message = pack(source)
                serialised = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
                self.assertNotIn(b"numpy", serialised.lower())
                restored = unpack(pickle.loads(serialised))
                np.testing.assert_array_equal(restored, source)
                self.assertTrue(restored.flags.c_contiguous)

    def test_super_resolution_quality_mapping(self):
        self.assertIn("LOW", create_sr_engine("nvvfx", quality="fast").name)
        self.assertIn("HIGH", create_sr_engine("nvvfx", quality="quality").name)
        self.assertIn("ultra", create_sr_engine("realcugan", quality="ultra").name)
        classic = create_sr_engine("esrgan", quality="ultra", ncnn_gpu=1)
        classic.initialize(64, 48, 128, 96)
        self.assertIn("ESRGAN classic", classic.name)
        self.assertIn("GPU 1", classic.name)
        self.assertEqual(classic.batch_output_pixels, 64 * 48 * 16)
        fast = create_sr_engine("realesrgan", quality="fast")
        fast.initialize(64, 48, 128, 96)
        self.assertEqual((fast._model_name, fast._native_scale),
                         ("realesr-animevideov3", 2))
        balanced = create_sr_engine("realesrgan", quality="balanced")
        balanced.initialize(64, 48, 128, 96)
        self.assertEqual((balanced._model_name, balanced._native_scale),
                         ("realesr-animevideov3", 2))
        balanced.initialize(64, 48, 256, 192)
        self.assertEqual((balanced._model_name, balanced._native_scale),
                         ("realesrgan-x4plus-anime", 4))

    def test_ncnn_fallback_tuning_overrides_are_validated(self):
        old_jobs = os.environ.get("LVE_NCNN_JOBS_RIFE")
        old_tile = os.environ.get("LVE_NCNN_TILE_REALCUGAN")
        try:
            os.environ["LVE_NCNN_JOBS_RIFE"] = "2:3:2"
            os.environ["LVE_NCNN_TILE_REALCUGAN"] = "256"
            self.assertEqual(ncnn_jobs(1280, 720, engine="rife"), "2:3:2")
            self.assertEqual(ncnn_tile("realcugan"), 256)
            os.environ["LVE_NCNN_JOBS_RIFE"] = "broken"
            os.environ["LVE_NCNN_TILE_REALCUGAN"] = "16"
            self.assertEqual(ncnn_jobs(1280, 720, engine="rife"), "4:4:4")
            self.assertEqual(ncnn_tile("realcugan"), 0)
        finally:
            if old_jobs is None:
                os.environ.pop("LVE_NCNN_JOBS_RIFE", None)
            else:
                os.environ["LVE_NCNN_JOBS_RIFE"] = old_jobs
            if old_tile is None:
                os.environ.pop("LVE_NCNN_TILE_REALCUGAN", None)
            else:
                os.environ["LVE_NCNN_TILE_REALCUGAN"] = old_tile

    def test_esrgan_models_are_reported_available(self):
        caps = quick_capabilities()
        self.assertTrue(caps["ncnn_esrgan"])
        self.assertTrue(caps["ncnn_classic_esrgan"])

    def test_ncnn_chain_uses_output_aware_batch_budget(self):
        class DirectoryEngine:
            supports_batch = True
            supports_directory_batch = True

            def process_batch(self, frames):
                return frames

            def process_directory(self, input_dir, output_dir, count):
                return count

        config = ProcessConfig(fi_engine="rife_ncnn", sr_engine="realcugan",
                               fi_multiplier=2, sr_first=False)
        enhancer = VideoEnhancer(config)
        enhancer._src_width, enhancer._src_height = 1280, 720
        enhancer._dst_width, enhancer._dst_height = 2560, 1440
        enhancer._fi_engine = DirectoryEngine()
        enhancer._sr_engine = DirectoryEngine()
        enhancer._sr_engine.batch_output_pixels = 2560 * 1440
        enhancer._sr_engine.batch_output_size = (2560, 1440)
        self.assertTrue(enhancer._directory_chain_available())
        self.assertEqual(enhancer._batch_size(), 18)
        enhancer._config.sr_first = True
        self.assertTrue(enhancer._directory_chain_available())
        self.assertEqual(enhancer._batch_size(), 18)
        enhancer._sr_engine.batch_output_size = (5120, 2880)
        self.assertFalse(enhancer._directory_chain_available())

    def test_chunk_overlap_preserves_all_pairs(self):
        source = [np.full((1, 1, 1), index, np.uint8) for index in range(8)]
        chunks = list(VideoEnhancer._chunks(source, 3))
        pairs = []
        for chunk in chunks:
            pairs.extend((int(left[0, 0, 0]), int(right[0, 0, 0]))
                         for left, right in zip(chunk, chunk[1:]))
        self.assertEqual(pairs, [(index, index + 1) for index in range(7)])


if __name__ == "__main__":
    unittest.main()

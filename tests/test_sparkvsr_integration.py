import json
import unittest
from pathlib import Path
from unittest import mock

from light_video_enhancer.config import ProcessConfig
from light_video_enhancer.model_manager import MODEL_PACKS
from light_video_enhancer.pipeline import VideoEnhancer
from light_video_enhancer.sr.sparkvsr import SparkVSREngine, parse_reference_indices


class SparkVSRIntegrationTests(unittest.TestCase):
    def test_runtime_archive_contains_pinned_official_source(self):
        package = Path(__file__).parents[1] / "light_video_enhancer"
        archive = package / "external" / "sparkvsr_runtime.zip"
        import zipfile
        with zipfile.ZipFile(archive) as bundle:
            self.assertIn("sparkvsr_wrapper/infer.py", bundle.namelist())
            self.assertIn("SPARKVSR_LICENSE.txt", bundle.namelist())
            self.assertEqual(
                bundle.read("UPSTREAM_COMMIT.txt").decode().strip(),
                "a082284b80005bb5615c0f5f5f5ed66650b1b1e7")

    def test_model_pack_is_optional_large_and_complete(self):
        pack = next(item for item in MODEL_PACKS
                    if item["id"] == "sparkvsr-stage2")
        self.assertEqual(pack["remote_download_size"], 42199097809)
        self.assertEqual(len(pack["files"]), 21)
        self.assertIn("never auto-selected", pack["description"]["en-US"])

    def test_reference_indices_are_strict(self):
        self.assertEqual(parse_reference_indices("0, 8 16"), [0, 8, 16])
        for invalid in ("-1", "4,4", "8,4", "0,3"):
            with self.assertRaises(ValueError):
                parse_reference_indices(invalid)

    def test_native_scale_and_reference_pair_are_validated(self):
        engine = SparkVSREngine(reference_path="missing", reference_indices=[0])
        with mock.patch("light_video_enhancer.sr.sparkvsr.os.name", "nt"), \
             mock.patch("light_video_enhancer.sr.sparkvsr.sys.getwindowsversion", return_value=(10, 0)):
            with self.assertRaisesRegex(ValueError, "native 4x"):
                engine.initialize(320, 180, 640, 360)
        engine = SparkVSREngine(reference_path=None, reference_indices=[0])
        with mock.patch("light_video_enhancer.sr.sparkvsr.os.name", "nt"), \
             mock.patch("light_video_enhancer.sr.sparkvsr.sys.getwindowsversion", return_value=(10, 0)):
            with self.assertRaisesRegex(ValueError, "both a reference path and indices"):
                engine.initialize(320, 180, 1280, 720)

    def test_pipeline_rejects_non_native_scale_before_engine_init(self):
        enhancer = VideoEnhancer(ProcessConfig(
            scale=2, sr_engine="sparkvsr", fi_engine="none"))
        enhancer._src_width, enhancer._src_height = 320, 180
        with self.assertRaisesRegex(ValueError, "native 4x"):
            enhancer._calculate_geometry()


if __name__ == "__main__":
    unittest.main()

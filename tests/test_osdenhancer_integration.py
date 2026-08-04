import unittest
import zipfile
from pathlib import Path
from unittest import mock

from light_video_enhancer import model_manager
from light_video_enhancer.config import ProcessConfig
from light_video_enhancer.pipeline import VideoEnhancer
from light_video_enhancer.sr.osdenhancer import OSDEnhancerEngine


class OSDEnhancerIntegrationTests(unittest.TestCase):
    def test_runtime_is_pinned_and_apache_licensed(self):
        package = Path(model_manager.__file__).parent
        archive = package / "external" / "osdenhancer_runtime.zip"
        with zipfile.ZipFile(str(archive), "r") as bundle:
            self.assertIsNone(bundle.testzip())
            self.assertEqual(
                bundle.read("UPSTREAM_COMMIT.txt").decode().strip(),
                "64dd6e56331cf7ed44e987859d47fa26b57fa662")
            self.assertIn(
                "Apache License", bundle.read(
                    "OSDENHANCER_LICENSE.txt").decode("utf-8"))
            self.assertTrue(bundle.getinfo(
                "pipeline/OSDEnhancer_pipeline.py").file_size > 0)

    def test_model_pack_has_complete_hashes_and_exact_size(self):
        pack = next(item for item in model_manager.MODEL_PACKS
                    if item["id"] == "osdenhancer-v1")
        self.assertEqual(set(pack["files"]), set(pack["remote_hashes"]))
        self.assertEqual(pack["remote_download_size"], 12846839231)

    def test_native_joint_scale_is_enforced_before_start(self):
        engine = OSDEnhancerEngine()
        with mock.patch("light_video_enhancer.sr.osdenhancer.os.name", "nt"), \
                mock.patch(
                    "light_video_enhancer.sr.osdenhancer.sys.getwindowsversion",
                    return_value=(10, 0), create=True):
            with self.assertRaisesRegex(ValueError, "joint 4x/2x"):
                engine.initialize(128, 72, 256, 144)

    def test_joint_engine_doubles_natural_fps(self):
        enhancer = VideoEnhancer(ProcessConfig(
            input_path="in.mp4", output_path="out.mp4",
            scale=4, sr_engine="osdenhancer", fi_engine="none"))
        enhancer._src_fps = 29.97
        self.assertAlmostEqual(enhancer._natural_fps(), 59.94)


if __name__ == "__main__":
    unittest.main()

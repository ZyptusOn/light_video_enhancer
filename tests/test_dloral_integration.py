import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from light_video_enhancer import _env, model_manager
from light_video_enhancer.sr.dloral import DLoRALEngine


class DLoRALIntegrationTests(unittest.TestCase):
    def test_runtime_is_pinned_and_has_no_mmcv_import(self):
        package = Path(model_manager.__file__).parent
        archive = package / "external" / "dloral_runtime.zip"
        with zipfile.ZipFile(str(archive), "r") as bundle:
            self.assertIsNone(bundle.testzip())
            self.assertEqual(
                bundle.read("UPSTREAM_COMMIT.txt").decode().strip(),
                "e8a5574124dd18d7d6ea71d6974bab6705f6e1f4")
            self.assertIn("MIT License", bundle.read(
                "DLoRAL_LICENSE.txt").decode("utf-8"))
            cfr = bundle.read(
                "src/cross_frame_retrieval/cfr_main.py").decode("utf-8")
            model = bundle.read("src/DLoRAL_model.py").decode("utf-8")
            self.assertTrue(bundle.getinfo(
                "src/my_utils/devices.py").file_size > 0)
        self.assertNotIn("from mmcv", cfr)
        self.assertNotIn("from mmengine", cfr)
        self.assertIn("torchvision.ops", cfr)
        self.assertIn("LVE_DLORAL_SPYNET", model)
        self.assertIn("weights_only=False", model)

    def test_core_and_prompt_packs_have_complete_hashes(self):
        packs = {item["id"]: item for item in model_manager.MODEL_PACKS}
        core = packs["dloral-core"]
        prompt = packs["dloral-prompt"]
        self.assertEqual(set(core["files"]), set(core["remote_hashes"]))
        self.assertEqual(set(prompt["files"]), set(prompt["remote_hashes"]))
        self.assertGreater(core["remote_download_size"], 8 * 1024 ** 3)
        self.assertGreater(prompt["remote_download_size"], 5 * 1024 ** 3)

    def test_native_scale_and_minimum_size_are_enforced_before_start(self):
        engine = DLoRALEngine()
        with mock.patch("light_video_enhancer.sr.dloral.os.name", "nt"), \
                mock.patch("light_video_enhancer.sr.dloral.sys.getwindowsversion",
                           return_value=(10, 0), create=True):
            with self.assertRaisesRegex(ValueError, "native 4x"):
                engine.initialize(128, 128, 256, 256)
            with self.assertRaisesRegex(ValueError, "at least 512"):
                engine.initialize(100, 100, 400, 400)

    def test_environment_feature_can_select_dloral_runtime(self):
        environments = [{
            "exe": "C:/dloral/python.exe",
            "torch": True,
            "cuda": True,
            "dloral": True,
        }]
        with mock.patch.object(_env, "get_all_python_envs",
                               return_value=environments):
            self.assertEqual(
                _env.get_python_for_feature("dloral"),
                "C:/dloral/python.exe")


if __name__ == "__main__":
    unittest.main()

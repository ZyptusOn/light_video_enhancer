import os
import unittest
import zipfile
from unittest import mock
from light_video_enhancer import capabilities, model_manager
from light_video_enhancer.fi.vfimamba import VFIMambaEngine


class VFIMambaIntegrationTests(unittest.TestCase):
    def test_runtime_contains_pinned_sources_and_fallback(self):
        package = os.path.dirname(os.path.dirname(os.path.abspath(capabilities.__file__)))
        archive = os.path.join(package, "light_video_enhancer", "external", "vfimamba_runtime.zip")
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
            self.assertIn("Trainer_finetune.py", names)
            self.assertIn("model/feature_extractor.py", names)
            self.assertIn("vfimamba_selective_scan.py", names)
            self.assertEqual(bundle.read("VFIMAMBA_COMMIT.txt").decode().strip(),
                             "8df805eb054cd423e188d509210731cffe9438af")
            source = bundle.read("model/feature_extractor.py").decode()
            self.assertIn("PyTorch reference", source)

    def test_remote_model_pack_is_pinned_and_hashed(self):
        pack = next(item for item in model_manager.MODEL_PACKS if item["id"] == "vfimamba")
        self.assertEqual(pack["remote_download_size"], 331712554)
        self.assertIn("7c383874883191d240bcc9435590eecc573f1055", pack["remote_bases"]["official"])
        self.assertEqual(pack["remote_hashes"]["vfimamba/VFIMamba_S.pkl"],
                         "ddc1e07e5917f1bbd254ca77e077354cf822c9af6cacd1434136e86b9961acc7")

    def test_full_worker_config_matches_official_metadata(self):
        worker = os.path.join(os.path.dirname(os.path.abspath(capabilities.__file__)),
                              "fi", "_vfimamba_infer.py")
        with open(worker, "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("F=32, depth=[2, 2, 2, 3, 3]", source)
        self.assertNotIn("F=32, depth=[2, 2, 2, 4, 4]", source)

    def test_quality_profiles(self):
        self.assertEqual(VFIMambaEngine._QUALITY, {
            "fast": ("small", 0.5, False), "balanced": ("small", 0.0, False),
            "quality": ("full", 0.5, False), "ultra": ("full", 0.0, True)})

    def test_capability_requires_runtime_and_both_models(self):
        def model_exists(*parts):
            return parts in {("vfimamba", "VFIMamba_S.pkl"), ("vfimamba", "VFIMamba.pkl")}
        with mock.patch.object(capabilities, "model_file_exists", side_effect=model_exists), \
             mock.patch.object(capabilities, "pkg_file_exists",
                               side_effect=lambda *p: p == ("external", "vfimamba_runtime.zip")), \
             mock.patch.object(capabilities, "detect_gpus", return_value=[]):
            caps = capabilities.quick_capabilities()
        self.assertTrue(caps["vfimamba_model"])

    def test_missing_model_fails_before_process_start(self):
        engine = VFIMambaEngine(torch_python="python.exe")
        with mock.patch("light_video_enhancer.fi.vfimamba.get_model_dir", return_value="Z:\\missing"):
            with self.assertRaises(FileNotFoundError):
                engine.initialize(64, 64, 2)


if __name__ == "__main__":
    unittest.main()

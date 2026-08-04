import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from light_video_enhancer.sr.seedvr2 import SeedVR2Engine


class SeedVR2ProfileTests(unittest.TestCase):
    def _model_tree(self, root: Path):
        files = {
            "seedvr2-3b-fp8": (
                "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                "ema_vae_fp16.safetensors",
            ),
            "seedvr2-7b-q4": ("seedvr2_ema_7b-Q4_K_M.gguf",),
            "seedvr2-7b-sharp-q4": (
                "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
            ),
        }
        for pack, names in files.items():
            directory = root / pack
            directory.mkdir(parents=True)
            for name in names:
                (directory / name).touch()

    def _select(self, root: Path, quality: str):
        engine = SeedVR2Engine(quality=quality)
        with mock.patch(
                "light_video_enhancer.sr.seedvr2.get_model_dir",
                side_effect=lambda pack: str(root / pack)):
            model_dir = engine._select_model()
        return engine, model_dir

    def test_quality_profiles_select_expected_weight_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._model_tree(root)

            fast, model_dir = self._select(root, "fast")
            self.assertEqual(model_dir, str(root / "seedvr2-3b-fp8"))
            self.assertEqual(fast._model_family, "3b")
            self.assertEqual(fast._model_label, "3B FP8")
            self.assertEqual(fast.preferred_batch_size, 5)

            balanced, _ = self._select(root, "balanced")
            self.assertEqual(balanced._model_family, "3b")
            self.assertEqual(balanced.preferred_batch_size, 9)

            quality, _ = self._select(root, "quality")
            self.assertEqual(quality._model_label, "7B Q4")
            self.assertEqual(quality._model_family, "7b")
            self.assertEqual(quality.preferred_batch_size, 5)
            self.assertTrue(os.path.isabs(quality._dit_model))

            ultra, _ = self._select(root, "ultra")
            self.assertEqual(ultra._model_label, "7B Sharp Q4")
            self.assertEqual(ultra._model_family, "7b")
            self.assertEqual(ultra.preferred_batch_size, 5)

    def test_missing_optional_7b_weights_fall_back_to_3b(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._model_tree(root)
            (root / "seedvr2-7b-q4" /
             "seedvr2_ema_7b-Q4_K_M.gguf").unlink()
            (root / "seedvr2-7b-sharp-q4" /
             "seedvr2_ema_7b_sharp-Q4_K_M.gguf").unlink()

            for profile in ("quality", "ultra"):
                engine, _ = self._select(root, profile)
                self.assertEqual(engine._model_family, "3b")
                self.assertEqual(engine._model_label, "3B FP8")


if __name__ == "__main__":
    unittest.main()

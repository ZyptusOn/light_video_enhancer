import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from light_video_enhancer import model_manager
from light_video_enhancer.frontend_protocol import capabilities_payload


class ModelManagerTests(unittest.TestCase):
    def test_protocol_versions_are_explicit(self):
        self.assertEqual(capabilities_payload()["protocol_version"], 1)
        self.assertEqual(model_manager.list_model_packs()["protocol_version"], 1)

    def test_weight_catalog_covers_all_active_families(self):
        paths = {path.replace("\\", "/") for path in model_manager.model_weight_paths()}
        self.assertIn("fi/flownet.pkl", paths)
        self.assertIn("ncnn/rife/rife-v4.6/flownet.bin", paths)
        self.assertIn("ncnn/realcugan/models-se/up4x-denoise3x.bin", paths)
        self.assertIn("ncnn/realesrgan/models/realesrgan-x4plus.bin", paths)
        self.assertIn("ncnn/realesrgan/models/esrgan-x4.bin", paths)

    def test_install_status_and_remove_use_external_root(self):
        pack = next(item for item in model_manager.MODEL_PACKS if item["id"] == "rife-ncnn")
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "pack.zip"
            with zipfile.ZipFile(str(archive), "w") as bundle:
                for relative in pack["files"]:
                    bundle.writestr(relative, relative.encode("utf-8"))
            model_root = Path(temporary) / "models"
            with mock.patch.dict(os.environ, {"LVE_MODEL_DIR": str(model_root)}), \
                    mock.patch.object(model_manager, "_metadata", return_value={}):
                model_manager.install_model_archive("rife-ncnn", str(archive))
                payload = model_manager.list_model_packs()
                status = next(item for item in payload["packs"] if item["id"] == "rife-ncnn")
                self.assertEqual(status["status"], "downloaded")
                model_manager.remove_downloaded_pack("rife-ncnn")
                self.assertFalse(any(path.is_file() for path in model_root.rglob("*")))

    def test_archive_must_match_exact_file_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "bad.zip"
            with zipfile.ZipFile(str(archive), "w") as bundle:
                bundle.writestr("../outside.bin", b"bad")
            with mock.patch.dict(os.environ, {"LVE_MODEL_DIR": str(Path(temporary) / "models")}), \
                    mock.patch.object(model_manager, "_metadata", return_value={}):
                with self.assertRaises(ValueError):
                    model_manager.install_model_archive("rife-ncnn", str(archive))


if __name__ == "__main__":
    unittest.main()

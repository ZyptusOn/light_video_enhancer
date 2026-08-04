import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import build_exe
from light_video_enhancer import model_manager
from light_video_enhancer.frontend_protocol import capabilities_payload


class ModelManagerTests(unittest.TestCase):
    def test_protocol_versions_are_explicit(self):
        self.assertEqual(capabilities_payload()["protocol_version"], 1)
        self.assertEqual(model_manager.list_model_packs()["protocol_version"], 1)

    def test_weight_catalog_covers_all_active_families(self):
        paths = {path.replace("\\", "/") for path in model_manager.model_weight_paths()}
        self.assertIn("fi/flownet.pkl", paths)
        self.assertIn("fi/ema_vfi/ours_small_t.pkl", paths)
        self.assertIn(
            "flashvsr-v1.1/TCDecoder.ckpt", paths)
        self.assertIn(
            "seedvr2-3b-fp8/ema_vae_fp16.safetensors", paths)
        self.assertIn("ncnn/rife/rife-v4.6/flownet.bin", paths)
        self.assertIn(
            "seedvr2-7b-q4/seedvr2_ema_7b-Q4_K_M.gguf", paths)
        self.assertIn("seedvr2-7b-sharp-q4/"
                      "seedvr2_ema_7b_sharp-Q4_K_M.gguf", paths)
        self.assertIn("ncnn/realcugan/models-se/up4x-denoise3x.bin", paths)
        self.assertIn("ncnn/realesrgan/models/realesrgan-x4plus.bin", paths)
        self.assertIn("ncnn/realesrgan/models/esrgan-x4.bin", paths)
        self.assertIn("ncnn/ifrnet/IFRNet_L_Vimeo90K/ifrnet.bin", paths)
        self.assertIn("ncnn/span/spanx4_ch52.bin", paths)

    def test_light_build_omits_weights_and_modern_runtime_is_targeted(self):
        modern = build_exe._data_files("light", "modern")
        modern_sources = [item.split(os.pathsep, 1)[0] for item in modern[1::2]]
        relative = [os.path.relpath(path, build_exe.PACKAGE_DIR).replace("\\", "/")
                    for path in modern_sources]
        self.assertIn("sr/_flashvsr_infer.py", relative)
        self.assertIn("sr/_seedvr2_infer.py", relative)
        self.assertIn("external/flashvsr_runtime.zip", relative)
        self.assertIn("external/seedvr2_runtime.zip", relative)
        self.assertFalse(any(build_exe._is_model_weight(path) for path in relative))

        win7 = build_exe._data_files("full", "win7")
        win7_sources = [item.split(os.pathsep, 1)[0] for item in win7[1::2]]
        win7_relative = [os.path.relpath(path, build_exe.PACKAGE_DIR).replace("\\", "/")
                         for path in win7_sources]
        self.assertNotIn("sr/_flashvsr_infer.py", win7_relative)
        self.assertNotIn("sr/_seedvr2_infer.py", win7_relative)
        self.assertFalse(any(path.startswith("external/") for path in win7_relative))

    def test_heavy_workers_protect_framed_stdout_and_native_crashes(self):
        package = Path(model_manager.__file__).parent
        for worker in ("_flashvsr_infer.py", "_seedvr2_infer.py"):
            source = (package / "sr" / worker).read_text(encoding="utf-8")
            self.assertIn("redirect_stdout(sys.stderr)", source)
            self.assertIn("SetErrorMode(0x0001 | 0x0002)", source)
        seed_adapter = (package / "sr" / "seedvr2.py").read_text(
            encoding="utf-8")
        self.assertIn(
            'child_env["PYTORCH_CUDA_ALLOC_CONF"] = '
            '"backend:cudaMallocAsync"', seed_adapter)
        for adapter in ("flashvsr.py", "seedvr2.py"):
            source = (package / "sr" / adapter).read_text(encoding="utf-8")
            self.assertIn('child_env["PYTHONIOENCODING"] = "utf-8"', source)

    def test_seed_runtime_contains_required_non_python_assets(self):
        package = Path(model_manager.__file__).parent
        archive = package / "external" / "seedvr2_runtime.zip"
        with zipfile.ZipFile(str(archive), "r") as bundle:
            names = set(bundle.namelist())
        self.assertIn(
            "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml", names)
        self.assertIn("configs_3b/main.yaml", names)
        self.assertIn("configs_7b/main.yaml", names)
        self.assertIn("neg_emb.pt", names)
        self.assertIn("pos_emb.pt", names)

    def test_remote_model_hashes_cover_every_download(self):
        for pack in model_manager.MODEL_PACKS:
            if not pack.get("downloads"):
                continue
            self.assertEqual(set(pack["files"]), set(pack["downloads"]))
            self.assertEqual(set(pack["files"]), set(pack["remote_hashes"]))
            self.assertTrue(all(len(value) == 64 for value in pack["remote_hashes"].values()))

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

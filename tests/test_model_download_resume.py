import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from light_video_enhancer import model_manager


def _remote_pack(pack_id, payload):
    relative = "%s/model.bin" % pack_id
    return model_manager._remote_pack(
        pack_id, pack_id, "测试", "Test", "测试", "Test", [relative],
        downloads={relative: "model.bin"},
        official_base="https://example.invalid/models",
        mirror_base="https://example.invalid/models",
        download_size=len(payload),
        hashes={relative: hashlib.sha256(payload).hexdigest()})


class _Response:
    def __init__(self, status, headers, data):
        self.status = status
        self.headers = headers
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        if not self._data:
            return b""
        value, self._data = self._data[:size], self._data[size:]
        return value


class ModelDownloadResumeTests(unittest.TestCase):
    def test_resume_truncates_mirror_bytes_past_content_range(self):
        payload = b"0123456789abcdef"
        pack = _remote_pack("resume-test", payload)
        response = _Response(
            206,
            {"Content-Range": "bytes 4-15/16", "Content-Length": "16"},
            payload[4:] + b"junk")

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(model_manager, "MODEL_PACKS", (pack,)), \
                mock.patch.object(model_manager, "get_model_root",
                                  return_value=temporary), \
                mock.patch.object(model_manager.urllib.request, "urlopen",
                                  return_value=response):
            part = (Path(temporary) / ".downloads" / "resume-test" /
                    "model.bin.part")
            part.parent.mkdir(parents=True)
            part.write_bytes(payload[:4])
            model_manager._download_remote_pack(pack, "github", None, None)
            installed = Path(temporary) / "resume-test" / "model.bin"
            self.assertEqual(installed.read_bytes(), payload)
            self.assertFalse(part.exists())

    def test_resume_replaces_partial_when_range_is_ignored(self):
        payload = b"complete model"
        pack = _remote_pack("restart-test", payload)
        response = _Response(
            200, {"Content-Length": str(len(payload))}, payload)

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(model_manager, "MODEL_PACKS", (pack,)), \
                mock.patch.object(model_manager, "get_model_root",
                                  return_value=temporary), \
                mock.patch.object(model_manager.urllib.request, "urlopen",
                                  return_value=response):
            part = (Path(temporary) / ".downloads" / "restart-test" /
                    "model.bin.part")
            part.parent.mkdir(parents=True)
            part.write_bytes(b"stale")
            model_manager._download_remote_pack(pack, "github", None, None)
            installed = Path(temporary) / "restart-test" / "model.bin"
            self.assertEqual(installed.read_bytes(), payload)

    def test_completed_partial_is_installed_without_network(self):
        payload = b"already complete"
        pack = _remote_pack("complete-test", payload)
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(model_manager, "MODEL_PACKS", (pack,)), \
                mock.patch.object(model_manager, "get_model_root",
                                  return_value=temporary), \
                mock.patch.object(model_manager.urllib.request, "urlopen") as open_url:
            part = (Path(temporary) / ".downloads" / "complete-test" /
                    "model.bin.part")
            part.parent.mkdir(parents=True)
            part.write_bytes(payload)
            model_manager._download_remote_pack(pack, "github", None, None)
            open_url.assert_not_called()
            installed = Path(temporary) / "complete-test" / "model.bin"
            self.assertEqual(installed.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()

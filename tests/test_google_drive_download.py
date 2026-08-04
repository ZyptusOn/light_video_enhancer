import unittest
from urllib.parse import parse_qs, urlparse
from unittest import mock

from light_video_enhancer import model_manager


class GoogleDriveDownloadTests(unittest.TestCase):
    def test_confirmation_form_is_parsed_and_encoded(self):
        document = """
        <form id="download-form"
              action="https://drive.usercontent.google.com/download"
              method="get">
          <input name="id" value="file-id">
          <input name="export" value="download">
          <input name="confirm" value="t">
          <input name="uuid" value="one two">
        </form>
        """
        result = model_manager._drive_confirmation_url(document)
        parsed = urlparse(result)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.hostname, "drive.usercontent.google.com")
        self.assertEqual(parse_qs(parsed.query), {
            "id": ["file-id"],
            "export": ["download"],
            "confirm": ["t"],
            "uuid": ["one two"],
        })

    def test_confirmation_form_rejects_untrusted_action(self):
        document = """
        <form id="download-form" action="https://example.com/steal">
          <input name="id" value="x"><input name="export" value="download">
          <input name="confirm" value="t"><input name="uuid" value="x">
        </form>
        """
        with self.assertRaises(ValueError):
            model_manager._drive_confirmation_url(document)

    def test_confirmed_quota_page_is_rejected(self):
        confirmation = """
        <form id="download-form"
              action="https://drive.usercontent.google.com/download">
          <input name="id" value="file-id">
          <input name="export" value="download">
          <input name="confirm" value="t">
          <input name="uuid" value="uuid">
        </form>
        """.encode()

        class Response:
            status = 200

            def __init__(self, content_type, data):
                self.headers = {"Content-Type": content_type}
                self._data = data

            def read(self, size=-1):
                value, self._data = self._data[:size], self._data[size:]
                return value

            def close(self):
                pass

        first = Response("text/html; charset=utf-8", confirmation)
        second = Response(
            "text/html; charset=utf-8",
            b"<title>Google Drive - Quota exceeded</title>Too many users")
        opener = mock.MagicMock()
        opener.open.side_effect = [first, second]
        with mock.patch.object(model_manager.urllib.request, "build_opener",
                               return_value=opener):
            with self.assertRaisesRegex(IOError, "quota exceeded"):
                model_manager._open_remote(
                    "https://drive.google.com/uc?id=file-id",
                    {"User-Agent": "test"}, 1)

    def test_remote_filename_may_be_an_explicit_https_url(self):
        pack = model_manager._remote_pack(
            "drive", "drive", "测试", "Test", "测试", "Test",
            ["drive/model.pkl"],
            downloads={
                "drive/model.pkl":
                    "https://drive.google.com/uc?export=download&id=abc",
            },
            official_base="https://unused.invalid",
            mirror_base="https://unused.invalid",
            hashes={"drive/model.pkl": "0" * 64})
        self.assertEqual(
            model_manager._remote_file_url(
                pack, "drive/model.pkl", "github", None),
            "https://drive.google.com/uc?export=download&id=abc")

    def test_explicit_url_uses_safe_temporary_filename(self):
        self.assertEqual(
            model_manager._remote_part_filename(
                "https://www.dropbox.com/scl/fi/key/model.pkl?dl=1",
                "dloral-core/model.pkl"),
            "model.pkl")
        self.assertEqual(
            model_manager._remote_part_filename(
                "../unsafe.bin", "pack/fallback.bin"),
            "unsafe.bin")


if __name__ == "__main__":
    unittest.main()

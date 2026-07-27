import os
import unittest
from unittest import mock

from light_video_enhancer import i18n
from light_video_enhancer.cli_app import interactive_arguments


class LanguageAndCliTests(unittest.TestCase):
    def setUp(self):
        self.original_language = os.environ.get("LVE_LANG")

    def tearDown(self):
        if self.original_language is None:
            os.environ.pop("LVE_LANG", None)
        else:
            os.environ["LVE_LANG"] = self.original_language
        i18n.set_language(self.original_language)

    def test_language_aliases(self):
        self.assertEqual(i18n.normalize_language("zh-Hans"), "zh-CN")
        self.assertEqual(i18n.normalize_language("en-GB"), "en-US")

    def test_language_follows_system_when_not_overridden(self):
        with mock.patch.object(
                i18n, "system_language_name", return_value="zh-CN"):
            self.assertEqual(i18n.normalize_language(None), "zh-CN")
        with mock.patch.object(
                i18n, "system_language_name", return_value="en-US"):
            self.assertEqual(i18n.normalize_language(None), "en-US")

    def test_interactive_cli_can_exit_without_loading_gui(self):
        i18n.set_language("en-US")
        with mock.patch("builtins.input", return_value="q"):
            self.assertIsNone(interactive_arguments())

    def test_interactive_nvvfx_rife_arguments_complete_after_overwrite(self):
        responses = iter([
            __file__, "", "", "nvvfx", "", "rife",
            "ultra", "ultra", "av1_nvenc", "y", "y",
        ])
        with mock.patch("builtins.input", side_effect=lambda prompt: next(responses)):
            arguments = interactive_arguments()
        self.assertIn("--overwrite", arguments)
        self.assertEqual(arguments[arguments.index("--sr-engine") + 1], "nvvfx")
        self.assertEqual(arguments[arguments.index("--fi-engine") + 1], "rife")
        self.assertEqual(arguments[arguments.index("--codec") + 1], "av1_nvenc")


if __name__ == "__main__":
    unittest.main()

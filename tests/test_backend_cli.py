import ast
import contextlib
import io
import pathlib
import unittest

from light_video_enhancer import backend_main


ROOT = pathlib.Path(__file__).resolve().parents[1]


class StandaloneBackendTests(unittest.TestCase):
    def test_backend_entry_has_no_gui_or_tk_import(self):
        source = (ROOT / "light_video_enhancer" / "backend_main.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        self.assertFalse(any(
            module == "tkinter" or module.endswith(".gui")
            for module in modules))

        launcher = (ROOT / "backend_launcher.py").read_text(encoding="utf-8")
        self.assertIn("light_video_enhancer.backend_main", launcher)
        self.assertNotIn("light_video_enhancer.__main__", launcher)

    def test_english_help_is_grouped_and_has_examples(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                backend_main.main(["--language", "en-US", "--help"])
        self.assertEqual(raised.exception.code, 0)
        rendered = output.getvalue()
        self.assertIn("Processing and encoding options", rendered)
        self.assertIn("Standalone commands:", rendered)
        self.assertIn("--capabilities-json", rendered)
        self.assertIn("Examples:", rendered)

    def test_chinese_help_is_localized(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                backend_main.main(["--language", "zh-CN", "--help"])
        self.assertEqual(raised.exception.code, 0)
        rendered = output.getvalue()
        self.assertIn("处理与编码选项", rendered)
        self.assertIn("独立命令：", rendered)
        self.assertIn("常用示例：", rendered)


if __name__ == "__main__":
    unittest.main()

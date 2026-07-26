import io
import logging
import unittest

from light_video_enhancer._logging import get_logger, set_gui_handler
from light_video_enhancer.i18n import set_language
from light_video_enhancer.log_i18n import translate_log_template


class RuntimeLogLocalizationTests(unittest.TestCase):
    def tearDown(self):
        set_language(None)
        set_gui_handler(None)

    def test_english_log_templates_and_errors(self):
        set_language("en-US")
        self.assertEqual(
            translate_log_template("输入: %dx%d @ %.3f fps, %s 帧"),
            "Input: %dx%d @ %.3f fps, %s frames",
        )
        self.assertEqual(
            translate_log_template("RIFE 插帧倍率至少为 2"),
            "RIFE interpolation multiplier must be at least 2",
        )

    def test_handler_filter_localizes_formatted_records(self):
        set_language("en-US")
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        set_gui_handler(handler)

        get_logger("test").info("输入: %dx%d @ %.3f fps, %s 帧", 1280, 720, 30.0, 15)

        self.assertEqual(
            output.getvalue().strip(),
            "[INFO] Input: 1280x720 @ 30.000 fps, 15 frames",
        )


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest

import numpy as np

from light_video_enhancer.ffmpeg_bridge import (
    FFmpegVideoDecoder,
    FFmpegVideoEncoder,
    encoder_is_available,
    worker_is_loadable,
)


@unittest.skipUnless(os.name == "nt", "the bundled worker is Windows-only")
class FFmpegBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not worker_is_loadable():
            raise unittest.SkipTest("bundled FFmpeg worker cannot be loaded")

    def _round_trip(self, codec):
        with tempfile.TemporaryDirectory(prefix="lve-test-") as directory:
            safe_name = codec.replace("-", "_")
            path = os.path.join(directory, safe_name + ".mp4")
            encoder = FFmpegVideoEncoder(
                path, 64, 64, 10.0, codec=codec, preset="fast", crf=28
            )
            encoder.open()
            for index in range(6):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                frame[:, :, 0] = index * 30
                frame[8:24, 8 + index:24 + index, 1] = 255
                encoder.encode(frame)
            encoder.close()
            with FFmpegVideoDecoder(path, hardware="cpu") as decoder:
                self.assertAlmostEqual(decoder.fps, 10.0, places=3)
                self.assertEqual(decoder.total_frames, 6)
                frames = list(decoder)
            self.assertEqual(len(frames), 6)
            self.assertTrue(all(frame.shape == (64, 64, 3) for frame in frames))

    def test_bundled_software_codec_round_trips(self):
        for codec in ("libx264", "libx265", "libsvtav1", "libaom-av1"):
            with self.subTest(codec=codec):
                self.assertTrue(encoder_is_available(codec), codec)
                self._round_trip(codec)

    def test_mpeg4_round_trip_is_frame_accurate(self):
        self._round_trip("mpeg4")

    def test_media_foundation_round_trip_is_frame_accurate(self):
        if not encoder_is_available("h264_mf"):
            self.skipTest("Media Foundation H.264 is unavailable")
        self._round_trip("h264_mf")


if __name__ == "__main__":
    unittest.main()

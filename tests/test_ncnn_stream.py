import os
import shutil
import unittest

import numpy as np

from light_video_enhancer._ncnn_directory_stream import NcnnDirectoryStream


class NcnnDirectoryStreamTests(unittest.TestCase):
    def test_pipeline_preserves_order_and_owns_workspaces(self):
        chunks = [
            [np.full((4, 6, 3), index, np.uint8)]
            for index in range(4)
        ]
        processed = []

        def process(work, input_dir, count):
            self.assertTrue(os.path.isfile(
                os.path.join(input_dir, "00000000.png")))
            output = os.path.join(work, "output")
            os.makedirs(output)
            processed.append(int(os.path.basename(work).split("_")[-1], 16)
                             if False else count)
            return output, count

        workspaces = []
        with NcnnDirectoryStream(chunks, process) as stream:
            jobs = list(stream)
            workspaces = [job.work for job in jobs]
            self.assertEqual([job.sequence for job in jobs], [0, 1, 2, 3])
            self.assertEqual([job.output_count for job in jobs], [1, 1, 1, 1])
            for job in jobs:
                self.assertTrue(os.path.isdir(job.work))
                shutil.rmtree(job.work)
        self.assertEqual(processed, [1, 1, 1, 1])
        self.assertTrue(all(not os.path.exists(path) for path in workspaces))

    def test_worker_error_is_forwarded_and_workspace_removed(self):
        chunks = [[np.zeros((2, 2, 3), np.uint8)]]
        work = []

        def fail(path, _input, _count):
            work.append(path)
            raise ValueError("expected")

        with self.assertRaisesRegex(ValueError, "expected"):
            with NcnnDirectoryStream(chunks, fail) as stream:
                list(stream)
        self.assertTrue(work)
        self.assertFalse(os.path.exists(work[0]))


if __name__ == "__main__":
    unittest.main()

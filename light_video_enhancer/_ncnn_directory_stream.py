"""Three-stage streaming fallback for directory-oriented NCNN executables.

The portable upstream tools only accept image files.  This stream keeps one
Vulkan stage in flight while the producer decodes/writes the next input batch
and the consumer reads/encodes the previous output batch.  Only the single GPU
worker calls NCNN, so two model processes never compete for the same device.
"""

import os
import queue
import shutil
import tempfile
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from ._image_batch import write_frames


@dataclass
class DirectoryJob:
    sequence: int
    input_count: int
    work: str
    input_dir: str
    output_dir: str = ""
    output_count: int = 0
    error: Optional[BaseException] = None


DirectoryProcessor = Callable[[str, str, int], Tuple[str, int]]


class NcnnDirectoryStream:
    """Overlap PNG preparation, a single NCNN GPU worker, and consumption."""

    def __init__(self, chunks: Iterable[List[np.ndarray]],
                 processor: DirectoryProcessor, queue_size: int = 2):
        self._chunks = iter(chunks)
        self._processor = processor
        self._prepared = queue.Queue(maxsize=max(1, queue_size))
        self._completed = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._producer = threading.Thread(
            target=self._produce, name="lve-ncnn-prepare", daemon=True)
        self._gpu = threading.Thread(
            target=self._process, name="lve-ncnn-gpu", daemon=True)
        self._producer.start()
        self._gpu.start()

    def _put(self, target: queue.Queue, value) -> bool:
        while not self._stop.is_set():
            try:
                target.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            for sequence, frames in enumerate(self._chunks):
                if self._stop.is_set():
                    break
                work = tempfile.mkdtemp(prefix="lve_ncnn_chain_")
                input_dir = os.path.join(work, "input")
                try:
                    write_frames(frames, input_dir, "NCNN 管线")
                except BaseException:
                    shutil.rmtree(work, ignore_errors=True)
                    raise
                job = DirectoryJob(sequence, len(frames), work, input_dir)
                if not self._put(self._prepared, job):
                    shutil.rmtree(work, ignore_errors=True)
                    break
        except BaseException as exc:
            self._put(self._prepared, DirectoryJob(
                -1, 0, "", "", error=exc))
        finally:
            self._put(self._prepared, None)

    def _process(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    job = self._prepared.get(timeout=0.1)
                except queue.Empty:
                    continue
                if job is None:
                    break
                if job.error is not None:
                    self._put(self._completed, job)
                    break
                try:
                    job.output_dir, job.output_count = self._processor(
                        job.work, job.input_dir, job.input_count)
                except BaseException as exc:
                    job.error = exc
                if not self._put(self._completed, job):
                    shutil.rmtree(job.work, ignore_errors=True)
                    break
                if job.error is not None:
                    break
        finally:
            self._put(self._completed, None)

    def __iter__(self) -> Iterator[DirectoryJob]:
        while True:
            try:
                job = self._completed.get(timeout=0.1)
            except queue.Empty:
                if not self._gpu.is_alive():
                    raise RuntimeError("NCNN 流水线程意外退出")
                continue
            if job is None:
                return
            if job.error is not None:
                if job.work:
                    shutil.rmtree(job.work, ignore_errors=True)
                raise job.error
            yield job

    @staticmethod
    def _cleanup_queue(target: queue.Queue) -> None:
        while True:
            try:
                job = target.get_nowait()
            except queue.Empty:
                return
            if isinstance(job, DirectoryJob) and job.work:
                shutil.rmtree(job.work, ignore_errors=True)

    def close(self) -> None:
        self._stop.set()
        self._cleanup_queue(self._prepared)
        self._cleanup_queue(self._completed)
        for target in (self._prepared, self._completed):
            try:
                target.put_nowait(None)
            except queue.Full:
                pass
        self._producer.join(timeout=5)
        self._gpu.join(timeout=5)
        self._cleanup_queue(self._prepared)
        self._cleanup_queue(self._completed)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

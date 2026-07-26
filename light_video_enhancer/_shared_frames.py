"""Small cross-version shared-memory and framed-pipe helpers.

The shared-memory payload contains only raw contiguous arrays. Control
messages remain tiny pickles, so Python/NumPy versions may differ between the
packaged GUI and an external CUDA environment.
"""

import pickle
import queue
import struct
import threading
from multiprocessing import shared_memory
from typing import Any, Dict, Optional, Tuple

import numpy as np


class SharedNDArray:
    """Own or attach to a named shared-memory NumPy array."""

    def __init__(self, memory, shape: Tuple[int, ...], dtype, owner: bool):
        self.memory = memory
        self.shape = tuple(int(part) for part in shape)
        self.dtype = np.dtype(dtype)
        self.owner = owner
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=memory.buf)

    @classmethod
    def create(cls, shape: Tuple[int, ...], dtype=np.uint8):
        normalised = tuple(int(part) for part in shape)
        data_type = np.dtype(dtype)
        size = int(np.prod(normalised, dtype=np.int64)) * data_type.itemsize
        memory = shared_memory.SharedMemory(create=True, size=max(1, size))
        return cls(memory, normalised, data_type, True)

    @classmethod
    def attach(cls, descriptor: Dict[str, Any]):
        memory = shared_memory.SharedMemory(name=str(descriptor["name"]), create=False)
        return cls(memory, tuple(descriptor["shape"]), descriptor["dtype"], False)

    def descriptor(self) -> Dict[str, Any]:
        return {
            "name": self.memory.name,
            "shape": self.shape,
            "dtype": self.dtype.str,
        }

    def close(self) -> None:
        self.array = None
        try:
            self.memory.close()
        finally:
            if self.owner:
                try:
                    self.memory.unlink()
                except FileNotFoundError:
                    pass


def close_process_pipes(process) -> None:
    """Close every parent-side subprocess pipe after reader threads stop."""
    if process is None:
        return
    for name in ("stdin", "stdout", "stderr"):
        pipe = getattr(process, name, None)
        if pipe is None or getattr(pipe, "closed", False):
            continue
        try:
            pipe.close()
        except OSError:
            pass


def write_framed(pipe, value: Any) -> None:
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    pipe.write(struct.pack("!I", len(data)))
    pipe.write(data)
    pipe.flush()


def read_framed(pipe) -> Any:
    raw_length = pipe.read(4)
    if len(raw_length) != 4:
        raise EOFError("subprocess pipe closed")
    length = struct.unpack("!I", raw_length)[0]
    data = bytearray()
    while len(data) < length:
        chunk = pipe.read(length - len(data))
        if not chunk:
            raise EOFError("subprocess message incomplete")
        data.extend(chunk)
    return pickle.loads(bytes(data))


class FramedPipeReader:
    """One persistent reader thread providing timeout-aware message reads."""

    def __init__(self, pipe, name: str):
        self._pipe = pipe
        self._queue = queue.Queue()
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                self._queue.put((True, read_framed(self._pipe)))
        except BaseException as exc:
            self._error = exc
            self._queue.put((False, exc))

    def read(self, timeout: float) -> Any:
        try:
            ok, value = self._queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("subprocess response timed out") from exc
        if not ok:
            raise EOFError(str(value)) from value
        return value

    def join(self, timeout: float = 1.0) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

"""
NV-VFX VSR subprocess 推理脚本 — 由 nvvfx_sr.py 启动，作为持久化服务运行。

通信协议:
  初始化:
    stdin  ← pickle: args_dict  {"src_w","src_h","dst_w","dst_h","quality"}
    stdin  ← pickle: []         (握手)
    stdout → pickle: []

  循环:
    stdin  ← pickle: [frame0, frame1, ...]  每帧是 HWC uint8 numpy
    stdout → pickle: [result0, result1, ...]
"""

import sys
import pickle
import struct
import numpy as np
import torch
from nvvfx import VideoSuperRes


def _read_pickle():
    raw_len = sys.stdin.buffer.read(4)
    if len(raw_len) < 4:
        raise EOFError("stdin closed")
    msg_len = struct.unpack('!I', raw_len)[0]
    data = bytearray()
    while len(data) < msg_len:
        chunk = sys.stdin.buffer.read(msg_len - len(data))
        if not chunk:
            raise EOFError("stdin data incomplete")
        data.extend(chunk)
    return pickle.loads(bytes(data))


def _write_pickle(obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.write(struct.pack('!I', len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def main():
    try:
        args = _read_pickle()
        _read_pickle()
    except Exception as e:
        _write_pickle({"error": str(e)})
        sys.exit(1)

    try:
        if not torch.cuda.is_available():
            _write_pickle({"error": "CUDA not available"})
            sys.exit(1)

        vsr = VideoSuperRes(quality=args["quality"])
        vsr.input_width = args["src_w"]
        vsr.input_height = args["src_h"]
        vsr.output_width = args["dst_w"]
        vsr.output_height = args["dst_h"]
        vsr.load()
    except Exception as e:
        _write_pickle({"error": str(e)})
        sys.exit(1)

    _write_pickle([])

    while True:
        try:
            frames = _read_pickle()
        except EOFError:
            break
        except Exception as e:
            _write_pickle({"error": str(e)})
            break

        try:
            results = []
            for frame in frames:
                bgr = np.ascontiguousarray(frame)
                t = torch.from_numpy(bgr).cuda().permute(2, 0, 1).contiguous().float().div_(255.0)
                out = vsr.run(t)
                r = torch.from_dlpack(out.image)
                r = r.permute(1, 2, 0).mul_(255.0).clamp_(0.0, 255.0).to(torch.uint8)
                results.append(r.cpu().numpy())
            _write_pickle(results)
        except Exception as e:
            _write_pickle({"error": str(e)})
            break


if __name__ == "__main__":
    main()

"""
RIFE v4.25 subprocess 推理脚本 — 由 rife.py 启动，作为持久化服务运行。

通信协议:
  初始化阶段:
    stdin  ← pickle: args_dict  (model_path, fp16)
    stdin  ← pickle: []         (空任务列表，握手用)
    stdout → pickle: []         (握手确认)

  推理循环 (重复至父进程关闭 stdin):
    stdin  ← pickle: [(frame0, frame1, t, pad_w, pad_h, scale), ...]
    stdout → pickle: [[np.ndarray, ...], ...]

模型定义来自同目录下的 _rife_model.py + warplayer.py。
"""

import sys
import os
import pickle
import struct
import numpy as np
import torch
import torch.nn.functional as F

_sys_path_0 = os.path.dirname(os.path.abspath(__file__))
if _sys_path_0 not in sys.path:
    sys.path.insert(0, _sys_path_0)

from _rife_model import FlownetCas


def _read_pickle():
    raw_len = sys.stdin.buffer.read(4)
    if len(raw_len) < 4:
        raise EOFError("stdin 关闭")
    msg_len = struct.unpack('!I', raw_len)[0]
    data = bytearray()
    while len(data) < msg_len:
        chunk = sys.stdin.buffer.read(msg_len - len(data))
        if not chunk:
            raise EOFError("stdin 数据不完整")
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
        tasks = _read_pickle()
    except Exception as e:
        _write_pickle({"error": f"读取输入失败: {e}"})
        sys.exit(1)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        use_fp16 = args.get("fp16", True) and torch.cuda.is_available()

        model = FlownetCas().to(device).eval()

        model_path = args.get("model_path", "")
        loaded = False
        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=device)
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            loaded = True

        if not loaded:
            _write_pickle({"error": f"模型权重未找到: {model_path}"})
            sys.exit(1)

        if use_fp16:
            model.half()
    except Exception as e:
        _write_pickle({"error": str(e)})
        sys.exit(1)

    all_results = []
    for task in tasks:
        frame0, frame1, timestep, pad_w, pad_h, scale = task
        result = _process_one(device, use_fp16, model, frame0, frame1, timestep, pad_w, pad_h, scale)
        if isinstance(result, dict):
            _write_pickle(result)
            sys.exit(1)
        all_results.append(result)
    _write_pickle(all_results)

    while True:
        try:
            tasks = _read_pickle()
        except EOFError:
            break
        except Exception as e:
            _write_pickle({"error": f"读取任务失败: {e}"})
            break

        try:
            all_results = []
            for task in tasks:
                frame0, frame1, timestep, pad_w, pad_h, scale = task
                result = _process_one(device, use_fp16, model, frame0, frame1, timestep, pad_w, pad_h, scale)
                if isinstance(result, dict):
                    _write_pickle(result)
                    sys.exit(1)
                all_results.append(result)
            _write_pickle(all_results)
        except Exception as e:
            _write_pickle({"error": str(e)})
            break


def _process_one(device, use_fp16, model, frame0, frame1, timestep, pad_w, pad_h, scale):
    try:
        i0 = torch.from_numpy(frame0).to(device, non_blocking=True)
        i0 = i0.permute(2, 0, 1).unsqueeze(0)
        i1 = torch.from_numpy(frame1).to(device, non_blocking=True)
        i1 = i1.permute(2, 0, 1).unsqueeze(0)

        if use_fp16:
            i0 = i0.half() / 255.0
            i1 = i1.half() / 255.0
        else:
            i0 = i0.float() / 255.0
            i1 = i1.float() / 255.0

        if pad_w > 0 or pad_h > 0:
            i0 = F.pad(i0, (0, pad_w, 0, pad_h))
            i1 = F.pad(i1, (0, pad_w, 0, pad_h))

        with torch.no_grad():
            pred = model.inference(i0, i1, timestep, scale)

        h, w = frame0.shape[0], frame0.shape[1]
        pred = pred[:, :, :h, :w]

        out = (
            pred[0].float().permute(1, 2, 0)
            .clamp(0, 1).mul(255).byte().cpu().numpy()
        )

        del i0, i1, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return out
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    main()

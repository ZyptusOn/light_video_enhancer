"""Build the optional DLoRAL runtime from a pinned upstream checkout."""

import argparse
import os
import zipfile
from pathlib import Path


PINNED_COMMIT = "e8a5574124dd18d7d6ea71d6974bab6705f6e1f4"

_MMCV_IMPORTS = """from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from mmengine import MMLogger, print_log
from mmengine.runner import load_checkpoint
from mmengine.model.weight_init import constant_init
"""

_TORCH_COMPAT = """from torchvision.ops import deform_conv2d as _torchvision_deform_conv2d

BaseModule = nn.Module

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding)
        self.activate = nn.ReLU(inplace=True) if act_cfg is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.activate is not None:
            x = self.activate(x)
        return x

class ModulatedDeformConv2d(nn.Conv2d):
    def __init__(self, *args, deform_groups=1, **kwargs):
        self.deform_groups = deform_groups
        super().__init__(*args, **kwargs)

def modulated_deform_conv2d(x, offset, mask, weight, bias, stride, padding,
                            dilation, groups, deform_groups):
    return _torchvision_deform_conv2d(
        x, offset, weight, bias, stride, padding, dilation, mask)

def constant_init(module, val, bias=0):
    if getattr(module, "weight", None) is not None:
        nn.init.constant_(module.weight, val)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)

class MMLogger:
    @staticmethod
    def get_current_instance():
        return None

def print_log(*args, **kwargs):
    pass

def load_checkpoint(module, checkpoint, strict=True, logger=None):
    if checkpoint.startswith(("http://", "https://")):
        value = torch.hub.load_state_dict_from_url(
            checkpoint, map_location="cpu", progress=False)
    else:
        value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = value.get("state_dict", value.get("params", value))
    return module.load_state_dict(state, strict=strict)
"""


def _writestr(bundle, name, data):
    info = zipfile.ZipInfo(name, (2025, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    bundle.writestr(info, data)


def build(source: Path, output: Path) -> None:
    required = {
        "src/DLoRAL_model.py": source / "src" / "DLoRAL_model.py",
        "src/my_utils/vaehook.py": source / "src" / "my_utils" / "vaehook.py",
        "src/my_utils/devices.py":
            source / "src" / "my_utils" / "devices.py",
        "src/my_utils/wavelet_color_fix.py":
            source / "src" / "my_utils" / "wavelet_color_fix.py",
        "src/cross_frame_retrieval/cfr_main.py":
            source / "src" / "cross_frame_retrieval" / "cfr_main.py",
        "src/cross_frame_retrieval/uncertainty_topk.py":
            source / "src" / "cross_frame_retrieval" / "uncertainty_topk.py",
        "DLoRAL_LICENSE.txt": source / "LICENSE",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing DLoRAL source assets: " + ", ".join(missing))

    cfr = required["src/cross_frame_retrieval/cfr_main.py"].read_text(
        encoding="utf-8")
    if _MMCV_IMPORTS not in cfr:
        raise RuntimeError("Pinned DLoRAL MMCV import block changed")
    cfr = cfr.replace(_MMCV_IMPORTS, _TORCH_COMPAT)

    model = required["src/DLoRAL_model.py"].read_text(encoding="utf-8")
    model = model.replace(
        "sd = torch.load(args.pretrained_path)",
        "sd = torch.load(args.pretrained_path, map_location='cpu', "
        "weights_only=False)")
    model = model.replace(
        "self.unet.enable_xformers_memory_efficient_attention()",
        "try:\n            self.unet.enable_xformers_memory_efficient_attention()"
        "\n        except (ImportError, ModuleNotFoundError):\n            pass")
    model = model.replace(
        ".from_pretrained(args.pretrained_model_path, subfolder=\"text_encoder\").cuda()",
        ".from_pretrained(args.pretrained_model_path, subfolder=\"text_encoder\")"
        ".to(\"cuda\", dtype=torch.float16)")
    for component in ("unet", "vae", "cfr_main_net"):
        model = model.replace(
            "self.%s.to(\"cuda\")" % component,
            "self.%s.to(\"cuda\", dtype=torch.float16)" % component)
    spynet_url = (
        "'https://download.openmmlab.com/mmediting/restorers/'"
        "'basicvsr/spynet_20210409-c6c1bd09.pth'")
    model = model.replace(
        spynet_url, "os.environ.get('LVE_DLORAL_SPYNET', " + spynet_url + ")")
    if "weights_only=False" not in model or "LVE_DLORAL_SPYNET" not in model:
        raise RuntimeError("Pinned DLoRAL model compatibility patch failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(str(temporary), "w") as bundle:
        _writestr(bundle, "src/__init__.py", b"")
        _writestr(bundle, "src/my_utils/__init__.py", b"")
        _writestr(bundle, "src/cross_frame_retrieval/__init__.py", b"")
        for name, path in required.items():
            if name == "src/cross_frame_retrieval/cfr_main.py":
                data = cfr.encode("utf-8")
            elif name == "src/DLoRAL_model.py":
                data = model.encode("utf-8")
            else:
                data = path.read_bytes()
            _writestr(bundle, name, data)
        _writestr(bundle, "UPSTREAM_COMMIT.txt", (PINNED_COMMIT + "\n").encode())
    os.replace(str(temporary), str(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="build/upstream/DLoRAL",
        help="Pinned DLoRAL checkout")
    parser.add_argument(
        "--output", default="light_video_enhancer/external/dloral_runtime.zip")
    args = parser.parse_args()
    build(Path(args.source), Path(args.output))


if __name__ == "__main__":
    main()

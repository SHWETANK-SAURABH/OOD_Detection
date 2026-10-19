"""Stage 1 check: CUDA torch build, GPU name, one-batch forward pass under autocast."""
import platform
import sys

import torch
import torchvision

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))
from src.utils import set_seed, get_device

set_seed(0)
print("OS:", platform.system(), platform.release())
print("Python:", sys.version.split()[0])
print("torch:", torch.__version__, "| torchvision:", torchvision.__version__, "| CUDA build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
dev = get_device()
print("GPU:", torch.cuda.get_device_name(0))

model = torchvision.models.resnet18(num_classes=10).to(dev)
x = torch.randn(128, 3, 32, 32, device=dev)
with torch.autocast("cuda", dtype=torch.float16):
    out = model(x)
print("forward OK, output shape:", tuple(out.shape), "dtype:", out.dtype)
assert "4060" in torch.cuda.get_device_name(0), "expected an RTX 4060"
print("STAGE 1 CHECK PASSED")

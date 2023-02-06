from __future__ import annotations
import torch
import matplotlib.pyplot as plt
from . import sequence

# NOTE: better approach would be that functions detect the device from their
# arguments or provide an use_gpu arg, so we dont need this global setting

# NOTE: maybe make util a submodule too to hide imports

use_gpu = False
gpu_dev = 0


def get_device() -> torch.device:
    """Return the device as given by ``util.use_gpu`` and ``util.gpu_dev``."""
    if use_gpu:
        return torch.device(f"cuda:{gpu_dev}")
    else:
        return torch.device("cpu")


def set_device(x: torch.Tensor) -> torch.Tensor:
    """Set the device of the passed tensor as given by :func:`get_deivce`."""
    if use_gpu:
        return x.cuda(gpu_dev)
    else:
        return x.cpu()


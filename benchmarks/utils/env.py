from __future__ import annotations

import platform
import sys

import numpy as np


def get_environment_info() -> dict:
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
    except Exception:
        info["torch_version"] = None

    try:
        import cupy

        info["cupy_version"] = cupy.__version__
    except Exception:
        info["cupy_version"] = None

    return info
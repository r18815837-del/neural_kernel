from __future__ import annotations

import os
import random
import numpy as np


def set_seed(seed: int) -> int:
    if not isinstance(seed, int):
        raise ValueError(f"seed must be int, got {type(seed).__name__}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
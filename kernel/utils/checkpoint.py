from __future__ import annotations

import pickle
from pathlib import Path

from kernel.nn.module import Module


def save_checkpoint(model: Module, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
    }

    with open(path_obj, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(model: Module, path: str) -> None:
    path_obj = Path(path)

    with open(path_obj, "rb") as f:
        payload = pickle.load(f)

    if "state_dict" not in payload:
        raise KeyError("Checkpoint does not contain 'state_dict'")

    model.load_state_dict(payload["state_dict"])
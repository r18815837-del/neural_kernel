from __future__ import annotations

import pickle
from pathlib import Path

from kernel.nn.module import Module


def save_checkpoint(
    model: Module,
    path: str | Path,
    optimizer=None,
    meta: dict | None = None,
) -> None:
    if not isinstance(model, Module):
        raise TypeError(
            f"save_checkpoint expects a Module, got {type(model).__name__}"
        )

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "meta": {} if meta is None else dict(meta),
    }

    if optimizer is not None:
        if not hasattr(optimizer, "state_dict"):
            raise TypeError("optimizer must provide state_dict()")
        payload["optimizer_state"] = optimizer.state_dict()

    with open(path_obj, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(
    model: Module,
    path: str | Path,
    optimizer=None,
):
    if not isinstance(model, Module):
        raise TypeError(
            f"load_checkpoint expects a Module, got {type(model).__name__}"
        )

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path_obj}")

    with open(path_obj, "rb") as f:
        payload = pickle.load(f)

    if "state_dict" not in payload:
        raise KeyError("Checkpoint does not contain 'state_dict'")

    model.load_state_dict(payload["state_dict"])

    if optimizer is not None:
        if "optimizer_state" not in payload:
            raise KeyError("Checkpoint does not contain 'optimizer_state'")
        if not hasattr(optimizer, "load_state_dict"):
            raise TypeError("optimizer must provide load_state_dict(...)")
        optimizer.load_state_dict(payload["optimizer_state"])

    return payload.get("meta", {})
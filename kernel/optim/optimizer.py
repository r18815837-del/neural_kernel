from __future__ import annotations

from typing import Any


class Optimizer:
    def __init__(self, params, lr: float = 1e-3):
        self.params = list(params)
        self.lr = lr

        self.defaults = {
            "lr": lr,
        }

        # Per-parameter optimizer state.
        # For plain SGD this stays empty, but it gives us a stable structure
        # for future optimizers such as momentum SGD or Adam.
        self.state = {param: {} for param in self.params}

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        serialized_state: dict[int, Any] = {}
        for idx, param in enumerate(self.params):
            serialized_state[idx] = self._serialize_value(self.state.get(param, {}))

        param_groups = [
            {
                "params": list(range(len(self.params))),
                "lr": self.lr,
            }
        ]

        return {
            "state": serialized_state,
            "param_groups": param_groups,
            "defaults": self._serialize_value(self.defaults),
            "meta": {
                "optimizer_class": self.__class__.__name__,
                "format_version": 1,
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._validate_state_dict(state_dict)

        meta = state_dict["meta"]
        expected_class = self.__class__.__name__
        received_class = meta["optimizer_class"]
        if received_class != expected_class:
            raise ValueError(
                f"Optimizer class mismatch: expected {expected_class}, got {received_class}"
            )

        expected_num_params = len(self.params)
        received_num_params = len(state_dict["state"])
        if received_num_params != expected_num_params:
            raise ValueError(
                f"Parameter count mismatch: expected {expected_num_params}, got {received_num_params}"
            )

        defaults = self._deserialize_value(state_dict["defaults"])
        if "lr" not in defaults:
            raise ValueError("Missing 'lr' in optimizer defaults")

        self.defaults = defaults
        self.lr = defaults["lr"]

        restored_state = {}
        for idx, param in enumerate(self.params):
            if idx not in state_dict["state"]:
                raise ValueError(f"Missing optimizer state for parameter index {idx}")
            restored_state[param] = self._deserialize_value(state_dict["state"][idx])

        self.state = restored_state

    def _validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError("Optimizer state_dict must be a dict")

        required_keys = {"state", "param_groups", "defaults", "meta"}
        missing = required_keys - set(state_dict.keys())
        if missing:
            raise ValueError(f"Missing keys in optimizer state_dict: {sorted(missing)}")

        if not isinstance(state_dict["state"], dict):
            raise TypeError("'state' must be a dict")

        if not isinstance(state_dict["param_groups"], list):
            raise TypeError("'param_groups' must be a list")

        if not isinstance(state_dict["defaults"], dict):
            raise TypeError("'defaults' must be a dict")

        if not isinstance(state_dict["meta"], dict):
            raise TypeError("'meta' must be a dict")

        meta = state_dict["meta"]
        if "optimizer_class" not in meta:
            raise ValueError("Missing 'optimizer_class' in optimizer state_dict['meta']")
        if "format_version" not in meta:
            raise ValueError("Missing 'format_version' in optimizer state_dict['meta']")

        if meta["format_version"] != 1:
            raise ValueError(
                f"Unsupported optimizer state format version: {meta['format_version']}"
            )

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        return value

    def _deserialize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._deserialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._deserialize_value(v) for v in value]
        return value
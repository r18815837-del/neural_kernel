from __future__ import annotations

from typing import Iterator

import numpy as np

from kernel.core.tensor import Tensor


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
            if name in self._buffers:
                del self._buffers[name]
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            if hasattr(self, "_parameters") and name in self._parameters:
                del self._parameters[name]
            if hasattr(self, "_modules") and name in self._modules and not isinstance(value, Module):
                del self._modules[name]
        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, value) -> None:
        array = np.array(value, copy=True)
        self._buffers[name] = array
        object.__setattr__(self, name, array)

    def parameters(self) -> Iterator[Tensor]:
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        for name, param in self._parameters.items():
            yield prefix + name, param
        for name, module in self._modules.items():
            yield from module.named_parameters(prefix=f"{prefix}{name}.")

    def buffers(self) -> Iterator[np.ndarray]:
        for buf in self._buffers.values():
            yield buf
        for module in self._modules.values():
            yield from module.buffers()

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, np.ndarray]]:
        for name, buf in self._buffers.items():
            yield prefix + name, buf
        for name, module in self._modules.items():
            yield from module.named_buffers(prefix=f"{prefix}{name}.")

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = None

    def train(self) -> None:
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self) -> None:
        self.training = False
        for module in self._modules.values():
            module.eval()

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {}

        for name, param in self.named_parameters():
            state[name] = param.data.copy()

        for name, buf in self.named_buffers():
            state[name] = np.array(buf, copy=True)

        return state

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        current_params = dict(self.named_parameters())
        current_buffers = dict(self.named_buffers())

        current_state = {}
        current_state.update(current_params)
        current_state.update(current_buffers)

        missing_keys = [name for name in current_state if name not in state_dict]
        unexpected_keys = [name for name in state_dict if name not in current_state]

        if missing_keys:
            raise KeyError(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")

        for name, param in current_params.items():
            value = np.array(state_dict[name], copy=True)

            if param.data.shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"expected {param.data.shape}, got {value.shape}"
                )

            param.data = value

        for name in current_buffers:
            value = np.array(state_dict[name], copy=True)

            if current_buffers[name].shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"expected {current_buffers[name].shape}, got {value.shape}"
                )

            self._set_buffer_by_name(name, value)

    def _set_buffer_by_name(self, name: str, value: np.ndarray) -> None:
        parts = name.split(".")
        module = self

        for part in parts[:-1]:
            module = module._modules[part]

        buf_name = parts[-1]
        module._buffers[buf_name] = value
        object.__setattr__(module, buf_name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
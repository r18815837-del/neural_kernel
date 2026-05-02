from __future__ import annotations

from typing import Callable, Dict


class ModelRegistry:
    def __init__(self):
        self._builders: Dict[str, Callable] = {}

    def register(self, name: str, builder: Callable) -> None:
        if not name:
            raise ValueError("Model name cannot be empty")
        if name in self._builders:
            raise ValueError(f"Model '{name}' is already registered")
        self._builders[name] = builder

    def get(self, name: str) -> Callable:
        if name not in self._builders:
            raise KeyError(f"Model '{name}' is not registered")
        return self._builders[name]

    def create(self, name: str, *args, **kwargs):
        builder = self.get(name)
        return builder(*args, **kwargs)

    def available(self):
        return sorted(self._builders.keys())


model_registry = ModelRegistry()
from __future__ import annotations

from kernel.nn.module import Module


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._list = []

        if modules is not None:
            for module in modules:
                self.append(module)

    def append(self, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(
                f"ModuleList can only store Module instances, got {type(module).__name__}"
            )

        index = len(self._list)
        self._list.append(module)
        setattr(self, str(index), module)

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, index):
        return self._list[index]


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._list = []

        for module in modules:
            self.append(module)

    def append(self, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(
                f"Sequential can only store Module instances, got {type(module).__name__}"
            )

        index = len(self._list)
        self._list.append(module)
        setattr(self, str(index), module)

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, index):
        return self._list[index]

    def forward(self, x):
        out = x
        for module in self._list:
            out = module(out)
        return out


class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._dict = {}

        if modules is not None:
            for name, module in modules.items():
                self[name] = module

    def __setitem__(self, key: str, module: Module) -> None:
        if not isinstance(key, str):
            raise TypeError(
                f"ModuleDict keys must be str, got {type(key).__name__}"
            )
        if not isinstance(module, Module):
            raise TypeError(
                f"ModuleDict values must be Module instances, got {type(module).__name__}"
            )

        self._dict[key] = module
        setattr(self, key, module)

    def __getitem__(self, key: str) -> Module:
        return self._dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()
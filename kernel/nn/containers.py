from __future__ import annotations

from kernel.nn.module import Module


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = list(modules)

        for idx, module in enumerate(self.modules_list):
            setattr(self, str(idx), module)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x

    def __len__(self):
        return len(self.modules_list)

    def __getitem__(self, index):
        return self.modules_list[index]
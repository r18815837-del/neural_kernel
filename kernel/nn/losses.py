from __future__ import annotations

import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.module import Module
from kernel.autograd.ops.loss_ops import cross_entropy


class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, targets) -> Tensor:
        if isinstance(targets, Tensor):
            targets_data = targets.data
        else:
            xp = logits.xp
            targets_data = xp.asarray(targets)

        return cross_entropy(logits, targets_data)
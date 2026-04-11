from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kernel.core.tensor import Tensor
from kernel.loss.regression import MSELoss
from kernel.nn.layers.linear import Linear
from kernel.optim.sgd import SGD


def main() -> None:
    np.random.seed(42)

    x_np = np.linspace(-2, 2, 100).reshape(-1, 1)
    noise = np.random.normal(0, 0.2, size=(100, 1))
    y_np = 3.0 * x_np + 2.0 + noise

    x = Tensor(x_np)
    y = Tensor(y_np)

    model = Linear(1, 1)
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    for epoch in range(1, 201):
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.data.item():.6f}")

    print("\nLearned parameters:")
    print("weight:", model.weight.data.ravel()[0])
    print("bias:", model.bias.data.ravel()[0])


if __name__ == "__main__":
    main()

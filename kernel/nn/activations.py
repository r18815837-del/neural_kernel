from __future__ import annotations

from kernel.nn.module import Module
from kernel.autograd.ops import relu, sigmoid, leaky_relu, tanh, softmax
class ReLU(Module):
    def forward(self, x):
        return relu(x)


class Sigmoid(Module):
    def forward(self, x):
        return sigmoid(x)

class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return leaky_relu(x, negative_slope=self.negative_slope)

class Tanh(Module):
    def forward(self, x):
        return tanh(x)

class Identity(Module):
    def forward(self, x):
        return x

class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return softmax(x, axis=self.axis)
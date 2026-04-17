"""Neural Kernel public API."""

from .core.tensor import Tensor
from .tokenization import BaseTokenizer, TokenizerInfo, ChatMessage, MockTokenizer
from .nn.module import Module
from .nn.activations import ReLU, Sigmoid, Tanh, LeakyReLU, Identity, Softmax
from .nn.containers import Sequential
from .nn.dropout import Dropout
from .nn.normalization import BatchNorm1d, BatchNorm2d, LayerNorm
from .nn.losses import CrossEntropyLoss
from .nn.init import (
    xavier_uniform_,
    xavier_normal_,
    kaiming_uniform_,
    kaiming_normal_,
)
from .nn.layers import (
    Linear,
    Conv2d,
    Flatten,
    ResidualBlock,
    MaxPool2d,
    AvgPool2d,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool2d,
    Embedding,
)
from .nn.modules import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
    FeedForward,
    TransformerBlock,
    TransformerEncoder,
    TransformerEncoderClassifier,
    TokenTransformerClassifier,
    TokenTransformerLM,
    ModuleList,
    ModuleDict,
)

from .optim import Optimizer, SGD, Adam, StepLR

from .utils import (
    save_checkpoint,
    load_checkpoint,
    History,
    set_seed,
    accuracy,
    mse,
    plot_history,
)

__all__ = [
    "Tensor",
    "Module",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "LeakyReLU",
    "Identity",
    "Softmax",
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "Dropout",
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    "CrossEntropyLoss",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "Linear",
    "Conv2d",
    "Flatten",
    "ResidualBlock",
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d",
    "Embedding",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "PositionalEncoding",
    "FeedForward",
    "TransformerBlock",
    "TransformerEncoder",
    "TransformerEncoderClassifier",
    "TokenTransformerClassifier",
    "TokenTransformerLM",
    "Optimizer",
    "SGD",
    "Adam",
    "StepLR",
    "save_checkpoint",
    "load_checkpoint",
    "History",
    "set_seed",
    "accuracy",
    "mse",
    "plot_history",
    "BaseTokenizer",
    "TokenizerInfo",
    "ChatMessage",
    "MockTokenizer",
]

__version__ = "0.1.0"

def test_root_public_imports():
    import kernel

    assert kernel.Tensor is not None
    assert kernel.Module is not None

    assert kernel.Linear is not None
    assert kernel.Conv2d is not None
    assert kernel.Flatten is not None
    assert kernel.ResidualBlock is not None
    assert kernel.Embedding is not None

    assert kernel.ReLU is not None
    assert kernel.Sigmoid is not None
    assert kernel.LeakyReLU is not None
    assert kernel.Tanh is not None
    assert kernel.Identity is not None
    assert kernel.Softmax is not None

    assert kernel.Sequential is not None
    assert kernel.Dropout is not None
    assert kernel.BatchNorm1d is not None
    assert kernel.BatchNorm2d is not None
    assert kernel.LayerNorm is not None

    assert kernel.CrossEntropyLoss is not None

    assert kernel.xavier_uniform_ is not None
    assert kernel.xavier_normal_ is not None
    assert kernel.kaiming_uniform_ is not None
    assert kernel.kaiming_normal_ is not None

    assert kernel.Optimizer is not None
    assert kernel.SGD is not None
    assert kernel.Adam is not None
    assert kernel.StepLR is not None

    assert kernel.save_checkpoint is not None
    assert kernel.load_checkpoint is not None
    assert kernel.History is not None
    assert kernel.set_seed is not None
    assert kernel.accuracy is not None
    assert kernel.mse is not None
    assert kernel.plot_history is not None


def test_root_all_is_defined():
    import kernel

    assert hasattr(kernel, "__all__")
    assert isinstance(kernel.__all__, list)

    expected = [
        "Tensor",
        "Module",
        "Linear",
        "Conv2d",
        "Flatten",
        "ResidualBlock",
        "Embedding",
        "CrossEntropyLoss",
        "SGD",
        "Adam",
        "StepLR",
        "save_checkpoint",
        "set_seed",
    ]

    for name in expected:
        assert name in kernel.__all__


def test_nn_public_imports():
    from kernel.nn import (
        Module,
        ReLU,
        Sigmoid,
        Sequential,
        Dropout,
        BatchNorm1d,
        BatchNorm2d,
        Linear,
        Conv2d,
        Flatten,
        MaxPool2d,
        AvgPool2d,
        AdaptiveAvgPool2d,
        LeakyReLU,
        Tanh,
        LayerNorm,
        Identity,
        Softmax,
        ResidualBlock,
        CrossEntropyLoss,
        xavier_uniform_,
        xavier_normal_,
        kaiming_uniform_,
        kaiming_normal_,
    )

    assert Module is not None
    assert ReLU is not None
    assert Sigmoid is not None
    assert Sequential is not None
    assert Dropout is not None
    assert BatchNorm1d is not None
    assert BatchNorm2d is not None
    assert Linear is not None
    assert Conv2d is not None
    assert Flatten is not None
    assert MaxPool2d is not None
    assert AvgPool2d is not None
    assert AdaptiveAvgPool2d is not None
    assert LeakyReLU is not None
    assert Tanh is not None
    assert LayerNorm is not None
    assert Identity is not None
    assert Softmax is not None
    assert ResidualBlock is not None
    assert CrossEntropyLoss is not None
    assert xavier_uniform_ is not None
    assert xavier_normal_ is not None
    assert kaiming_uniform_ is not None
    assert kaiming_normal_ is not None


def test_nn_all_is_defined():
    import kernel.nn as nn

    assert hasattr(nn, "__all__")
    assert isinstance(nn.__all__, list)
    assert "Module" in nn.__all__
    assert "Linear" in nn.__all__
    assert "CrossEntropyLoss" in nn.__all__


def test_layers_public_imports():
    from kernel.nn.layers import (
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

    assert Linear is not None
    assert Conv2d is not None
    assert Flatten is not None
    assert ResidualBlock is not None
    assert MaxPool2d is not None
    assert AvgPool2d is not None
    assert AdaptiveAvgPool2d is not None
    assert AdaptiveMaxPool2d is not None
    assert Embedding is not None


def test_layers_all_is_defined():
    import kernel.nn.layers as layers

    assert hasattr(layers, "__all__")
    assert isinstance(layers.__all__, list)
    assert "Linear" in layers.__all__
    assert "Embedding" in layers.__all__


def test_functional_public_imports():
    from kernel.nn.functional import (
        scaled_dot_product_attention,
        make_causal_mask,
        make_padding_mask,
    )

    assert scaled_dot_product_attention is not None
    assert make_causal_mask is not None
    assert make_padding_mask is not None


def test_functional_all_is_defined():
    import kernel.nn.functional as F

    assert hasattr(F, "__all__")
    assert isinstance(F.__all__, list)
    assert "scaled_dot_product_attention" in F.__all__
    assert "make_causal_mask" in F.__all__
    assert "make_padding_mask" in F.__all__


def test_optim_public_imports():
    from kernel.optim import Optimizer, SGD, Adam, StepLR

    assert Optimizer is not None
    assert SGD is not None
    assert Adam is not None
    assert StepLR is not None


def test_utils_public_imports():
    from kernel.utils import (
        save_checkpoint,
        load_checkpoint,
        History,
        set_seed,
        accuracy,
        mse,
        plot_history,
    )

    assert save_checkpoint is not None
    assert load_checkpoint is not None
    assert History is not None
    assert set_seed is not None
    assert accuracy is not None
    assert mse is not None
    assert plot_history is not None
def test_canonical_import_style():
    import kernel as K

    x = K.Tensor([1.0, 2.0, 3.0])

    model = K.Sequential(
        K.Linear(3, 4),
        K.ReLU(),
        K.Linear(4, 2),
    )

    criterion = K.CrossEntropyLoss()
    optimizer = K.SGD(model.parameters(), lr=0.01)

    assert x is not None
    assert model is not None
    assert criterion is not None
    assert optimizer is not None


def test_utils_from_root():
    import kernel as K

    assert K.set_seed is not None
    assert K.save_checkpoint is not None
    assert K.load_checkpoint is not None
    assert K.accuracy is not None
    assert K.mse is not None


def test_optimizers_from_root():
    import kernel as K

    assert K.SGD is not None
    assert K.Adam is not None
    assert K.StepLR is not None


def test_common_layers_from_root():
    import kernel as K

    assert K.Linear is not None
    assert K.Conv2d is not None
    assert K.Flatten is not None
    assert K.Dropout is not None
    assert K.BatchNorm1d is not None
    assert K.BatchNorm2d is not None
    assert K.LayerNorm is not None


def test_common_activations_from_root():
    import kernel as K

    assert K.ReLU is not None
    assert K.Sigmoid is not None
    assert K.LeakyReLU is not None
    assert K.Tanh is not None
    assert K.Identity is not None
    assert K.Softmax is not None
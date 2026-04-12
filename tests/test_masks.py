import numpy as np

from kernel.nn.functional import make_causal_mask, make_padding_mask


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "data"):
        data = x.data
        if isinstance(data, np.ndarray):
            return data
        try:
            return np.array(data)
        except Exception:
            pass
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


def test_make_causal_mask_shape():
    mask = make_causal_mask(4)
    arr = to_numpy(mask)

    assert arr.shape == (1, 1, 4, 4)


def test_make_causal_mask_blocks_future_positions():
    mask = make_causal_mask(4)
    arr = to_numpy(mask)

    m = arr[0, 0]

    assert m.shape == (4, 4)

    # allowed lower-triangular positions
    assert m[0, 0] == m[1, 0]
    assert m[1, 0] == m[2, 1]
    assert m[2, 1] == m[3, 3]

    # blocked future positions
    assert m[0, 1] == m[0, 2]
    assert m[0, 2] == m[0, 3]
    assert m[1, 2] == m[0, 1]

    # allowed and blocked values must differ
    assert m[0, 0] != m[0, 1]


def test_make_padding_mask_shape():
    lengths = np.array([2, 3], dtype=np.int64)
    mask = make_padding_mask(lengths, max_len=4)
    arr = to_numpy(mask)

    assert arr.shape == (2, 1, 1, 4)


def test_make_padding_mask_marks_padding_positions():
    lengths = np.array([2, 3], dtype=np.int64)
    mask = make_padding_mask(lengths, max_len=4)
    arr = to_numpy(mask)

    m0 = arr[0, 0, 0]
    m1 = arr[1, 0, 0]

    # sample 0: valid valid pad pad
    assert m0[0] == m0[1]
    assert m0[2] == m0[3]
    assert m0[0] != m0[2]

    # sample 1: valid valid valid pad
    assert m1[0] == m1[1]
    assert m1[1] == m1[2]
    assert m1[2] != m1[3]


def test_make_padding_mask_no_padding_case():
    lengths = np.array([3, 3], dtype=np.int64)
    mask = make_padding_mask(lengths, max_len=3)
    arr = to_numpy(mask)

    assert arr.shape == (2, 1, 1, 3)

    m0 = arr[0, 0, 0]
    m1 = arr[1, 0, 0]

    assert np.all(m0 == m0[0])
    assert np.all(m1 == m1[0])
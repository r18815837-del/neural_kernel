from kernel.utils.history import History


def test_history_log_single_metric():
    h = History()
    h.log(loss=1.23)

    assert h.get("loss") == [1.23]


def test_history_log_multiple_metrics():
    h = History()
    h.log(loss=1.0, acc=0.9)
    h.log(loss=0.8, acc=0.95)

    assert h.get("loss") == [1.0, 0.8]
    assert h.get("acc") == [0.9, 0.95]


def test_history_keys_and_as_dict():
    h = History()
    h.log(loss=1.0, acc=0.5)

    d = h.as_dict()

    assert "loss" in h.keys()
    assert "acc" in h.keys()
    assert d["loss"] == [1.0]
    assert d["acc"] == [0.5]
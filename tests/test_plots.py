from kernel.utils.history import History
from kernel.utils.plots import plot_history


def test_plot_history_returns_figure():
    h = History()
    h.log(epoch=1, train_loss=1.0, train_acc=0.8, test_acc=0.75, lr=0.001)
    h.log(epoch=2, train_loss=0.8, train_acc=0.85, test_acc=0.80, lr=0.001)

    fig = plot_history(h, show=False)

    assert fig is not None
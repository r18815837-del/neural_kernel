import numpy as np

from kernel.core.tensor import Tensor
from kernel.optim.adam import Adam
from kernel.nn.modules.classifier import TransformerEncoderClassifier


def make_batch(batch_size=16, seq_len=6, d_model=8):
    x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float64)

    # простая метка: зависит от среднего по входу
    score = x_np.mean(axis=(1, 2), keepdims=False)
    y_np = (score > 0.0).astype(np.float64).reshape(batch_size, 1)

    x = Tensor(x_np, requires_grad=True)
    y = Tensor(y_np, requires_grad=False)

    return x, y


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred - target
    return (diff * diff).mean()


def compute_accuracy(pred: Tensor, target: Tensor) -> float:
    pred_np = pred.detach().numpy()
    target_np = target.detach().numpy()

    pred_label = (pred_np > 0.5).astype(np.float64)
    acc = (pred_label == target_np).mean()
    return float(acc)


def main():
    np.random.seed(42)

    model = TransformerEncoderClassifier(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=1,
        dropout_p=0.0,
        max_len=32,
        use_positional_encoding=True,
        activation="gelu",
        pooling="mean",
    )

    optimizer = Adam(model.parameters(), lr=1e-2)

    print("start training demo")
    first_loss = None
    last_loss = None

    for step in range(1, 61):
        x, y = make_batch(batch_size=16, seq_len=6, d_model=8)

        model.train()
        model.zero_grad()

        pred, _ = model(x)
        loss = mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().numpy())
        acc_value = compute_accuracy(pred, y)

        if first_loss is None:
            first_loss = loss_value
        last_loss = loss_value

        if step % 10 == 0 or step == 1:
            print(f"step={step:03d}  loss={loss_value:.6f}  acc={acc_value:.3f}")

    print()
    print(f"first_loss={first_loss:.6f}")
    print(f"last_loss={last_loss:.6f}")

    if last_loss > first_loss:
        raise AssertionError(
            f"Training did not improve: first_loss={first_loss:.6f}, last_loss={last_loss:.6f}"
        )

    print("[OK] training demo")


if __name__ == "__main__":
    main()
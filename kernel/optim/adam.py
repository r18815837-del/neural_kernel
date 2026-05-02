from __future__ import annotations

from kernel.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, lr=lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        self.m = [p.xp.zeros_like(p.data) for p in self.params]
        self.v = [p.xp.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            xp = param.xp
            grad = param.grad

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            param.data = param.data - self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict:
        return {
            "type": "Adam",
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "t": self.t,
            "m": [p._backend.to_cpu(buf).copy() for p, buf in zip(self.params, self.m)],
            "v": [p._backend.to_cpu(buf).copy() for p, buf in zip(self.params, self.v)],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict.get("type") != "Adam":
            raise ValueError(f"Invalid optimizer type: {state_dict.get('type')}")

        self.lr = state_dict["lr"]
        self.beta1 = state_dict["beta1"]
        self.beta2 = state_dict["beta2"]
        self.eps = state_dict["eps"]
        self.t = state_dict["t"]

        m_list = state_dict["m"]
        v_list = state_dict["v"]

        if len(m_list) != len(self.params) or len(v_list) != len(self.params):
            raise ValueError(
                f"Optimizer state size mismatch: "
                f"expected {len(self.params)} params, got m={len(m_list)}, v={len(v_list)}"
            )

        self.m = []
        self.v = []

        for param, m_buf, v_buf in zip(self.params, m_list, v_list):
            backend = param._backend
            self.m.append(backend.from_cpu(m_buf))
            self.v.append(backend.from_cpu(v_buf))
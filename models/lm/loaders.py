from __future__ import annotations

from pathlib import Path

from models.lm.modeling_token_lm import TokenLanguageModel


class ModelLoader:
    @staticmethod
    def load_token_lm(config, checkpoint_path: str):
        model = TokenLanguageModel(config)

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # TODO:
        # подключить твой existing checkpoint loader
        # например:
        # from kernel.utils.checkpoint import load_checkpoint
        # state = load_checkpoint(...)
        # model.load_state_dict(state["model_state"])

        return model
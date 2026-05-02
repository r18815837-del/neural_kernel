from __future__ import annotations

from models.base.registry import model_registry
from models.lm.modeling_token_lm import TokenLanguageModel


def create_token_lm(config):
    return TokenLanguageModel(config)


def register_models() -> None:
    try:
        model_registry.register("token_lm", create_token_lm)
    except ValueError:
        # уже зарегистрировано
        pass
from __future__ import annotations

import subprocess
import sys

import torch
from transformers import Qwen2ForCausalLM

from dispider.model import (
    Decision,
    DecisionConfig,
    Perception,
    PerceptionDecision,
    Reaction,
    ReactionConfig,
    load_pretrained_model,
)


def _tiny_config(config_type):
    return config_type(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        torch_dtype=torch.float16,
    )


def test_public_api_contains_only_canonical_components():
    import dispider.model as model_api

    assert set(model_api.__all__) == {
        "Decision",
        "DecisionConfig",
        "Perception",
        "PerceptionDecision",
        "Reaction",
        "ReactionConfig",
        "load_pretrained_model",
    }
    assert callable(load_pretrained_model)
    assert Perception.__name__ == "Perception"
    assert PerceptionDecision.__name__ == "PerceptionDecision"
    assert Decision.__name__ == "Decision"
    assert Reaction.__name__ == "Reaction"


def test_public_api_import_is_lazy():
    code = """
import sys
import dispider.model
assert 'dispider.model.decision' not in sys.modules
assert 'dispider.model.reaction' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_canonical_config_types():
    assert DecisionConfig.model_type == "dispider_decision"
    assert ReactionConfig.model_type == "dispider_reaction"
    assert ReactionConfig.mm_use_im_start_end is False


def test_tiny_decision_keeps_released_weight_schema():
    model = Decision(_tiny_config(DecisionConfig))
    keys = set(model.state_dict())

    assert "silent_head.weight" in keys
    assert "model.embed_tokens.weight" in keys
    assert "lm_head.weight" in keys
    assert model.decision_head is model.silent_head


def test_tiny_reaction_uses_canonical_model():
    model = Reaction(_tiny_config(ReactionConfig))

    assert model.get_perception_decision() is None
    assert "model.embed_tokens.weight" in model.state_dict()


def test_reaction_generate_preserves_explicit_masks(monkeypatch):
    model = Reaction(_tiny_config(ReactionConfig))
    attention_mask = torch.tensor([[True, True]])
    position_ids = torch.tensor([[4, 5]])
    captured = {}

    def record_generate(_self, **kwargs):
        captured.update(kwargs)
        return torch.zeros((1, 1), dtype=torch.long)

    monkeypatch.setattr(Qwen2ForCausalLM, "generate", record_generate)
    model.generate(
        torch.tensor([[1, 2]]),
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    assert captured["attention_mask"] is attention_mask
    assert captured["position_ids"] is position_ids

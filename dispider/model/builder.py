"""Load the released portable Dispider checkpoint."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from .checkpoint import (
    load_checkpoint_layout,
    prepare_reaction_config,
    resolve_checkpoint_root,
)
from .reaction import Reaction, ReactionConfig


def load_pretrained_model(
    model_path,
    _model_base=None,
    _model_name=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    """Return tokenizer, Reaction model, processors, and context length."""

    hub_kwargs = {
        key: kwargs.pop(key)
        for key in ("cache_dir", "revision", "token", "local_files_only")
        if key in kwargs
    }
    checkpoint_root = resolve_checkpoint_root(model_path, **hub_kwargs)
    layout = load_checkpoint_layout(checkpoint_root)
    config = ReactionConfig.from_pretrained(
        checkpoint_root,
        local_files_only=True,
    )
    prepare_reaction_config(config, layout)

    model_kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        model_kwargs["device_map"] = {"": device}
    if load_8bit:
        model_kwargs["load_in_8bit"] = True
    elif load_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_root,
        use_fast=False,
        local_files_only=True,
    )
    model = Reaction.from_pretrained(
        checkpoint_root,
        config=config,
        low_cpu_mem_usage=True,
        **model_kwargs,
    )

    model.resize_token_embeddings(len(tokenizer))

    perception_decision = model.get_perception_decision()
    processors = (
        perception_decision.perception.image_processor,
        perception_decision.tokenizer,
    )
    context_length = getattr(
        config,
        "max_sequence_length",
        getattr(config, "tokenizer_model_max_length", 2048),
    )
    return tokenizer, model, processors, context_length


__all__ = ["load_pretrained_model"]

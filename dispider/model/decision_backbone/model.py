"""Public Decision Qwen2 classes with no vendored Transformer implementation."""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import Tensor, nn
from transformers import Qwen2Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM as TransformersQwen2ForCausalLM,
    Qwen2PreTrainedModel,
)

from .adapter import Qwen2Model, _validate_cache_position
from .streaming import DecisionStreamingMixin


class Qwen2ForCausalLM(
    DecisionStreamingMixin,
    Qwen2PreTrainedModel,
):
    """Qwen2 language-model shell plus Dispider's Decision task methods."""

    config_class = Qwen2Config
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    get_input_embeddings = TransformersQwen2ForCausalLM.get_input_embeddings
    set_input_embeddings = TransformersQwen2ForCausalLM.set_input_embeddings
    get_output_embeddings = TransformersQwen2ForCausalLM.get_output_embeddings
    set_output_embeddings = TransformersQwen2ForCausalLM.set_output_embeddings
    set_decoder = TransformersQwen2ForCausalLM.set_decoder
    get_decoder = TransformersQwen2ForCausalLM.get_decoder
    prepare_inputs_for_generation = (
        TransformersQwen2ForCausalLM.prepare_inputs_for_generation
    )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        indicators: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        current = inputs_embeds if inputs_embeds is not None else input_ids
        if current is not None:
            batch_size, sequence_length = current.shape[:2]
            if indicators is not None and indicators.shape != (
                batch_size,
                sequence_length,
            ):
                raise ValueError("indicators must align with the current tokens")
            _validate_cache_position(
                cache_position,
                past_key_values,
                sequence_length,
            )
        return TransformersQwen2ForCausalLM.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


__all__ = ["Qwen2Config", "Qwen2ForCausalLM", "Qwen2Model"]

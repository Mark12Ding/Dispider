"""Thin Dispider adapter over the pinned Transformers Qwen2 decoder."""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import Tensor
from transformers import Qwen2Config
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model as TransformersQwen2Model,
)


def _cache_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, Cache):
        return int(past_key_values.get_seq_length())
    if not past_key_values:
        return 0
    return int(past_key_values[0][0].shape[-2])


def _validate_cache_position(
    cache_position: Optional[Tensor],
    past_key_values,
    sequence_length: int,
) -> None:
    """Require Decision cache positions to be contiguous."""

    if cache_position is None:
        return
    position = cache_position.reshape(-1)
    start = _cache_length(past_key_values)
    expected = torch.arange(
        start,
        start + sequence_length,
        dtype=position.dtype,
        device=position.device,
    )
    if position.shape != expected.shape or not torch.equal(position, expected):
        raise ValueError("Decision KV cache requires contiguous cache_position values")


class Qwen2Model(TransformersQwen2Model):
    """Released Decision decoder backed by Transformers 4.41 Qwen2.

    Dispider uses FlashAttention2 and reads the complete 28-layer model. Token
    role indicators are consumed by the Decision logic around this decoder.
    """

    config_class = Qwen2Config

    def __init__(self, config: Qwen2Config):
        config._attn_implementation = "flash_attention_2"
        super().__init__(config)

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
        select_layer: Optional[int] = None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if inputs_embeds is not None:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        elif input_ids is not None:
            batch_size, sequence_length = input_ids.shape
        else:
            raise ValueError("input_ids or inputs_embeds must be provided")

        if indicators is not None and indicators.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError("indicators must align with the current tokens")

        if select_layer is not None and 0 <= select_layer < len(self.layers) - 1:
            raise ValueError(
                "partial Decision readout layers are not part of the released model"
            )

        _validate_cache_position(
            cache_position,
            past_key_values,
            sequence_length,
        )
        return super().forward(
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


__all__ = ["Qwen2Config", "Qwen2Model"]

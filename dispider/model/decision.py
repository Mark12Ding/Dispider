"""Inference-only Dispider Decision model."""

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn

from .decision_backbone import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from .decision_inputs import DecisionInputsMixin, DecisionModelMixin


DECISION_READOUT_LAYER = 100


@dataclass
class _PreparedDecisionInputs:
    input_ids: Any
    position_ids: Any
    attention_mask: Any
    past_key_values: Any
    inputs_embeds: Any
    clip_embeds: Any
    question_embeds: Any
    question_mask: Any
    indicators: Any
    answer_embed: Any
    todo_embed: Any

    def decoder_arguments(self):
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "past_key_values": self.past_key_values,
            "inputs_embeds": self.inputs_embeds,
            "qs_embeds": self.question_embeds,
            "qs_mask": self.question_mask,
            "indicators": self.indicators,
            "select_layer": DECISION_READOUT_LAYER,
            "ans_token": self.answer_embed,
            "todo_token": self.todo_embed,
        }


class DecisionConfig(Qwen2Config):
    model_type = "dispider_decision"


class _DecisionModel(DecisionModelMixin, Qwen2Model):
    config_class = DecisionConfig


class Decision(Qwen2ForCausalLM, DecisionInputsMixin):
    config_class = DecisionConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = _DecisionModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.silent_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.vocab_size = config.vocab_size
        self.post_init()

    def get_model(self):
        return self.model

    @property
    def decision_head(self):
        return self.silent_head

    def _prepare_decision_inputs(
        self,
        input_ids,
        position_ids,
        attention_mask,
        question_ids,
        question_mask,
        past_key_values,
        inputs_embeds,
        images,
        answer_token,
        todo_token,
    ):
        if inputs_embeds is not None:
            raise ValueError(
                "Decision entry points do not accept precomputed inputs_embeds"
            )
        values = self.prepare_inference_inputs(
            input_ids,
            position_ids,
            attention_mask,
            question_ids,
            question_mask,
            past_key_values,
            images,
            answer_token,
            todo_token,
        )
        return _PreparedDecisionInputs(*values)

    def forward_token(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        qs_ids: torch.LongTensor,
        qs_mask: torch.Tensor,
        images: torch.FloatTensor,
        insert_position: int,
        ans_position: Optional[list],
        ans_token: torch.LongTensor,
        todo_token: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        prepared = self._prepare_decision_inputs(
            input_ids,
            position_ids,
            attention_mask,
            qs_ids,
            qs_mask,
            past_key_values,
            inputs_embeds,
            images,
            ans_token,
            todo_token,
        )
        arguments = prepared.decoder_arguments()
        arguments.update(
            clip_embeds=torch.stack(prepared.clip_embeds, dim=0),
            insert_position=insert_position,
            ans_position=ans_position,
        )
        return super().forward_token_stream(**arguments)

    def forward_inference(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        qs_ids: torch.LongTensor,
        qs_mask: torch.Tensor,
        images: torch.FloatTensor,
        ans_token: torch.LongTensor,
        todo_token: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        prepared = self._prepare_decision_inputs(
            input_ids,
            position_ids,
            attention_mask,
            qs_ids,
            qs_mask,
            past_key_values,
            inputs_embeds,
            images,
            ans_token,
            todo_token,
        )
        return super().infer_trigger_sequence(**prepared.decoder_arguments())


__all__ = ["Decision", "DecisionConfig"]

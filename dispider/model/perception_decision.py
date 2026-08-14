"""The compact Perception/Decision component embedded in Reaction."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .checkpoint import prepare_perception_decision_config
from .decision import Decision, DecisionConfig


class PerceptionDecision(nn.Module):
    def __init__(self, metadata_dir):
        super().__init__()
        self.metadata_dir = os.fspath(metadata_dir)
        config = DecisionConfig.from_pretrained(
            self.metadata_dir,
            local_files_only=True,
        )
        prepare_perception_decision_config(config, self.metadata_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.metadata_dir,
            local_files_only=True,
        )

        # The attribute name is part of the released composite state dict.
        self.compressor = Decision(config)

    @property
    def perception(self):
        return self.compressor.get_vision_tower()

    @property
    def decision(self):
        return self.compressor

    def forward_decision(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        question_ids: torch.LongTensor,
        question_mask: torch.Tensor,
        images: torch.FloatTensor,
        insert_position: int,
        answer_position: list,
        answer_token: torch.LongTensor,
        todo_token: torch.LongTensor,
    ):
        return self.compressor.forward_token(
            input_ids=input_ids,
            attention_mask=attention_mask,
            qs_ids=question_ids,
            qs_mask=question_mask,
            images=images,
            insert_position=insert_position,
            ans_position=answer_position,
            ans_token=answer_token,
            todo_token=todo_token,
        )

    def forward(
        self,
        clips,
        _clips_large,
        sequences,
        sequence_mask,
        question_ids,
        question_mask,
        answer_token,
        todo_token,
        insert_position,
        answer_position,
    ):
        return self.forward_decision(
            input_ids=sequences,
            attention_mask=sequence_mask,
            question_ids=question_ids,
            question_mask=question_mask,
            images=clips,
            answer_token=answer_token,
            todo_token=todo_token,
            insert_position=insert_position,
            answer_position=answer_position,
        )

    @property
    def dtype(self):
        return self.compressor.dtype

    @property
    def device(self):
        return self.compressor.device

    @property
    def config(self):
        return self.compressor.config

    @property
    def hidden_size(self):
        return self.config.hidden_size


def build_perception_decision(config):
    metadata_dir = getattr(config, "perception_decision", None)
    if not metadata_dir or not os.path.isdir(metadata_dir):
        raise ValueError(f"Invalid Perception/Decision metadata: {metadata_dir}")
    return PerceptionDecision(metadata_dir)


__all__ = ["PerceptionDecision", "build_perception_decision"]

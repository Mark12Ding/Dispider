"""Inference-only Dispider Reaction model and video-token assembly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from dispider.constants import IMAGE_TOKEN_INDEX

from .perception_decision import build_perception_decision
from .projectors import build_mlp_projector


def _reaction_projector(config, config_name, input_size):
    projector_type = getattr(config, config_name, None)
    if projector_type != "mlp2x_gelu":
        raise ValueError(
            f"Released Dispider requires {config_name}='mlp2x_gelu'; "
            f"found {projector_type!r}"
        )
    return build_mlp_projector(input_size, config.hidden_size)


class _ReactionModelMixin:
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "perception_decision", None):
            self.compressor = build_perception_decision(config)
            self.compress_projector = _reaction_projector(
                config,
                "decision_projector_type",
                config.decision_hidden_size,
            )
            self.clip_projector = _reaction_projector(
                config,
                "perception_projector_type",
                config.perception_hidden_size,
            )

    @property
    def perception_decision(self):
        return getattr(self, "compressor", None)

    def get_perception_decision(self):
        return self.perception_decision


class _ReactionMultimodalMixin(ABC):
    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    @property
    def perception_decision(self):
        return self.get_model().get_perception_decision()

    def get_perception_decision(self):
        return self.get_model().get_perception_decision()

    def encode_sequences(
        self,
        clips,
        clips_large,
        sequences,
        sequence_mask,
        question_ids,
        question_mask,
        answer_token,
        todo_token,
        insert_position,
        answer_positions,
    ):
        model = self.get_model()
        (
            clip_features,
            clip_embeds,
            global_memory,
            _,
            _,
        ) = model.get_perception_decision()(
            clips,
            clips_large,
            sequences,
            sequence_mask,
            question_ids,
            question_mask,
            answer_token,
            todo_token,
            insert_position,
            answer_positions,
        )

        if isinstance(clip_features, list):
            projected_features = model.compress_projector(
                torch.cat(clip_features, dim=0)
            )
            projected_clips = model.clip_projector(torch.cat(clip_embeds, dim=0))
            projected_memory = model.compress_projector(torch.cat(global_memory, dim=0))
            combined = torch.cat((projected_clips, projected_features), dim=-2)

            clip_index = 0
            memory_index = 0
            merged_features = []
            for current_clips, current_memory in zip(clip_features, global_memory):
                clip_count = current_clips.shape[0]
                memory_count = current_memory.shape[0]
                video_features = combined[clip_index : clip_index + clip_count].view(
                    -1, combined.shape[-1]
                )
                video_memory = projected_memory[
                    memory_index : memory_index + memory_count
                ]
                merged_features.append(torch.cat((video_features, video_memory), dim=0))
                clip_index += clip_count
                memory_index += memory_count
            return merged_features

        projected_features = model.compress_projector(clip_features)
        projected_clips = model.clip_projector(clip_embeds)
        projected_memory = model.compress_projector(global_memory)
        combined = torch.cat((projected_clips, projected_features), dim=-2)
        combined = combined.view(combined.shape[0], -1, combined.shape[-1])
        return torch.cat((combined, projected_memory), dim=-2)

    def _embed_tokens(self, input_ids):
        return self.get_model().embed_tokens(input_ids)

    def _interleave_image_features(self, input_ids, image_features):
        input_embeddings = []
        image_index = 0
        for tokens in input_ids:
            current_image = image_features[image_index]
            image_positions = torch.where(tokens == IMAGE_TOKEN_INDEX)[0].tolist()
            if not image_positions:
                text = self._embed_tokens(tokens)
                input_embeddings.append(torch.cat((text, current_image[0:0]), dim=0))
                image_index += 1
                continue

            boundaries = [-1, *image_positions, tokens.shape[0]]
            text_segments = [
                tokens[start + 1 : end]
                for start, end in zip(boundaries, boundaries[1:])
            ]
            sizes = [segment.shape[0] for segment in text_segments]
            text_embeddings = torch.split(
                self._embed_tokens(torch.cat(text_segments)),
                sizes,
                dim=0,
            )
            combined = []
            for index, text in enumerate(text_embeddings):
                combined.append(text)
                if index < len(image_positions):
                    combined.append(image_features[image_index])
                    image_index += 1
            input_embeddings.append(torch.cat(combined))
        return input_embeddings

    def _pad_multimodal_batch(self, input_embeddings, attention_mask, position_ids):
        max_length = max(item.shape[0] for item in input_embeddings)
        batch_size = len(input_embeddings)
        padded_embeddings = []
        padded_attention = torch.zeros(
            (batch_size, max_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        padded_positions = torch.zeros(
            (batch_size, max_length),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        pad_left = getattr(self.config, "tokenizer_padding_side", "right") == "left"

        for index, embeddings in enumerate(input_embeddings):
            current_length = embeddings.shape[0]
            padding = torch.zeros(
                (max_length - current_length, embeddings.shape[1]),
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            if pad_left:
                padded_embeddings.append(torch.cat((padding, embeddings), dim=0))
                destination = (
                    slice(-current_length, None) if current_length else slice(0, 0)
                )
            else:
                padded_embeddings.append(torch.cat((embeddings, padding), dim=0))
                destination = slice(0, current_length)
            padded_attention[index, destination] = True
            padded_positions[index, destination] = torch.arange(
                current_length,
                dtype=position_ids.dtype,
                device=position_ids.device,
            )

        return (
            torch.stack(padded_embeddings, dim=0),
            padded_attention,
            padded_positions,
        )

    def prepare_inference_inputs(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        clips,
        clips_large,
        sequences,
        sequence_mask,
        question_ids,
        question_mask,
        answer_token,
        todo_token,
        insert_position,
        answer_positions,
    ):
        if self.get_perception_decision() is None or clips is None:
            return input_ids, position_ids, attention_mask, past_key_values, None

        image_features = self.encode_sequences(
            clips,
            clips_large,
            sequences,
            sequence_mask,
            question_ids,
            question_mask,
            answer_token,
            todo_token,
            insert_position,
            answer_positions,
        )

        original_positions = position_ids
        original_attention = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )

        unpadded_ids = [tokens[mask] for tokens, mask in zip(input_ids, attention_mask)]
        input_embeddings = self._interleave_image_features(
            unpadded_ids,
            image_features,
        )
        max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if max_length is not None:
            input_embeddings = [item[:max_length] for item in input_embeddings]
        input_embeddings, attention_mask, position_ids = self._pad_multimodal_batch(
            input_embeddings,
            attention_mask,
            position_ids,
        )
        if original_attention is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=original_attention.dtype)
        if original_positions is None:
            position_ids = None
        return None, position_ids, attention_mask, past_key_values, input_embeddings


class ReactionConfig(Qwen2Config):
    model_type = "dispider_reaction"
    mm_use_im_start_end = False


class _ReactionModel(_ReactionModelMixin, Qwen2Model):
    config_class = ReactionConfig


class Reaction(Qwen2ForCausalLM, _ReactionMultimodalMixin):
    config_class = ReactionConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = _ReactionModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_large: Optional[torch.FloatTensor] = None,
        seqs: Optional[torch.LongTensor] = None,
        compress_mask: Optional[torch.Tensor] = None,
        qs: Optional[torch.LongTensor] = None,
        qs_mask: Optional[torch.Tensor] = None,
        ans_token: Optional[torch.LongTensor] = None,
        todo_token: Optional[torch.LongTensor] = None,
        insert_position: Optional[int] = None,
        ans_position: Optional[list] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        del cache_position
        if inputs_embeds is None and images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
            ) = self.prepare_inference_inputs(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                images,
                images_large,
                seqs,
                compress_mask,
                qs,
                qs_mask,
                ans_token,
                todo_token,
                insert_position,
                ans_position,
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

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        images: Optional[torch.FloatTensor] = None,
        images_large: Optional[torch.FloatTensor] = None,
        seqs: Optional[torch.LongTensor] = None,
        compress_mask: Optional[torch.Tensor] = None,
        qs: Optional[torch.LongTensor] = None,
        qs_mask: Optional[torch.Tensor] = None,
        ans_token: Optional[torch.LongTensor] = None,
        todo_token: Optional[torch.LongTensor] = None,
        insert_position: Optional[int] = None,
        ans_position: Optional[list] = None,
        q_id: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        del q_id
        if "inputs_embeds" in kwargs:
            raise ValueError("Pass input_ids; Reaction assembles its own embeddings")
        if images is not None:
            (
                _,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
            ) = self.prepare_inference_inputs(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                images,
                images_large,
                seqs,
                compress_mask,
                qs,
                qs_mask,
                ans_token,
                todo_token,
                insert_position,
                ans_position,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


__all__ = ["Reaction", "ReactionConfig"]

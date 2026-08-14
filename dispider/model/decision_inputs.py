"""Video and text assembly for the inference-only Decision model."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
import math

import torch

from dispider.constants import IMAGE_TOKEN_INDEX

from .perception import build_perception
from .projectors import build_vision_projector


def _encode_in_chunks(
    images: torch.Tensor,
    encoder: Callable[[torch.Tensor], torch.Tensor],
    block_size: int = 16,
) -> torch.Tensor:
    features = [
        encoder(images[start : start + block_size])
        for start in range(0, images.shape[0], block_size)
    ]
    output = torch.cat(features, dim=0)
    if output.shape[0] != images.shape[0]:
        raise RuntimeError("Perception changed the frame batch size")
    return output


def _split_text_segments(input_ids, indicators, embed_tokens):
    boundaries = (
        [-1]
        + torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        + [input_ids.shape[0]]
    )
    id_segments = []
    indicator_segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        id_segments.append(input_ids[start + 1 : end])
        indicator_segments.append(indicators[start + 1 : end])

    sizes = [segment.shape[0] for segment in id_segments]
    embeddings = embed_tokens(torch.cat(id_segments))
    return torch.split(embeddings, sizes, dim=0), indicator_segments


def _memory_slot_count(projector, image_features: torch.Tensor) -> int:
    pool_num = projector.pool_num
    frame_width = projector.resolution + pool_num
    if (image_features.shape[0] - 1) % frame_width:
        raise ValueError("Projected clip tokens do not match the pool layout")
    count = (image_features.shape[0] - 1) // frame_width * pool_num
    if count < 1:
        raise ValueError("Projected clips contain no Decision memory slots")
    return count


def _interleave_text_and_images(
    text_embeddings: Sequence[torch.Tensor],
    text_indicators: Sequence[torch.Tensor],
    image_features: Sequence[torch.Tensor],
    image_index: int,
    projector,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    output_embeddings = []
    output_indicators = []
    num_images = len(text_embeddings) - 1

    for index in range(num_images + 1):
        output_embeddings.append(text_embeddings[index])
        output_indicators.append(text_indicators[index])
        if index == num_images:
            continue

        current_image = image_features[image_index]
        image_index += 1
        output_embeddings.append(current_image)
        image_indicators = torch.full(
            (current_image.shape[0],),
            2,
            device=text_indicators[index].device,
            dtype=text_indicators[index].dtype,
        )
        memory_slots = _memory_slot_count(projector, current_image)
        image_indicators[-memory_slots - 1 : -1] = 100
        image_indicators[-1] = 200
        output_indicators.append(image_indicators)

    return (
        torch.cat(output_embeddings),
        torch.cat(output_indicators),
        image_index,
    )


def _pad_multimodal_batch(
    embeddings: Sequence[torch.Tensor],
    indicators: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    padding_side: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(item.shape[0] for item in embeddings)
    batch_size = len(embeddings)
    padded_embeddings = []
    padded_indicators = torch.zeros(
        (batch_size, max_length),
        dtype=indicators[0].dtype,
        device=indicators[0].device,
    )
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

    for index, (current_embeddings, current_indicators) in enumerate(
        zip(embeddings, indicators)
    ):
        current_length = current_embeddings.shape[0]
        padding = torch.zeros(
            (max_length - current_length, current_embeddings.shape[1]),
            dtype=current_embeddings.dtype,
            device=current_embeddings.device,
        )
        positions = torch.arange(
            current_length,
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        if padding_side == "left":
            padded_embeddings.append(torch.cat((padding, current_embeddings), dim=0))
            destination = (
                slice(-current_length, None) if current_length else slice(0, 0)
            )
        else:
            padded_embeddings.append(torch.cat((current_embeddings, padding), dim=0))
            destination = slice(0, current_length)
        padded_indicators[index, destination] = current_indicators
        padded_attention[index, destination] = True
        padded_positions[index, destination] = positions

    return (
        torch.stack(padded_embeddings, dim=0),
        padded_indicators,
        padded_attention,
        padded_positions,
    )


class DecisionModelMixin:
    def __init__(self, config):
        super().__init__(config)
        if hasattr(config, "perception_vision_tower"):
            self.vision_tower = build_perception(config)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        return getattr(self, "vision_tower", None)


class DecisionInputsMixin(ABC):
    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def _mix_spatial_tokens(self, features):
        batch, tokens, channels = features.shape
        height = math.isqrt(tokens)
        if height * height != tokens or height % 2:
            raise ValueError("Perception patch tokens must form an even square grid")
        features = (
            features.view(batch, height // 2, 2, height // 2, 2, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
        )
        return features.view(batch, tokens // 4, 4 * channels).contiguous()

    def _encode_clips(self, images):
        if not isinstance(images, torch.Tensor) or images.ndim != 5:
            raise ValueError("Decision expects clips shaped [batch, frames, C, H, W]")
        frames = images.view(-1, *images.shape[2:])
        encoded = _encode_in_chunks(frames, self.get_vision_tower())
        encoded = self._mix_spatial_tokens(encoded)
        encoded = encoded.view(
            images.shape[0],
            images.shape[1] * encoded.shape[1],
            encoded.shape[2],
        )
        clip_features = [item.to(self.device) for item in encoded]
        projected = self.get_model().mm_projector(encoded)
        return clip_features, [item.to(self.device) for item in projected]

    def prepare_inference_inputs(
        self,
        input_ids,
        position_ids,
        attention_mask,
        question_ids,
        question_mask,
        past_key_values,
        images,
        answer_token,
        todo_token,
    ):
        if self.get_vision_tower() is None:
            raise RuntimeError("Decision has no Perception tower")
        if images is None:
            raise ValueError("Decision requires video clips")
        if len(images) != len(input_ids):
            raise ValueError("Each Decision sequence requires one video clip batch")

        question_embeds = self.get_model().embed_tokens(question_ids)
        answer_embed = self.get_model().embed_tokens(answer_token)
        todo_embed = self.get_model().embed_tokens(todo_token)
        clip_features, image_features = self._encode_clips(images)

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
        indicators = torch.zeros_like(input_ids)
        input_ids = [tokens[mask] for tokens, mask in zip(input_ids, attention_mask)]
        indicators = [roles[mask] for roles, mask in zip(indicators, attention_mask)]

        input_embeddings = []
        output_indicators = []
        image_index = 0
        for tokens, roles, current_image in zip(input_ids, indicators, image_features):
            if not torch.any(tokens == IMAGE_TOKEN_INDEX):
                text = self.get_model().embed_tokens(tokens)
                input_embeddings.append(torch.cat((text, current_image[0:0]), dim=0))
                output_indicators.append(roles)
                image_index += 1
                continue

            text_embeddings, text_indicators = _split_text_segments(
                tokens,
                roles,
                self.get_model().embed_tokens,
            )
            embeddings, current_indicators, image_index = _interleave_text_and_images(
                text_embeddings,
                text_indicators,
                image_features,
                image_index,
                self.get_model().mm_projector,
            )
            input_embeddings.append(embeddings)
            output_indicators.append(current_indicators)

        max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if max_length is not None:
            input_embeddings = [item[:max_length] for item in input_embeddings]
            output_indicators = [item[:max_length] for item in output_indicators]

        (
            input_embeddings,
            output_indicators,
            attention_mask,
            position_ids,
        ) = _pad_multimodal_batch(
            input_embeddings,
            output_indicators,
            attention_mask,
            position_ids,
            getattr(self.config, "tokenizer_padding_side", "right"),
        )
        if original_attention is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=original_attention.dtype)
        if original_positions is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            input_embeddings,
            clip_features,
            question_embeds,
            question_mask,
            output_indicators,
            answer_embed,
            todo_embed,
        )


__all__ = ["DecisionInputsMixin", "DecisionModelMixin"]

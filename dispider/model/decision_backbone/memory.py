"""Shared tensor assembly for Decision memory and query passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .constants import (
    DEFAULT_SELECTED_CLIPS,
    DYNAMIC_SELECTION_THRESHOLD,
    GLOBAL_MEMORY_TOKEN,
    MAX_SELECTED_CLIPS,
    MEMORY_TOKEN,
    QUERY_TOKEN,
    SIMILARITY_TEMPERATURE,
    TIME_TOKEN,
)


@dataclass(frozen=True)
class PerceptionMemory:
    """Hidden states extracted from a batch of encoded clips."""

    clip: Tensor
    time: Tensor
    per_clip: Tensor


@dataclass(frozen=True)
class QueryBatch:
    """Aligned embeddings, masks, and token roles for a decoder pass."""

    embeddings: Tensor
    mask: Tensor
    indicators: Tensor


def run_decoder(
    owner,
    input_ids: Optional[Tensor],
    embeddings: Tensor,
    mask: Tensor,
    indicators: Tensor,
    select_layer: Optional[int],
    *,
    past_key_values=None,
):
    """Run the owner's bare decoder with the released Decision arguments."""

    return owner.model(
        input_ids,
        inputs_embeds=embeddings,
        past_key_values=past_key_values,
        attention_mask=mask,
        indicators=indicators,
        select_layer=select_layer,
        use_cache=False,
    )


def extract_perception_memory(
    owner,
    input_ids: Optional[Tensor],
    embeddings: Tensor,
    mask: Tensor,
    indicators: Tensor,
    select_layer: Optional[int],
    *,
    past_key_values=None,
) -> PerceptionMemory:
    """Encode clips and collect their memory and time token states."""

    hidden = run_decoder(
        owner,
        input_ids,
        embeddings,
        mask,
        indicators,
        select_layer,
        past_key_values=past_key_values,
    )[0]
    clip = hidden[indicators == MEMORY_TOKEN]
    time = hidden[indicators == TIME_TOKEN]
    count = embeddings.shape[0]
    return PerceptionMemory(
        clip=clip,
        time=time,
        per_clip=clip.view(count, -1, hidden.shape[-1]),
    )


def summarize_memory(
    owner,
    input_ids: Optional[Tensor],
    memory: PerceptionMemory,
    select_layer: Optional[int],
    *,
    mark_global: bool = False,
    past_key_values=None,
) -> Tensor:
    """Compress all clip and time states into one state per clip."""

    embeddings = torch.cat((memory.clip, memory.time), dim=0).unsqueeze(0)
    mask = torch.ones_like(embeddings[:, :, 0])
    indicators = torch.ones_like(mask)
    if mark_global:
        indicators[:, -memory.time.shape[0] :] = GLOBAL_MEMORY_TOKEN
    hidden = run_decoder(
        owner,
        input_ids,
        embeddings,
        mask,
        indicators,
        select_layer,
        past_key_values=past_key_values,
    )[0]
    return hidden[0, -memory.time.shape[0] :]


def summarize_range(
    owner,
    input_ids: Optional[Tensor],
    memory: PerceptionMemory,
    start: int,
    end: int,
    select_layer: Optional[int],
) -> Tensor:
    """Compress a half-open clip range into its time-token summaries."""

    count = memory.per_clip.shape[0]
    per_memory = memory.clip.shape[0] // count
    per_time = memory.time.shape[0] // count
    embeddings = torch.cat(
        (
            memory.clip[start * per_memory : end * per_memory],
            memory.time[start * per_time : end * per_time],
        ),
        dim=0,
    ).unsqueeze(0)
    mask = torch.ones_like(embeddings[:, :, 0])
    indicators = torch.ones_like(mask)
    hidden = run_decoder(
        owner,
        input_ids,
        embeddings,
        mask,
        indicators,
        select_layer,
    )[0]
    return hidden[0, -(end - start) * per_time :]


def make_query_batch(
    prefixes: Sequence[Sequence[Tensor]],
    questions: Tensor,
    question_mask: Tensor,
    *,
    suffixes: Optional[Sequence[Sequence[Tensor]]] = None,
) -> QueryBatch:
    """Append questions to variable prefixes and mark their final live token."""

    embeddings: List[Tensor] = []
    masks: List[Tensor] = []
    indicators: List[Tensor] = []
    if len(prefixes) != questions.shape[0]:
        raise ValueError("one prefix is required for each question")
    if suffixes is None:
        suffixes = [()] * len(prefixes)

    for prefix, question, mask, suffix in zip(
        prefixes, questions, question_mask, suffixes
    ):
        parts = [*prefix, question, *suffix]
        current = torch.cat(parts, dim=0)
        prefix_length = sum(part.shape[0] for part in prefix)
        suffix_length = sum(part.shape[0] for part in suffix)
        current_mask = torch.cat(
            (
                torch.ones(
                    prefix_length,
                    dtype=mask.dtype,
                    device=mask.device,
                ),
                mask,
                torch.ones(
                    suffix_length,
                    dtype=mask.dtype,
                    device=mask.device,
                ),
            ),
            dim=0,
        )
        current_indicators = torch.zeros_like(current_mask)
        current_indicators[torch.where(current_mask == 1)[0][-1]] = QUERY_TOKEN
        embeddings.append(current)
        masks.append(current_mask)
        indicators.append(current_indicators)

    return QueryBatch(
        embeddings=torch.stack(embeddings, dim=0),
        mask=torch.stack(masks, dim=0),
        indicators=torch.stack(indicators, dim=0),
    )


def make_all_ones_query(parts: Sequence[Tensor]) -> QueryBatch:
    """Build a single unpadded query and mark its final token."""

    embeddings = torch.cat(parts, dim=0).unsqueeze(0)
    mask = torch.ones_like(embeddings[:, :, 0])
    indicators = torch.zeros_like(mask)
    indicators[:, -1] = QUERY_TOKEN
    return QueryBatch(embeddings, mask, indicators)


def pad_and_concat(batches: Sequence[QueryBatch]) -> QueryBatch:
    """Right-pad query batches and concatenate their batch dimensions."""

    max_length = max(batch.embeddings.shape[1] for batch in batches)
    embeddings: List[Tensor] = []
    masks: List[Tensor] = []
    indicators: List[Tensor] = []
    for batch in batches:
        padding = max_length - batch.embeddings.shape[1]
        embeddings.append(torch.nn.functional.pad(batch.embeddings, (0, 0, 0, padding)))
        masks.append(torch.nn.functional.pad(batch.mask, (0, padding)))
        indicators.append(torch.nn.functional.pad(batch.indicators, (0, padding)))
    return QueryBatch(
        torch.cat(embeddings, dim=0),
        torch.cat(masks, dim=0),
        torch.cat(indicators, dim=0),
    )


def query_states(hidden: Tensor, indicators: Tensor) -> Tensor:
    """Collect the marked query state from each sequence."""

    return hidden[indicators == QUERY_TOKEN]


def similarity(query: Tensor, time: Tensor, *, normalize: bool = True) -> Tensor:
    """Compute query-to-time cosine similarities."""

    query = torch.nn.functional.normalize(query, dim=1, p=2)
    time = torch.nn.functional.normalize(time, dim=1, p=2)
    scores = torch.einsum("qc,nc->qn", query, time)
    if normalize:
        scores = torch.softmax(scores / SIMILARITY_TEMPERATURE, dim=-1)
    return scores


def selected_count(raw_similarity: Tensor) -> int:
    """Choose the released dynamic number of clips for offline inference."""

    count = int(torch.sum(raw_similarity > DYNAMIC_SELECTION_THRESHOLD).item())
    return min(max(DEFAULT_SELECTED_CLIPS, count), MAX_SELECTED_CLIPS)


def select_topk(
    scores: Tensor,
    memory: PerceptionMemory,
    clip_embeddings: Tensor,
    *,
    count: int = DEFAULT_SELECTED_CLIPS,
) -> Tuple[Tensor, Tensor]:
    """Select chronologically ordered clip states for every query."""

    indices = torch.topk(
        scores,
        dim=-1,
        k=min(count, memory.time.shape[0]),
    )[1].sort(
        dim=-1
    )[0]
    selected_memory = torch.stack([memory.per_clip[index] for index in indices], dim=0)
    selected_clips = torch.stack([clip_embeddings[index] for index in indices], dim=0)
    return selected_memory, selected_clips

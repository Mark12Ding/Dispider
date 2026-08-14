"""Inference and clip-selection paths for the streaming Decision model."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .constants import (
    DEFAULT_SELECTED_CLIPS,
    MEMORY_TOKEN,
    SIMILARITY_TEMPERATURE,
)
from .memory import (
    PerceptionMemory,
    QueryBatch,
    extract_perception_memory,
    make_all_ones_query,
    make_query_batch,
    pad_and_concat,
    query_states,
    run_decoder,
    select_topk,
    selected_count,
    similarity,
    summarize_memory,
    summarize_range,
)


def _segment_summaries(
    owner,
    input_ids: Optional[Tensor],
    memory: PerceptionMemory,
    insert_position: int,
    answer_positions: Sequence[int],
    select_layer: Optional[int],
) -> List[Optional[Tensor]]:
    summaries: List[Optional[Tensor]] = []
    previous = 0
    if insert_position == 0:
        summaries.append(None)
    if insert_position not in answer_positions and insert_position > 0:
        summaries.append(
            summarize_range(
                owner,
                input_ids,
                memory,
                0,
                insert_position,
                select_layer,
            )
        )
        previous = insert_position
    for position in answer_positions:
        summaries.append(
            summarize_range(
                owner,
                input_ids,
                memory,
                previous,
                position,
                select_layer,
            )
        )
        previous = position
    return summaries


def _conversation_query(
    memory: PerceptionMemory,
    summaries: Sequence[Optional[Tensor]],
    memory_splits: Sequence[int],
    position: int,
    insert_position: int,
    answer_positions: Sequence[int],
    question: Tensor,
    answer_token: Tensor,
    todo_token: Tensor,
) -> QueryBatch:
    per_clip = memory.clip.shape[0] // memory.per_clip.shape[0]
    if position == insert_position:
        parts = [answer_token]
        if insert_position > 0:
            parts.append(memory.clip[: position * per_clip])
        parts.extend((question, todo_token))
        return make_all_ones_query(parts)

    earlier = np.where(np.asarray(memory_splits) < position)[0]
    first = int(earlier[0])
    if insert_position == 0:
        parts = [answer_token, question]
    elif insert_position not in answer_positions:
        parts = [answer_token, summaries[first], question]
    else:
        parts = [answer_token, summaries[first], question, answer_token]
    for index in earlier[1:]:
        parts.extend((summaries[int(index)], answer_token))
    start = memory_splits[int(earlier[-1])] * per_clip
    parts.extend((memory.clip[start : position * per_clip], question, todo_token))
    return make_all_ones_query(parts)


def _partial_memory_batch(memory: PerceptionMemory, position: int) -> QueryBatch:
    count = memory.per_clip.shape[0]
    per_clip = memory.clip.shape[0] // count
    per_time = memory.time.shape[0] // count
    embeddings = torch.cat(
        (
            memory.clip[: position * per_clip],
            memory.time[: position * per_time],
        ),
        dim=0,
    ).unsqueeze(0)
    mask = torch.ones_like(embeddings[:, :, 0])
    indicators = torch.zeros_like(mask)
    if position:
        indicators[:, -position * per_time :] = MEMORY_TOKEN
    return QueryBatch(embeddings, mask, indicators)


class DecisionStreamingMixin:
    """Dispider's released online trigger and memory-selection behavior."""

    def _perception_memory(
        self,
        input_ids,
        inputs_embeds,
        attention_mask,
        indicators,
        select_layer,
        past_key_values,
    ) -> PerceptionMemory:
        with torch.no_grad():
            return extract_perception_memory(
                self,
                input_ids,
                inputs_embeds,
                attention_mask,
                indicators,
                select_layer,
                past_key_values=past_key_values,
            )

    def _offline_video_selection(
        self,
        input_ids,
        memory: PerceptionMemory,
        clip_embeds,
        qs_embeds,
        qs_mask,
        ans_token,
        select_layer,
    ):
        with torch.no_grad():
            global_memory = summarize_memory(
                self,
                input_ids,
                memory,
                select_layer,
            )
            prefixes = [(ans_token[0], memory.clip) for _ in range(qs_embeds.shape[0])]
            batch = make_query_batch(prefixes, qs_embeds, qs_mask)
            hidden = run_decoder(
                self,
                input_ids,
                batch.embeddings,
                batch.mask,
                batch.indicators,
                select_layer,
            )[0]
            query = query_states(hidden, batch.indicators)

        raw_scores = similarity(query, memory.time, normalize=False)
        count = selected_count(raw_scores)
        scores = torch.softmax(
            raw_scores / SIMILARITY_TEMPERATURE,
            dim=-1,
        )
        selected_memory, selected_clips = select_topk(
            scores,
            memory,
            clip_embeds,
            count=count,
        )
        global_memory = global_memory.unsqueeze(0).repeat([qs_embeds.shape[0], 1, 1])
        return selected_memory, selected_clips, global_memory, 0, scores

    def _streaming_selection(
        self,
        input_ids,
        memory: PerceptionMemory,
        clip_embeds,
        qs_embeds,
        insert_position,
        ans_position,
        ans_token,
        todo_token,
        select_layer,
    ):
        with torch.no_grad():
            summaries = _segment_summaries(
                self,
                input_ids,
                memory,
                insert_position,
                ans_position,
                select_layer,
            )
            memory_splits = (
                ans_position
                if insert_position in ans_position
                else [insert_position, *ans_position]
            )
            positions = [memory.per_clip.shape[0]]
            batches = []
            for position in positions:
                batches.extend(
                    (
                        _partial_memory_batch(memory, position),
                        _conversation_query(
                            memory,
                            summaries,
                            memory_splits,
                            position,
                            insert_position,
                            ans_position,
                            qs_embeds[0],
                            ans_token[0],
                            todo_token[0],
                        ),
                    )
                )
            batch = pad_and_concat(batches)
            hidden = run_decoder(
                self,
                input_ids,
                batch.embeddings,
                batch.mask,
                batch.indicators,
                select_layer,
            )[0]

        query = query_states(hidden, batch.indicators)
        scores = similarity(query, memory.time)
        selected_memory = []
        selected_clips = []
        global_memory = []
        for index, position in enumerate(positions):
            selected = torch.topk(
                scores[index, :position],
                dim=-1,
                k=min(DEFAULT_SELECTED_CLIPS, position),
            )[1].sort(dim=-1)[0]
            selected_memory.append(memory.per_clip[selected])
            selected_clips.append(clip_embeds[selected])
            global_memory.append(
                hidden[2 * index][batch.indicators[2 * index] == MEMORY_TOKEN]
            )

        is_silent = self.silent_head(query)[:, 0]
        return selected_memory, selected_clips, global_memory, is_silent, scores

    def forward_token_stream(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        inputs_embeds: Optional[Tensor] = None,
        clip_embeds: Optional[Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        qs_embeds: Optional[Tensor] = None,
        qs_mask: Optional[torch.BoolTensor] = None,
        indicators: Optional[torch.LongTensor] = None,
        select_layer: Optional[int] = None,
        insert_position: Optional[int] = None,
        ans_position: Optional[list] = None,
        ans_token: Optional[Tensor] = None,
        todo_token: Optional[Tensor] = None,
    ):
        memory = self._perception_memory(
            input_ids,
            inputs_embeds,
            attention_mask,
            indicators,
            select_layer,
            past_key_values,
        )

        if ans_position is None and insert_position is None:
            return self._offline_video_selection(
                input_ids,
                memory,
                clip_embeds,
                qs_embeds,
                qs_mask,
                ans_token,
                select_layer,
            )

        if ans_position is not None:
            return self._streaming_selection(
                input_ids,
                memory,
                clip_embeds,
                qs_embeds,
                insert_position,
                ans_position,
                ans_token,
                todo_token,
                select_layer,
            )
        raise ValueError("insert_position requires answer positions")

    def infer_trigger_sequence(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        inputs_embeds: Optional[Tensor] = None,
        clip_embeds=None,
        attention_mask: Optional[torch.BoolTensor] = None,
        qs_embeds: Optional[Tensor] = None,
        qs_mask=None,
        indicators: Optional[torch.LongTensor] = None,
        select_layer: Optional[int] = None,
        ans_token: Optional[Tensor] = None,
        todo_token: Optional[Tensor] = None,
    ):
        del clip_embeds, qs_mask
        memory = self._perception_memory(
            input_ids,
            inputs_embeds,
            attention_mask,
            indicators,
            select_layer,
            past_key_values,
        )
        previous = 0
        answer_positions = []
        summaries = []
        silent_scores = []
        for position in range(1, inputs_embeds.shape[0] + 1):
            if not answer_positions:
                parts = [
                    ans_token[0],
                    qs_embeds[0],
                    memory.clip[: position * memory.per_clip.shape[1]],
                    qs_embeds[0],
                    todo_token[0],
                ]
            else:
                parts = [ans_token[0], qs_embeds[0]]
                for summary in summaries:
                    parts.extend((summary, ans_token[0]))
                parts.extend(
                    (
                        memory.clip[
                            previous
                            * memory.per_clip.shape[1] : position
                            * memory.per_clip.shape[1]
                        ],
                        qs_embeds[0],
                        todo_token[0],
                    )
                )
            batch = make_all_ones_query(parts)
            with torch.no_grad():
                hidden = run_decoder(
                    self,
                    input_ids,
                    batch.embeddings,
                    batch.mask,
                    batch.indicators,
                    select_layer,
                )[0]
                query = query_states(hidden, batch.indicators)
                score = self.silent_head(query)[:, 0]
            if score.item() > 0:
                with torch.no_grad():
                    summaries.append(
                        summarize_range(
                            self,
                            input_ids,
                            memory,
                            previous,
                            position,
                            select_layer,
                        )
                    )
                answer_positions.append(position)
                previous = position
            silent_scores.append(score.item())
        return answer_positions, silent_scores

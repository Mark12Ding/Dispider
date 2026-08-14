"""Offline single-video inference entry point for Dispider."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList

from dispider.constants import (
    DEFAULT_ANS_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_SILENT_TOKEN,
    DEFAULT_TODO_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from dispider.conversation import conv_templates
from dispider.mm_utils import get_model_name_from_path, tokenizer_image_token
from dispider.model.builder import load_pretrained_model


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generation when the first sequence ends with a configured token."""

    def __init__(
        self,
        stops: Optional[Sequence[torch.Tensor]] = None,
        encounters: int = 1,
    ) -> None:
        super().__init__()
        del encounters
        self.stops = tuple(stops or ())

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        del scores, kwargs
        for stop in self.stops:
            stop_length = stop.numel()
            if stop_length == 0 or input_ids.shape[1] < stop_length:
                continue
            if torch.equal(stop.reshape(-1), input_ids[0, -stop_length:]):
                return True
        return False


def get_seq_frames(total_frames: int, desired_frames: int) -> List[int]:
    if total_frames <= 0:
        raise ValueError("video must contain at least one frame")
    if desired_frames <= 0:
        raise ValueError("desired_num_frames must be positive")

    segment_size = float(total_frames - 1) / desired_frames
    frame_indices = []
    for index in range(desired_frames):
        start = int(np.round(segment_size * index))
        end = int(np.round(segment_size * (index + 1)))
        frame_indices.append((start + end) // 2)
    return frame_indices


def get_seq_time(
    video_reader: VideoReader,
    frame_indices: Sequence[int],
    num_clips: int,
) -> np.ndarray:
    frames_per_clip = len(frame_indices) // num_clips
    key_frames = [
        [
            frame_indices[index * frames_per_clip],
            frame_indices[(index + 1) * frames_per_clip - 1],
        ]
        for index in range(num_clips)
    ]
    timestamps = video_reader.get_frame_timestamp(key_frames)
    return np.hstack([timestamps[:, 0, 0], timestamps[:, 1, 1]])


def calculate_diff(boundaries: Sequence[int], start_frame: int) -> List[int]:
    differences = [boundaries[0] - start_frame]
    for index in range(len(boundaries) - 1):
        differences.append(boundaries[index + 1] - boundaries[index])
    return differences


def _local_video_path(video_path: os.PathLike[str] | str) -> str:
    path = os.fspath(video_path)
    if path.lower().startswith("s3://"):
        reason = "S3 video paths are not supported"
        raise ValueError(f"{reason}; download the video locally first")
    return path


def _frame_range(
    video_length: int,
    sample_frame: Optional[Sequence[Sequence[int]]],
) -> Tuple[int, int]:
    if sample_frame is None:
        return 0, video_length
    if len(sample_frame) == 0 or len(sample_frame[0]) != 2:
        raise ValueError("sample_frame must contain one [start, end] range")
    start_frame, end_frame = map(int, sample_frame[0])
    if start_frame < 0 or end_frame > video_length or start_frame >= end_frame:
        raise ValueError("sample_frame range is outside the video")
    return start_frame, end_frame


def load_video(
    vis_path: os.PathLike[str] | str,
    scene_sep: Sequence[float],
    num_frm: int = 16,
    max_clip: int = 4,
    sample_frame: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[List[Image.Image], np.ndarray, int]:
    if num_frm <= 0 or max_clip <= 0:
        raise ValueError("num_frm and max_clip must be positive")

    video_reader = VideoReader(_local_video_path(vis_path), num_threads=1)
    range_start, range_end = _frame_range(len(video_reader), sample_frame)
    total_frame_count = range_end - range_start
    frames_per_second = float(video_reader.get_avg_fps())
    if frames_per_second <= 0:
        raise ValueError("video FPS must be positive")

    if len(scene_sep) == 0:
        total_time = total_frame_count / frames_per_second
        num_clips = int(np.round(total_time / num_frm))
        num_clips = min(max(num_clips, 1), max_clip)
        sample_count = num_frm * num_clips
        frame_indices = [
            range_start + index
            for index in get_seq_frames(total_frame_count, sample_count)
        ]
    else:
        scene_ends = [
            max(
                range_start,
                min(
                    int(frames_per_second * (boundary + 1)),
                    range_end - 1,
                ),
            )
            for boundary in scene_sep
        ]
        scene_ends.append(range_end - 1)
        if len(scene_ends) > max_clip:
            differences = calculate_diff(scene_ends, range_start)
            remove_count = len(scene_ends) - max_clip
            remove_indices = np.argsort(differences[:-1])[:remove_count]
            for index in np.sort(remove_indices)[::-1]:
                del scene_ends[int(index)]

        frame_indices = []
        segment_start = range_start
        for segment_end in scene_ends:
            indices = np.linspace(
                segment_start,
                segment_end,
                num=num_frm,
                endpoint=False,
            )
            frame_indices.extend(int(index) for index in indices)
            segment_start = segment_end
        num_clips = len(scene_ends)
        sample_count = num_frm * num_clips

    time_indices = get_seq_time(video_reader, frame_indices, num_clips)
    image_array = video_reader.get_batch(frame_indices).asnumpy()
    _, height, width, _ = image_array.shape
    if height != width:
        square_size = min(height, width)
        image_tensor = torch.from_numpy(image_array)
        image_tensor = image_tensor.permute(0, 3, 1, 2).float()
        image_tensor = torch.nn.functional.interpolate(
            image_tensor,
            size=(square_size, square_size),
        )
        image_array = image_tensor.permute(0, 2, 3, 1)
        image_array = image_array.to(torch.uint8).numpy()

    image_array = image_array.reshape(
        1,
        sample_count,
        image_array.shape[-3],
        image_array.shape[-2],
        image_array.shape[-1],
    )
    to_image = Image.fromarray
    frames = [to_image(image_array[0, index]) for index in range(sample_count)]
    return frames, time_indices, num_clips


def preprocess_time(
    time: np.ndarray, num_clip: int, tokenizer: Any
) -> List[torch.Tensor]:
    time = time.reshape(2, num_clip)
    sequences = []
    for index in range(num_clip):
        start, end = time[:, index]
        sentence = (
            "This contains a clip sampled in "
            f"{int(np.round(start))} to {int(np.round(end))} seconds"
            f"{DEFAULT_IMAGE_TOKEN}"
        )
        sequences.append(
            tokenizer_image_token(
                sentence,
                tokenizer,
                return_tensors="pt",
            )
        )
    return sequences


def preprocess_question(
    questions: Sequence[str],
    tokenizer: Any,
) -> List[torch.Tensor]:
    return [
        tokenizer_image_token(
            question + DEFAULT_TODO_TOKEN,
            tokenizer,
            return_tensors="pt",
        )
        for question in questions
    ]


def process_data(
    video_id: os.PathLike[str] | str,
    scene_sep: Sequence[float],
    question: str,
    model_config: Any,
    tokenizer: Any,
    processor: Any,
    processor_large: Any,
    time_tokenizer: Any,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del processor_large
    num_frames = 16
    max_clips = 100
    if getattr(model_config, "mm_use_im_start_end", False):
        user_prompt = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + question
        )
    else:
        user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + question

    conversation = conv_templates["qwen"].copy()
    conversation.append_message(conversation.roles[0], user_prompt)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()

    frames, time_indices, num_clips = load_video(
        video_id,
        scene_sep,
        num_frames,
        max_clips,
    )
    pixel_values = processor.preprocess(frames, return_tensors="pt")
    video = pixel_values["pixel_values"]
    video = video.view(num_clips, num_frames, *video.shape[1:])
    video_large = video[:, :1].contiguous()

    sequences = preprocess_time(time_indices, num_clips, time_tokenizer)
    sequences = torch.nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=time_tokenizer.pad_token_id,
    )
    compress_mask = sequences.ne(time_tokenizer.pad_token_id)

    question_ids = preprocess_question([question], time_tokenizer)
    question_ids = torch.nn.utils.rnn.pad_sequence(
        question_ids,
        batch_first=True,
        padding_value=time_tokenizer.pad_token_id,
    )
    question_mask = question_ids.ne(time_tokenizer.pad_token_id)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    return (
        input_ids,
        video,
        video_large,
        sequences,
        compress_mask,
        question_ids,
        question_mask,
    )


def _model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and torch.device(device).type != "meta":
        return torch.device(device)
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if parameter.device.type == "meta":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return parameter.device


@dataclass(frozen=True)
class ReactionDecision:
    """One Reaction generation with its explicit silence decision."""

    should_respond: bool
    response: str
    raw_response: str


class VideoStream:
    """Offline single-video Dispider inference runtime."""

    def __init__(self, model_path: os.PathLike[str] | str) -> None:
        expanded_path = os.path.expanduser(os.fspath(model_path))
        model_name = get_model_name_from_path(expanded_path)
        (
            self.tokenizer,
            self.model,
            image_processors,
            self.context_len,
        ) = load_pretrained_model(expanded_path, None, model_name)
        self.image_processor, self.time_tokenizer = image_processors
        self.image_processor_large = self.image_processor
        if self.time_tokenizer.pad_token is None:
            self.time_tokenizer.pad_token = "<pad>"

        self.device = _model_device(self.model)
        stop_token = torch.as_tensor(
            self.tokenizer("<|im_end|>").input_ids,
            device=self.device,
        ).reshape(-1)
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=[stop_token])]
        )
        self._ans_token = self.time_tokenizer(
            DEFAULT_ANS_TOKEN,
            return_tensors="pt",
        ).input_ids
        self._todo_token = self.time_tokenizer(
            DEFAULT_TODO_TOKEN,
            return_tensors="pt",
        ).input_ids

    def _to_device(
        self,
        tensor: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return tensor.to(
            device=self.device,
            dtype=dtype,
            non_blocking=self.device.type == "cuda",
        )

    def react(
        self,
        file: os.PathLike[str] | str,
        prompt: str,
    ) -> ReactionDecision:
        """Let the 7B Reaction model decide between silence and a response."""

        (
            input_ids,
            image_tensor,
            image_tensor_large,
            sequences,
            compress_mask,
            question_ids,
            question_mask,
        ) = process_data(
            file,
            [],
            prompt,
            self.model.config,
            self.tokenizer,
            self.image_processor,
            self.image_processor_large,
            self.time_tokenizer,
        )
        input_ids = self._to_device(input_ids.unsqueeze(0))
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=self._to_device(image_tensor, dtype=torch.float16),
                images_large=self._to_device(
                    image_tensor_large,
                    dtype=torch.float16,
                ),
                seqs=self._to_device(sequences),
                compress_mask=self._to_device(compress_mask),
                qs=self._to_device(question_ids),
                qs_mask=self._to_device(question_mask),
                ans_token=self._to_device(self._ans_token),
                todo_token=self._to_device(self._todo_token),
                q_id=None,
                insert_position=0,
                ans_position=[],
                do_sample=False,
                max_new_tokens=1024,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self.stopping_criteria,
                use_cache=True,
            )
        response = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0].strip()
        raw_response = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=False,
        )[0].strip()
        keep_silent = DEFAULT_SILENT_TOKEN in raw_response
        return ReactionDecision(
            should_respond=bool(response) and not keep_silent,
            response="" if keep_silent else response,
            raw_response=raw_response,
        )

    def run(self, file: os.PathLike[str] | str, prompt: str) -> str:
        return self.react(file, prompt).response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run video inference.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model repository.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the video file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for the model.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    streamer = VideoStream(args.model_path)
    print(streamer.run(args.video_path, args.prompt))


if __name__ == "__main__":
    main()

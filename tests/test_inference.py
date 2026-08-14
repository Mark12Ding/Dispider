from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Sequence

import numpy as np
import pytest
import torch

import inference


class FakeBatch:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def asnumpy(self) -> np.ndarray:
        return self._array


class FakeVideoReader:
    def __init__(self, path: str, num_threads: int) -> None:
        self.path = path
        self.num_threads = num_threads
        self.requested_frames: List[int] = []

    def __len__(self) -> int:
        return 100

    def get_avg_fps(self) -> float:
        return 10.0

    def get_frame_timestamp(
        self,
        frames: Sequence[Sequence[int]],
    ) -> np.ndarray:
        timestamps = np.zeros((len(frames), 2, 2), dtype=np.float32)
        for index, (start, end) in enumerate(frames):
            timestamps[index, 0] = (start / 10.0, (start + 1) / 10.0)
            timestamps[index, 1] = (end / 10.0, (end + 1) / 10.0)
        return timestamps

    def get_batch(self, frame_indices: Sequence[int]) -> FakeBatch:
        self.requested_frames = list(frame_indices)
        frames = np.zeros((len(frame_indices), 4, 6, 3), dtype=np.uint8)
        for index in range(len(frame_indices)):
            frames[index].fill(index)
        return FakeBatch(frames)


def test_stopping_criteria_handles_suffix_and_short_input() -> None:
    criterion = inference.StoppingCriteriaSub([torch.tensor([8, 9])])
    scores = torch.empty(1)

    assert criterion(torch.tensor([[1, 8, 9]]), scores)
    assert not criterion(torch.tensor([[9]]), scores)
    assert not criterion(torch.tensor([[1, 8, 7]]), scores)


def test_s3_path_fails_explicitly_before_video_reader() -> None:
    with pytest.raises(ValueError, match="S3 video paths are not supported"):
        inference.load_video("s3://bucket/video.mp4", [])


def test_scene_sampling_initializes_state(monkeypatch: Any) -> None:
    readers: List[FakeVideoReader] = []

    def reader_factory(path: str, num_threads: int) -> FakeVideoReader:
        reader = FakeVideoReader(path, num_threads)
        readers.append(reader)
        return reader

    monkeypatch.setattr(inference, "VideoReader", reader_factory)

    frames, timestamps, num_clips = inference.load_video(
        "video.mp4",
        [1.0, 3.0],
        num_frm=2,
        max_clip=4,
    )

    assert num_clips == 3
    assert len(frames) == 6
    assert timestamps.shape == (6,)
    assert readers[0].requested_frames == [0, 10, 20, 30, 40, 69]
    assert all(frame.size == (4, 4) for frame in frames)


class CountingProcessor:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def preprocess(
        self, frames: Sequence[Any], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        self.calls += 1
        if self.fail:
            raise AssertionError("large processor must not run")
        assert return_tensors == "pt"
        values = torch.arange(
            len(frames) * 3 * 2 * 2,
            dtype=torch.float32,
        )
        return {"pixel_values": values.view(len(frames), 3, 2, 2)}


class FakeTimeTokenizer:
    pad_token_id = 0
    pad_token = None

    def __call__(self, text: str, return_tensors: str) -> Any:
        del text, return_tensors
        return SimpleNamespace(input_ids=torch.tensor([[12]]))


def test_process_data_preprocesses_frames_once(monkeypatch: Any) -> None:
    frames = [object() for _ in range(32)]
    timestamps = np.array([0.0, 1.0, 1.0, 2.0])
    monkeypatch.setattr(
        inference,
        "load_video",
        lambda *args, **kwargs: (frames, timestamps, 2),
    )

    def tokenize(
        text: str,
        tokenizer: Any,
        image_token_index: int = inference.IMAGE_TOKEN_INDEX,
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        del text, tokenizer, image_token_index, return_tensors
        return torch.tensor([4, 5])

    monkeypatch.setattr(inference, "tokenizer_image_token", tokenize)
    processor = CountingProcessor()
    large_processor = CountingProcessor(fail=True)

    result = inference.process_data(
        "video.mp4",
        [],
        "question",
        SimpleNamespace(mm_use_im_start_end=False),
        object(),
        processor,
        large_processor,
        FakeTimeTokenizer(),
    )

    _, video, video_large, sequences, mask, question, question_mask = result
    assert processor.calls == 1
    assert large_processor.calls == 0
    assert video.shape == (2, 16, 3, 2, 2)
    assert torch.equal(video_large, video[:, :1])
    assert sequences.shape == (2, 2)
    assert torch.equal(mask, sequences.ne(0))
    assert torch.equal(question_mask, question.ne(0))


class FakeTokenizer:
    eos_token_id = 2

    def __call__(self, text: str) -> Any:
        del text
        return SimpleNamespace(input_ids=[9])

    def batch_decode(
        self, output_ids: torch.Tensor, skip_special_tokens: bool
    ) -> List[str]:
        del output_ids
        return ["  answer  " if skip_special_tokens else "answer<|im_end|>"]


class FakeModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(mm_use_im_start_end=False)
        self.generate_args: Any = None

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.generate_args = (input_ids, kwargs)
        return torch.tensor([[10, 11]])


def test_run_and_generation_contract(monkeypatch: Any) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    processor = object()
    time_tokenizer = FakeTimeTokenizer()
    monkeypatch.setattr(
        inference,
        "load_pretrained_model",
        lambda *args: (tokenizer, model, (processor, time_tokenizer), 4096),
    )
    tensors = (
        torch.tensor([1, 2]),
        torch.ones(1, 16, 3, 2, 2),
        torch.ones(1, 1, 3, 2, 2),
        torch.tensor([[3, 4]]),
        torch.tensor([[True, True]]),
        torch.tensor([[5, 6]]),
        torch.tensor([[True, True]]),
    )
    monkeypatch.setattr(inference, "process_data", lambda *args: tensors)

    stream = inference.VideoStream("model")
    output = stream.run("video.mp4", "prompt")

    assert output == "answer"
    input_ids, kwargs = model.generate_args
    assert input_ids.shape == (1, 2)
    assert kwargs["images"].dtype == torch.float16
    assert kwargs["images_large"].dtype == torch.float16
    assert kwargs["q_id"] is None
    assert kwargs["insert_position"] == 0
    assert kwargs["ans_position"] == []
    assert kwargs["do_sample"] is False
    assert kwargs["max_new_tokens"] == 1024
    assert kwargs["use_cache"] is True


def test_reaction_poll_detects_keep_silent_without_a_second_generation(
    monkeypatch: Any,
) -> None:
    class SilentTokenizer(FakeTokenizer):
        def batch_decode(
            self,
            output_ids: torch.Tensor,
            skip_special_tokens: bool,
        ) -> List[str]:
            del output_ids
            return [
                (
                    "leaked text"
                    if skip_special_tokens
                    else "<keep_silent>leaked text<|im_end|>"
                )
            ]

    tokenizer = SilentTokenizer()
    model = FakeModel()
    processor = object()
    time_tokenizer = FakeTimeTokenizer()
    monkeypatch.setattr(
        inference,
        "load_pretrained_model",
        lambda *args: (tokenizer, model, (processor, time_tokenizer), 4096),
    )
    tensors = (
        torch.tensor([1, 2]),
        torch.ones(1, 16, 3, 2, 2),
        torch.ones(1, 1, 3, 2, 2),
        torch.tensor([[3, 4]]),
        torch.tensor([[True, True]]),
        torch.tensor([[5, 6]]),
        torch.tensor([[True, True]]),
    )
    monkeypatch.setattr(inference, "process_data", lambda *args: tensors)

    stream = inference.VideoStream("model")
    decision = stream.react("video.mp4", "prompt")

    assert decision.should_respond is False
    assert decision.response == ""
    assert decision.raw_response == "<keep_silent>leaked text<|im_end|>"


def test_cli_runs_offline_inference(monkeypatch: Any, capsys: Any) -> None:
    calls: List[tuple[str, str, str]] = []

    class FakeStream:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

        def run(self, video_path: str, prompt: str) -> str:
            calls.append((self.model_path, video_path, prompt))
            return "result"

    monkeypatch.setattr(inference, "VideoStream", FakeStream)
    inference.main(
        [
            "--model_path",
            "model",
            "--video_path",
            "video.mp4",
            "--prompt",
            "question",
        ]
    )

    assert calls == [("model", "video.mp4", "question")]
    assert capsys.readouterr().out == "result\n"

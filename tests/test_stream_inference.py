import io
import json
from types import SimpleNamespace

import pytest

import stream_inference
from dispider.streaming import (
    DecisionKVCacheCapability,
    DecisionKVCacheUnavailable,
    StreamingEvent,
)


def args(**overrides):
    values = {
        "video": "sample.mp4",
        "model": "checkpoint",
        "prompt": "Tell me when the event occurs.",
        "chunk_frames": 4,
        "window_seconds": 16.0,
        "frames_per_window": 16,
        "max_history_windows": 8,
        "device": "cuda",
        "device_map": "auto",
        "max_new_tokens": 32,
        "decision_kv_cache": "auto",
        "verify_cache": False,
        "cache_oracle_atol": 0.125,
        "cache_fallback_margin": 0.125,
        "output": "-",
        "decision_trace": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeRuntime:
    def __init__(self, supported=False):
        self.cache_capability = DecisionKVCacheCapability(
            supported=supported,
            stable_prefix_safe=True,
            pending_suffix_committable=supported,
            reason="test backend",
        )


class FakeAdapter:
    instances = []

    def __init__(self, runtime, prompt, **kwargs):
        self.runtime = runtime
        self.prompt = prompt
        self.kwargs = kwargs
        self.decisions = (
            SimpleNamespace(
                clip_position=1,
                timestamp_s=15.0,
                score=0.5,
                triggered=True,
                kv_cache_used=kwargs["enable_decision_kv_cache"],
                cached_score=0.5,
                oracle_score=0.5,
                used_oracle_for_decision=False,
                stable_prefix_trigger_positions_before=(),
                pending_suffix_clip_positions=(1,),
                stable_prefix_trigger_positions_after=(1,),
            ),
        )
        self.instances.append(self)

    def new_session(self, **kwargs):
        return SimpleNamespace(settings=kwargs)


def test_run_emits_timestamped_json_and_reports_auto_fallback():
    loaded = []
    streamed = []

    def load(model, **kwargs):
        loaded.append((model, kwargs))
        return FakeRuntime(supported=False)

    def streamer(session, video, **kwargs):
        streamed.append((session, video, kwargs))
        return iter([StreamingEvent(15.125, "the event happened")])

    output = io.StringIO()
    diagnostics = io.StringIO()
    answer_count = stream_inference.run(
        args(),
        backend_loader=load,
        adapter_type=FakeAdapter,
        streamer=streamer,
        stdout=output,
        stderr=diagnostics,
    )

    assert answer_count == 1
    assert json.loads(output.getvalue()) == {
        "timestamp_s": 15.125,
        "timestamp": "00:00:15.125",
        "answer": "the event happened",
    }
    assert loaded == [
        (
            "checkpoint",
            {
                "device": "cuda",
                "device_map": "auto",
                "generation_kwargs": {"max_new_tokens": 32},
                "cache_oracle_atol": 0.125,
                "cache_fallback_margin": 0.125,
            },
        )
    ]
    assert streamed[0][1:] == ("sample.mp4", {"chunk_frames": 4})
    assert not FakeAdapter.instances[-1].kwargs["enable_decision_kv_cache"]
    assert "Decision KV cache: disabled" in diagnostics.getvalue()
    assert "emitted 1 answers" in diagnostics.getvalue()


def test_cache_on_is_strict_and_auto_enables_supported_backend():
    def unsupported(*args, **kwargs):
        return FakeRuntime(supported=False)

    strict_args = args(decision_kv_cache="on")
    with pytest.raises(DecisionKVCacheUnavailable, match="test backend"):
        stream_inference.run(strict_args, backend_loader=unsupported)

    output = io.StringIO()
    stream_inference.run(
        args(),
        backend_loader=lambda *args, **kwargs: FakeRuntime(supported=True),
        adapter_type=FakeAdapter,
        streamer=lambda *args, **kwargs: iter(()),
        stdout=output,
        stderr=io.StringIO(),
    )
    assert FakeAdapter.instances[-1].kwargs["enable_decision_kv_cache"]


def test_trace_file_and_timestamp_rounding(tmp_path):
    trace_path = tmp_path / "nested" / "trace.jsonl"
    output_path = tmp_path / "answers" / "answers.jsonl"
    stream_inference.run(
        args(decision_trace=str(trace_path), output=str(output_path)),
        backend_loader=lambda *args, **kwargs: FakeRuntime(),
        adapter_type=FakeAdapter,
        streamer=lambda *args, **kwargs: iter(()),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    record = json.loads(trace_path.read_text())
    assert output_path.read_text() == ""
    assert record["timestamp"] == "00:00:15.000"
    assert record["pending_clip_positions"] == [1]
    assert stream_inference.format_timestamp(59.9996) == "00:01:00.000"
    with pytest.raises(ValueError, match="negative"):
        stream_inference.format_timestamp(-0.1)


def test_verify_cache_requires_an_available_cache():
    with pytest.raises(ValueError, match="requires an available"):
        stream_inference.run(
            args(verify_cache=True),
            backend_loader=lambda *args, **kwargs: FakeRuntime(False),
        )


def test_released_cli_requires_checkpoint_aligned_frame_count():
    with pytest.raises(ValueError, match="requires --frames-per-window=16"):
        stream_inference.run(
            args(frames_per_window=8),
            backend_loader=lambda *args, **kwargs: FakeRuntime(False),
        )

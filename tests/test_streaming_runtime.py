import pytest

from dispider.streaming import (
    FrameChunk,
    StreamingSession,
    drive_frame_source,
    iter_video_frame_chunks,
)


class RecordingPerception:
    def __init__(self):
        self.windows = []

    def perceive(self, window):
        self.windows.append(window)
        return {"window": len(self.windows), "last_frame": window.frames[-1]}


class AlternatingDecision:
    def __init__(self, answers=True):
        self.answers = answers
        self.calls = []

    def should_respond(self, perception, *, window, history):
        self.calls.append((perception, window, history))
        if isinstance(self.answers, bool):
            return self.answers
        return self.answers[len(self.calls) - 1]


class RecordingReaction:
    def __init__(self):
        self.calls = []

    def respond(self, perception, *, window, history):
        self.calls.append((perception, window, history))
        return "  answer-%d  " % perception["window"]


def make_session(**kwargs):
    perception = RecordingPerception()
    decision = AlternatingDecision(kwargs.pop("answers", True))
    reaction = RecordingReaction()
    session = StreamingSession(
        perception=perception,
        decision=decision,
        reaction=reaction,
        **kwargs,
    )
    return session, perception, decision, reaction


def test_push_frames_uses_fixed_checkpoint_windows_and_samples():
    session, perception, _, reaction = make_session()

    assert session.push_frames(range(8), range(8)) == []
    events = session.push_frames(range(8, 17), range(8, 17))

    assert [event.as_dict() for event in events] == [
        {"timestamp_s": 15.0, "answer": "answer-1"}
    ]
    window = perception.windows[0]
    assert (window.start_s, window.end_s, window.complete) == (0.0, 16.0, True)
    assert len(window.frames) == 16
    assert window.timestamps_s == tuple(float(value) for value in range(16))
    assert len(reaction.calls) == 1


def test_flush_is_idempotent_and_reset_starts_a_fresh_stream():
    session, perception, _, _ = make_session()
    session.push_frames(["a", "b", "c"], [0.0, 1.0, 2.0])

    events = session.flush()

    assert len(events) == 1
    assert events[0].timestamp_s == 2.0
    assert len(perception.windows[0].frames) == 16
    assert not perception.windows[0].complete
    assert session.flush() == []
    with pytest.raises(RuntimeError, match="after flush"):
        session.push_frames(["d"], [3.0])

    session.reset()
    assert not session.closed
    assert session.history == ()
    session.push_frames(["new"], [0.0])
    assert session.flush()[0].timestamp_s == 0.0


def test_timestamp_errors_are_rejected_before_state_changes():
    session, _, _, _ = make_session()

    with pytest.raises(ValueError, match="same length"):
        session.push_frames([1, 2], [0.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        session.push_frames([1, 2], [1.0, 1.0])
    assert session.history == ()

    session.push_frames([1], [1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        session.push_frames([2, 3], [2.0, 1.5])
    assert session.flush()[0].timestamp_s == 1.0


def test_decision_gates_reaction_and_history_is_bounded():
    session, _, decision, reaction = make_session(
        answers=[False, True, True, True], max_history_windows=2
    )

    events = session.push_frames(range(49), range(49))
    events.extend(session.flush())

    assert [event.timestamp_s for event in events] == [31.0, 47.0, 48.0]
    assert len(reaction.calls) == 3
    assert len(decision.calls) == 4
    assert [len(call[2]) for call in decision.calls] == [0, 1, 2, 2]
    assert len(session.history) == 2


class FakeBatch:
    def __init__(self, values):
        self.values = values

    def asnumpy(self):
        return self.values


class FakeVideoReader:
    def __init__(self):
        self.values = ["f0", "f1", "f2", "f3", "f4"]

    def __len__(self):
        return len(self.values)

    def get_avg_fps(self):
        return 2.0

    def get_batch(self, indices):
        return FakeBatch([self.values[index] for index in indices])

    def get_frame_timestamp(self, indices):
        return [(index / 2.0, (index + 1) / 2.0) for index in indices]


def test_file_source_is_chunked_and_can_drive_session():
    seen_paths = []

    def reader_factory(path):
        seen_paths.append(path)
        return FakeVideoReader()

    chunks = list(
        iter_video_frame_chunks(
            "sample.mp4", chunk_frames=2, reader_factory=reader_factory
        )
    )

    assert seen_paths == ["sample.mp4"]
    assert [chunk.frames for chunk in chunks] == [
        ("f0", "f1"),
        ("f2", "f3"),
        ("f4",),
    ]
    assert chunks[-1].timestamps_s == (2.0,)

    list(
        iter_video_frame_chunks(
            "https://example.test/video.mp4",
            chunk_frames=2,
            reader_factory=reader_factory,
        )
    )
    assert seen_paths[-1] == "https://example.test/video.mp4"

    session, _, _, _ = make_session()
    events = list(drive_frame_source(session, chunks))
    assert [event.timestamp_s for event in events] == [2.0]


def test_drive_source_can_leave_session_open():
    session, _, _, _ = make_session()
    chunks = [FrameChunk(frames=("frame",), timestamps_s=(0.0,))]

    assert list(drive_frame_source(session, chunks, flush=False)) == []
    assert not session.closed


class CombinedAdapter:
    def __init__(self):
        self.reset_calls = 0

    def perceive(self, window):
        return window

    def should_respond(self, perception, *, window, history):
        return False

    def respond(self, perception, *, window, history):
        raise AssertionError("the decision does not trigger")

    def reset(self):
        self.reset_calls += 1


def test_reset_invokes_a_shared_adapters_hook_once():
    adapter = CombinedAdapter()
    session = StreamingSession(adapter, adapter, adapter)
    session.push_frames(["frame"], [0.0])

    session.reset()

    assert adapter.reset_calls == 1
    assert session.history == ()

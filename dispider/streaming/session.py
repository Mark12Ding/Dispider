import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, List, Protocol, Sequence, Tuple

from .types import SampledWindow, StreamingEvent, WindowRecord

_Events = List[StreamingEvent]


class Perception(Protocol):
    def perceive(self, window: SampledWindow) -> Any: ...


class Decision(Protocol):
    def should_respond(
        self,
        perception: Any,
        *,
        window: SampledWindow,
        history: Sequence[WindowRecord],
    ) -> bool: ...


class Reaction(Protocol):
    def respond(
        self,
        perception: Any,
        *,
        window: SampledWindow,
        history: Sequence[WindowRecord],
    ) -> str: ...


@dataclass
class _Candidate:
    frame: Any
    timestamp_s: float
    distance_s: float


class StreamingSession:
    """Serial, bounded-state scheduler for online Dispider inference.

    The session divides the video timeline into non-overlapping 16-second
    windows and retains only the 16 frames nearest to uniformly spaced sample
    points. Model-owning perception, decision, and reaction adapters are
    injected rather than registered on this object.
    """

    def __init__(
        self,
        perception: Perception,
        decision: Decision,
        reaction: Reaction,
        *,
        window_seconds: float = 16.0,
        frames_per_window: int = 16,
        max_history_windows: int = 8,
        timeline_origin_s: float = 0.0,
    ) -> None:
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        if frames_per_window <= 0:
            raise ValueError("frames_per_window must be positive")
        if max_history_windows < 0:
            raise ValueError("max_history_windows cannot be negative")
        if not math.isfinite(timeline_origin_s):
            raise ValueError("timeline_origin_s must be finite")

        self.perception = perception
        self.decision = decision
        self.reaction = reaction
        self.window_seconds = float(window_seconds)
        self.frames_per_window = int(frames_per_window)
        self.max_history_windows = int(max_history_windows)
        self.timeline_origin_s = float(timeline_origin_s)

        self._lock = threading.RLock()
        self._history = deque(maxlen=max_history_windows)
        self._last_input_timestamp_s = None
        self._last_event_timestamp_s = None
        self._window_start_s = None
        self._candidates: List[_Candidate] = []
        self._closed = False

    @property
    def history(self) -> Tuple[WindowRecord, ...]:
        with self._lock:
            return tuple(self._history)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def push_frames(
        self, frames: Iterable[Any], timestamps: Iterable[float]
    ) -> List[StreamingEvent]:
        """Consume a finite frame chunk and return answers produced by it.

        Timestamps must be finite and strictly increasing across the whole
        session. The chunk is validated before any session state is changed.
        """

        frame_chunk = tuple(frames)
        coerce = self._coerce_timestamp
        timestamp_chunk = tuple(coerce(value) for value in timestamps)
        if len(frame_chunk) != len(timestamp_chunk):
            raise ValueError("frames and timestamps must have the same length")

        with self._lock:
            if self._closed:
                message = "cannot push frames after flush"
                raise RuntimeError(f"{message}; call reset first")
            self._validate_timestamp_chunk(timestamp_chunk)

            events = []
            for frame, timestamp_s in zip(frame_chunk, timestamp_chunk):
                events.extend(self._push_frame(frame, timestamp_s))
            return events

    def flush(self) -> List[StreamingEvent]:
        """Finish a partial window and close the stream.

        Repeated calls are idempotent. Use :meth:`reset` before starting a new
        stream with the same adapters.
        """

        with self._lock:
            if self._closed:
                return []
            if not self._candidates:
                self._closed = True
                return []
            events = self._finish_window(complete=False)
            self._closed = True
            return events

    def reset(self) -> None:
        """Clear video state and invoke each adapter's optional reset hook."""

        with self._lock:
            reset_components = set()
            for component in (self.perception, self.decision, self.reaction):
                if id(component) in reset_components:
                    continue
                reset_components.add(id(component))
                reset = getattr(component, "reset", None)
                if callable(reset):
                    reset()
            self._history.clear()
            self._last_input_timestamp_s = None
            self._last_event_timestamp_s = None
            self._window_start_s = None
            self._candidates = []
            self._closed = False

    @staticmethod
    def _coerce_timestamp(value: float) -> float:
        timestamp_s = float(value)
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamps must be finite")
        return timestamp_s

    def _validate_timestamp_chunk(self, timestamps: Sequence[float]) -> None:
        previous = self._last_input_timestamp_s
        for timestamp_s in timestamps:
            if timestamp_s < self.timeline_origin_s:
                raise ValueError("timestamps cannot precede timeline_origin_s")
            if previous is not None and timestamp_s <= previous:
                raise ValueError("timestamps must be strictly increasing")
            previous = timestamp_s

    def _push_frame(self, frame: Any, timestamp_s: float) -> _Events:
        events = []
        if self._window_start_s is None:
            self._window_start_s = self._window_start_for(timestamp_s)
            self._candidates = self._empty_candidates()

        window_end_s = self._window_start_s + self.window_seconds
        if timestamp_s >= window_end_s:
            if self._has_samples():
                events.extend(self._finish_window(complete=True))
            self._window_start_s = self._window_start_for(timestamp_s)
            self._candidates = self._empty_candidates()

        self._consider_frame(frame, timestamp_s)
        self._last_input_timestamp_s = timestamp_s
        return events

    def _window_start_for(self, timestamp_s: float) -> float:
        offset = timestamp_s - self.timeline_origin_s
        window_index = math.floor(offset / self.window_seconds)
        return self.timeline_origin_s + window_index * self.window_seconds

    def _empty_candidates(self) -> List[_Candidate]:
        return [None] * self.frames_per_window

    def _sample_target(self, sample_index: int) -> float:
        stride_s = self.window_seconds / self.frames_per_window
        return self._window_start_s + (sample_index + 0.5) * stride_s

    def _consider_frame(self, frame: Any, timestamp_s: float) -> None:
        for sample_index, candidate in enumerate(self._candidates):
            distance_s = abs(timestamp_s - self._sample_target(sample_index))
            if candidate is None or distance_s < candidate.distance_s:
                self._candidates[sample_index] = _Candidate(
                    frame=frame,
                    timestamp_s=timestamp_s,
                    distance_s=distance_s,
                )

    def _has_samples(self) -> bool:
        return bool(self._candidates) and self._candidates[0] is not None

    def _finish_window(self, *, complete: bool) -> List[StreamingEvent]:
        candidates = tuple(self._candidates)
        if not candidates or candidates[0] is None:
            return []

        sampled_timestamps = tuple(item.timestamp_s for item in candidates)
        window = SampledWindow(
            start_s=self._window_start_s,
            end_s=self._window_start_s + self.window_seconds,
            frames=tuple(candidate.frame for candidate in candidates),
            timestamps_s=sampled_timestamps,
            complete=complete,
        )
        history = tuple(self._history)
        perception = self.perception.perceive(window)
        answer = None
        events = []
        should_respond = self.decision.should_respond(
            perception, window=window, history=history
        )
        if should_respond:
            raw_answer = self.reaction.respond(
                perception, window=window, history=history
            )
            if not isinstance(raw_answer, str):
                raise TypeError("reaction.respond must return a string")
            answer = raw_answer.strip()
            if answer:
                event = StreamingEvent(window.timestamp_s, answer)
                if (
                    self._last_event_timestamp_s is not None
                    and event.timestamp_s <= self._last_event_timestamp_s
                ):
                    message = "streaming event timestamps"
                    raise RuntimeError(f"{message} must increase")
                events.append(event)
                self._last_event_timestamp_s = event.timestamp_s

        self._history.append(
            WindowRecord(
                start_s=window.start_s,
                end_s=window.end_s,
                timestamp_s=window.timestamp_s,
                perception=perception,
                answer=answer,
            )
        )
        self._candidates = self._empty_candidates()
        return events

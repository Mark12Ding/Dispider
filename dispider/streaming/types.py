from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class SampledWindow:
    """A fixed-size sample from one interval of the video timeline."""

    start_s: float
    end_s: float
    frames: Tuple[Any, ...]
    timestamps_s: Tuple[float, ...]
    complete: bool

    @property
    def timestamp_s(self) -> float:
        return self.timestamps_s[-1]


@dataclass(frozen=True)
class WindowRecord:
    """Bounded context retained after a window has been processed."""

    start_s: float
    end_s: float
    timestamp_s: float
    perception: Any
    answer: Optional[str]


@dataclass(frozen=True)
class StreamingEvent:
    """A timestamped answer emitted by a streaming session."""

    timestamp_s: float
    answer: str

    def as_dict(self):
        return {"timestamp_s": self.timestamp_s, "answer": self.answer}

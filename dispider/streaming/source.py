from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

from .session import StreamingSession
from .types import StreamingEvent


@dataclass(frozen=True)
class FrameChunk:
    frames: Tuple[Any, ...]
    timestamps_s: Tuple[float, ...]


def iter_video_frame_chunks(
    path,
    *,
    chunk_frames: int = 64,
    reader_factory: Optional[Callable[[str], Any]] = None,
) -> Iterator[FrameChunk]:
    """Read a video lazily in chunks suitable for simulating a live source."""

    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    factory = reader_factory
    if factory is None:
        from decord import VideoReader

        def factory(video_path):
            return VideoReader(video_path, num_threads=1)

    reader = factory(str(path))
    fps = float(reader.get_avg_fps())
    if fps <= 0:
        raise ValueError("video reader returned a non-positive frame rate")

    for start in range(0, len(reader), chunk_frames):
        indices = list(range(start, min(start + chunk_frames, len(reader))))
        batch = reader.get_batch(indices)
        if hasattr(batch, "asnumpy"):
            batch = batch.asnumpy()
        frames = tuple(batch)
        timestamps_s = _read_timestamps(reader, indices, fps)
        yield FrameChunk(frames=frames, timestamps_s=timestamps_s)


def drive_frame_source(
    session: StreamingSession,
    chunks: Iterable[FrameChunk],
    *,
    flush: bool = True,
) -> Iterator[StreamingEvent]:
    """Feed chunks serially into a session and yield events as they occur."""

    for chunk in chunks:
        yield from session.push_frames(chunk.frames, chunk.timestamps_s)
    if flush:
        yield from session.flush()


def stream_video_file(
    session: StreamingSession,
    path,
    *,
    chunk_frames: int = 64,
    reader_factory: Optional[Callable[[str], Any]] = None,
) -> Iterator[StreamingEvent]:
    """Convenience wrapper for deterministic file-backed live simulation."""

    chunks = iter_video_frame_chunks(
        path, chunk_frames=chunk_frames, reader_factory=reader_factory
    )
    yield from drive_frame_source(session, chunks)


def _read_timestamps(reader, indices, fps: float) -> Tuple[float, ...]:
    get_timestamps = getattr(reader, "get_frame_timestamp", None)
    if get_timestamps is None:
        return tuple(index / fps for index in indices)

    raw_timestamps = get_timestamps(indices)
    result = []
    for raw_timestamp in raw_timestamps:
        try:
            timestamp_s = float(raw_timestamp[0])
        except (IndexError, TypeError):
            timestamp_s = float(raw_timestamp)
        result.append(timestamp_s)
    return tuple(result)

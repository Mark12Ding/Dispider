#!/usr/bin/env python3
"""Run timestamped Dispider inference over a file-backed video stream."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO

from dispider.streaming import (
    DecisionKVCacheUnavailable,
    DispiderStreamingAdapter,
    ReleasedPerceptionDecisionReactionBackend,
    StreamingEvent,
    stream_video_file,
)

_BACKEND_TYPE = ReleasedPerceptionDecisionReactionBackend
_DEFAULT_BACKEND_LOADER = _BACKEND_TYPE.from_pretrained


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a video incrementally and emit one JSON object for every "
            "timestamped Dispider response."
        )
    )
    parser.add_argument("video", help="video file or URI understood by Decord")
    parser.add_argument(
        "--model",
        required=True,
        help="local checkpoint directory or Hugging Face repository ID",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="instruction monitored during the stream",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=64,
        help="number of source frames decoded at a time (default: 64)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=16.0,
        help="Decision interval in seconds (default: 16)",
    )
    parser.add_argument(
        "--frames-per-window",
        type=int,
        default=16,
        help="uniform samples per Decision interval; released backend requires 16",
    )
    parser.add_argument(
        "--max-history-windows",
        type=int,
        default=8,
        help="bounded scheduler history (default: 8)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device map; use 'none' to disable it",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Reaction token limit"
    )
    parser.add_argument(
        "--decision-kv-cache",
        choices=("auto", "on", "off"),
        default="auto",
        help="Decision-model KV cache policy (default: auto)",
    )
    parser.add_argument(
        "--verify-cache",
        action="store_true",
        help="compare cached Decision scores with the exact no-cache path",
    )
    parser.add_argument(
        "--cache-oracle-atol",
        type=float,
        default=0.125,
        help="absolute score tolerance for --verify-cache (default: 0.125)",
    )
    parser.add_argument(
        "--cache-fallback-margin",
        type=float,
        default=0.125,
        help="use the no-cache score this close to threshold (default: 0.125)",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="answer JSONL path, or '-' for stdout (default: '-')",
    )
    parser.add_argument(
        "--decision-trace",
        help="optional path for per-window Decision scores and cache metadata",
    )
    return parser


def format_timestamp(timestamp_s: float) -> str:
    """Format a non-negative timestamp without losing millisecond carries."""

    if timestamp_s < 0:
        raise ValueError("timestamp cannot be negative")
    total_ms = int(round(timestamp_s * 1000.0))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def run(
    args: argparse.Namespace,
    *,
    backend_loader: Any = _DEFAULT_BACKEND_LOADER,
    adapter_type: Any = DispiderStreamingAdapter,
    streamer: Any = stream_video_file,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """Load the model, consume one stream, and return the number of answers."""

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")
    if args.frames_per_window != 16:
        raise ValueError("released Dispider requires --frames-per-window=16")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not math.isfinite(args.cache_oracle_atol) or args.cache_oracle_atol < 0:
        raise ValueError("--cache-oracle-atol must be finite and nonnegative")
    if not math.isfinite(args.cache_fallback_margin) or args.cache_fallback_margin < 0:
        raise ValueError("--cache-fallback-margin must be finite and nonnegative")

    device_map = None if args.device_map.lower() == "none" else args.device_map
    runtime = backend_loader(
        args.model,
        device=args.device,
        device_map=device_map,
        generation_kwargs={"max_new_tokens": args.max_new_tokens},
        cache_oracle_atol=args.cache_oracle_atol,
        cache_fallback_margin=args.cache_fallback_margin,
    )
    cache_supported = runtime.cache_capability.supported
    if args.decision_kv_cache == "on" and not cache_supported:
        raise DecisionKVCacheUnavailable(runtime.cache_capability.reason)
    cache_enabled = args.decision_kv_cache == "on" or (
        args.decision_kv_cache == "auto" and cache_supported
    )
    if args.verify_cache and not cache_enabled:
        message = "--verify-cache requires an available Decision KV cache"
        raise ValueError(message)

    cache_status = "enabled" if cache_enabled else "disabled"
    cache_reason = runtime.cache_capability.reason
    print(
        "Decision KV cache: %s (%s)" % (cache_status, cache_reason),
        file=stderr,
    )
    adapter = adapter_type(
        runtime,
        args.prompt,
        enable_decision_kv_cache=cache_enabled,
        verify_cache_with_oracle=args.verify_cache,
    )
    session = adapter.new_session(
        window_seconds=args.window_seconds,
        frames_per_window=args.frames_per_window,
        max_history_windows=args.max_history_windows,
    )

    if args.output == "-":
        output_context = nullcontext(stdout)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_context = output_path.open("w", encoding="utf-8")
    answer_count = 0
    with output_context as output:
        events: Iterable[StreamingEvent] = streamer(
            session,
            args.video,
            chunk_frames=args.chunk_frames,
        )
        for event in events:
            record = {
                "timestamp_s": event.timestamp_s,
                "timestamp": format_timestamp(event.timestamp_s),
                "answer": event.answer,
            }
            serialized = json.dumps(record, ensure_ascii=False)
            print(serialized, file=output, flush=True)
            answer_count += 1

    if args.decision_trace:
        _write_decision_trace(Path(args.decision_trace), adapter.decisions)
    used_count = sum(trace.kv_cache_used for trace in adapter.decisions)
    print(
        "Processed %d Decision windows; emitted %d answers; "
        "KV cache used for %d windows."
        % (len(adapter.decisions), answer_count, used_count),
        file=stderr,
    )
    return answer_count


def _write_decision_trace(path: Path, decisions: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for trace in decisions:
            record = {
                "clip_position": trace.clip_position,
                "timestamp_s": trace.timestamp_s,
                "timestamp": format_timestamp(trace.timestamp_s),
                "score": trace.score,
                "triggered": trace.triggered,
                "kv_cache_used": trace.kv_cache_used,
                "cached_score": trace.cached_score,
                "oracle_score": trace.oracle_score,
                "used_oracle_for_decision": trace.used_oracle_for_decision,
                "stable_trigger_positions_before": (
                    trace.stable_prefix_trigger_positions_before
                ),
                "pending_clip_positions": trace.pending_suffix_clip_positions,
                "stable_trigger_positions_after": (
                    trace.stable_prefix_trigger_positions_after
                ),
            }
            print(json.dumps(record), file=output)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

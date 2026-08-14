#!/usr/bin/env python3
"""Run and strictly score the current OVO-Bench edition with Dispider."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


CATEGORY_TASKS: dict[str, tuple[str, ...]] = {
    "backward": ("EPM", "ASI", "HLD"),
    "realtime": ("STU", "OJR", "ATR", "ACR", "OCR", "FPD"),
    "forward": ("REC", "SSR", "CRR"),
}
TASK_CATEGORY = {
    task: category for category, tasks in CATEGORY_TASKS.items() for task in tasks
}
TASK_ORDER = tuple(task for tasks in CATEGORY_TASKS.values() for task in tasks)

# OVO-Bench c34093f, data/ovo_bench_new.json.
CURRENT_EDITION_ROWS = {
    "EPM": 297,
    "ASI": 148,
    "HLD": 186,
    "STU": 178,
    "OJR": 184,
    "ATR": 116,
    "ACR": 109,
    "OCR": 149,
    "FPD": 101,
    "REC": 82,
    "SSR": 42,
    "CRR": 48,
}
CURRENT_EDITION_CALLS = {
    **{task: CURRENT_EDITION_ROWS[task] for task in TASK_ORDER[:9]},
    "REC": 698,
    "SSR": 629,
    "CRR": 240,
}

RESULT_NAME = f"Dispider_{'_'.join(TASK_ORDER)}_offline_1.json"
MANIFEST_NAME = "run_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
SHARD_ALGORITHM = "largest-call-count-first-v1"


class EvaluationError(RuntimeError):
    """Raised when inputs, worker output, or resume state are unsafe."""


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read JSON {path}: {error}") from error


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise EvaluationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise EvaluationError(f"{label} is not a file: {path}")
    return resolved


def _require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise EvaluationError(f"{label} is not a directory: {path}")
    return resolved


def _valid_id(value: Any) -> bool:
    return isinstance(value, (int, str)) and not isinstance(value, bool)


def _annotation_manifest(
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = Counter(dict.fromkeys(TASK_ORDER, 0))
    calls = Counter(dict.fromkeys(TASK_ORDER, 0))
    seen_ids: set[int | str] = set()

    for position, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise EvaluationError(f"annotation {position} is not an object")
        annotation_id = annotation.get("id")
        task = annotation.get("task")
        if not _valid_id(annotation_id):
            raise EvaluationError(
                f"annotation {position} has invalid id {annotation_id!r}"
            )
        if annotation_id in seen_ids:
            raise EvaluationError(f"duplicate annotation id {annotation_id!r}")
        seen_ids.add(annotation_id)
        if task not in TASK_CATEGORY:
            raise EvaluationError(
                f"annotation id {annotation_id!r} has unknown task {task!r}"
            )

        rows[task] += 1
        if task in CATEGORY_TASKS["forward"]:
            test_info = annotation.get("test_info")
            if not isinstance(test_info, list) or not test_info:
                raise EvaluationError(
                    f"forward annotation id {annotation_id!r} needs test_info"
                )
            calls[task] += len(test_info)
        else:
            calls[task] += 1

    return {
        "rows_by_task": {task: rows[task] for task in TASK_ORDER},
        "calls_by_task": {task: calls[task] for task in TASK_ORDER},
        "total_rows": sum(rows.values()),
        "total_calls": sum(calls.values()),
    }


def _validate_current_edition(manifest: Mapping[str, Any]) -> None:
    errors = []
    if manifest["rows_by_task"] != CURRENT_EDITION_ROWS:
        errors.append(
            f"row counts are {manifest['rows_by_task']!r}, "
            f"expected {CURRENT_EDITION_ROWS!r}"
        )
    if manifest["calls_by_task"] != CURRENT_EDITION_CALLS:
        errors.append(
            f"call counts are {manifest['calls_by_task']!r}, "
            f"expected {CURRENT_EDITION_CALLS!r}"
        )
    if errors:
        raise EvaluationError(
            "annotations do not match the pinned OVO-Bench current edition: "
            + "; ".join(errors)
        )


def _call_count(annotation: Mapping[str, Any]) -> int:
    if annotation["task"] in CATEGORY_TASKS["forward"]:
        return len(annotation["test_info"])
    return 1


def _make_shards(
    annotations: Sequence[Mapping[str, Any]], num_shards: int
) -> list[list[dict[str, Any]]]:
    if num_shards < 1:
        raise EvaluationError("at least one GPU is required")
    if num_shards > len(annotations):
        raise EvaluationError(
            f"cannot create {num_shards} non-empty shards from "
            f"{len(annotations)} annotations"
        )

    assignments: list[list[tuple[int, dict[str, Any]]]] = [
        [] for _ in range(num_shards)
    ]
    shard_calls = [0] * num_shards
    indexed = [
        (index, copy.deepcopy(dict(row))) for index, row in enumerate(annotations)
    ]
    indexed.sort(key=lambda item: (-_call_count(item[1]), item[0]))
    for index, annotation in indexed:
        shard_id = min(range(num_shards), key=lambda item: (shard_calls[item], item))
        assignments[shard_id].append((index, annotation))
        shard_calls[shard_id] += _call_count(annotation)

    return [[annotation for _, annotation in sorted(shard)] for shard in assignments]


def _require_keys(
    value: Mapping[str, Any], required: Iterable[str], *, context: str
) -> None:
    missing = [key for key in required if key not in value]
    if missing:
        raise EvaluationError(f"{context} is missing keys: {', '.join(missing)}")


def _video_matches(observed: Any, expected: Any) -> bool:
    if not isinstance(observed, str) or not isinstance(expected, str):
        return False
    normalized_observed = observed.replace("\\", "/")
    normalized_expected = expected.replace("\\", "/")
    return normalized_observed == normalized_expected or normalized_observed.endswith(
        f"/{normalized_expected.lstrip('/')}"
    )


def _validate_response(response: Any, *, context: str) -> None:
    if response is not None and not isinstance(response, str):
        raise EvaluationError(
            f"{context} response must be a string or null, "
            f"got {type(response).__name__}"
        )


def _validate_result(
    result: Mapping[str, Any],
    annotation: Mapping[str, Any],
    bucket: str,
    source: Path,
) -> int:
    annotation_id = annotation["id"]
    task = annotation["task"]
    context = f"result id {annotation_id!r} in {source}"
    if TASK_CATEGORY[task] != bucket:
        raise EvaluationError(
            f"{context} is in {bucket!r}, expected {TASK_CATEGORY[task]!r}"
        )
    if result.get("task") != task:
        raise EvaluationError(
            f"{context} has task {result.get('task')!r}, expected {task!r}"
        )
    if not _video_matches(result.get("video"), annotation.get("video")):
        raise EvaluationError(f"{context} video differs from the annotation")

    if bucket != "forward":
        expected_fields = {
            "id",
            "video",
            "task",
            "question",
            "response",
            "ground_truth",
        }
        if set(result) != expected_fields:
            raise EvaluationError(
                f"{context} fields are {sorted(result)}, "
                f"expected {sorted(expected_fields)}"
            )
        if result["question"] != annotation.get("question"):
            raise EvaluationError(f"{context} question differs from the annotation")
        expected_ground_truth = chr(65 + annotation["gt"])
        if result["ground_truth"] != expected_ground_truth:
            raise EvaluationError(
                f"{context} ground_truth is {result['ground_truth']!r}, "
                f"expected {expected_ground_truth!r}"
            )
        _validate_response(result["response"], context=context)
        return int(result["response"] is None)

    if set(result) != set(annotation):
        raise EvaluationError(
            f"{context} fields are {sorted(result)}, " f"expected {sorted(annotation)}"
        )
    for key, expected_value in annotation.items():
        if key in {"video", "test_info"}:
            continue
        if result[key] != expected_value:
            raise EvaluationError(f"{context} field {key!r} differs")

    result_test_info = result.get("test_info")
    expected_test_info = annotation["test_info"]
    if not isinstance(result_test_info, list):
        raise EvaluationError(f"{context} test_info is not a list")
    if len(result_test_info) != len(expected_test_info):
        raise EvaluationError(
            f"{context} has {len(result_test_info)} calls, "
            f"expected {len(expected_test_info)}"
        )

    null_count = 0
    for index, (observed, expected) in enumerate(
        zip(result_test_info, expected_test_info)
    ):
        call_context = f"{context}, test_info[{index}]"
        if not isinstance(observed, Mapping):
            raise EvaluationError(f"{call_context} is not an object")
        if set(observed) != set(expected) | {"response"}:
            raise EvaluationError(f"{call_context} has unexpected or missing fields")
        for key, expected_value in expected.items():
            if observed[key] != expected_value:
                raise EvaluationError(f"{call_context} field {key!r} differs")
        _validate_response(observed["response"], context=call_context)
        null_count += int(observed["response"] is None)
    return null_count


def _merge_results(
    paths: Sequence[Path], annotations: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    if not paths:
        raise EvaluationError("no worker result files were provided")
    annotation_by_id = {annotation["id"]: annotation for annotation in annotations}
    observed: dict[int | str, tuple[dict[str, Any], str, Path]] = {}
    null_count = 0

    for path in sorted(paths):
        payload = _load_json(path)
        if not isinstance(payload, Mapping):
            raise EvaluationError(f"worker output {path} is not an object")
        if set(payload) != set(CATEGORY_TASKS):
            raise EvaluationError(
                f"worker output {path} must contain exactly "
                f"{', '.join(CATEGORY_TASKS)}"
            )
        for bucket in CATEGORY_TASKS:
            results = payload[bucket]
            if not isinstance(results, list):
                raise EvaluationError(f"{path} field {bucket!r} is not a list")
            for index, raw_result in enumerate(results):
                context = f"{path}:{bucket}[{index}]"
                if not isinstance(raw_result, Mapping):
                    raise EvaluationError(f"{context} is not an object")
                annotation_id = raw_result.get("id")
                if not _valid_id(annotation_id):
                    raise EvaluationError(f"{context} has invalid id {annotation_id!r}")
                if annotation_id not in annotation_by_id:
                    raise EvaluationError(
                        f"{context} has unexpected annotation id {annotation_id!r}"
                    )
                if annotation_id in observed:
                    previous = observed[annotation_id][2]
                    raise EvaluationError(
                        "duplicate result id "
                        f"{annotation_id!r} in {previous} and {path}"
                    )
                result = copy.deepcopy(dict(raw_result))
                annotation = annotation_by_id[annotation_id]
                null_count += _validate_result(result, annotation, bucket, path)
                observed[annotation_id] = (result, bucket, path)

    missing = [row for row in annotations if row["id"] not in observed]
    if missing:
        by_task = Counter(row["task"] for row in missing)
        examples = [row["id"] for row in missing[:10]]
        raise EvaluationError(
            f"missing {len(missing)} result rows; by task={dict(by_task)}; "
            f"first ids={examples}"
        )

    merged = {bucket: [] for bucket in CATEGORY_TASKS}
    for annotation in annotations:
        result, bucket, _ = observed[annotation["id"]]
        merged[bucket].append(result)
    return merged, null_count


def _observed_manifest(
    merged: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = Counter(dict.fromkeys(TASK_ORDER, 0))
    calls = Counter(dict.fromkeys(TASK_ORDER, 0))
    for bucket in CATEGORY_TASKS:
        for result in merged[bucket]:
            task = result["task"]
            rows[task] += 1
            calls[task] += len(result["test_info"]) if bucket == "forward" else 1
    return {
        "rows_by_task": {task: rows[task] for task in TASK_ORDER},
        "calls_by_task": {task: calls[task] for task in TASK_ORDER},
        "total_rows": sum(rows.values()),
        "total_calls": sum(calls.values()),
    }


def _score_backward_realtime(response: str | None, ground_truth: str) -> int:
    return int(response is not None and ground_truth in response)


def _score_rec(response: str | None, count: Any) -> int:
    if response is None:
        return 0
    digits = "".join(re.findall(r"\d+", response))
    return int(digits == str(count))


def _score_binary(response: str | None, expected_type: Any) -> int:
    if response is None:
        return 0
    if (response == "N" and expected_type == 0) or (
        response == "Y" and expected_type == 1
    ):
        return 1
    ground_truth = "No" if expected_type == 0 else "Yes"
    return int(ground_truth in response)


def _score(
    merged: Mapping[str, Sequence[Mapping[str, Any]]],
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    annotation_by_id = {annotation["id"]: annotation for annotation in annotations}
    correct = Counter(dict.fromkeys(TASK_ORDER, 0))
    totals = Counter(dict.fromkeys(TASK_ORDER, 0))
    null_responses = Counter(dict.fromkeys(TASK_ORDER, 0))
    empty_responses = Counter(dict.fromkeys(TASK_ORDER, 0))

    for bucket in ("backward", "realtime"):
        for result in merged[bucket]:
            task = result["task"]
            response = result["response"]
            ground_truth = chr(65 + annotation_by_id[result["id"]]["gt"])
            correct[task] += _score_backward_realtime(response, ground_truth)
            totals[task] += 1
            null_responses[task] += int(response is None)
            empty_responses[task] += int(response == "")

    for result in merged["forward"]:
        task = result["task"]
        annotation = annotation_by_id[result["id"]]
        for index, observed in enumerate(result["test_info"]):
            expected = annotation["test_info"][index]
            response = observed["response"]
            if task == "REC":
                correct[task] += _score_rec(response, expected["count"])
            else:
                correct[task] += _score_binary(response, expected["type"])
            totals[task] += 1
            null_responses[task] += int(response is None)
            empty_responses[task] += int(response == "")

    tasks: dict[str, dict[str, Any]] = {}
    for task in TASK_ORDER:
        accuracy = 100.0 * correct[task] / totals[task]
        tasks[task] = {
            "category": TASK_CATEGORY[task],
            "correct": correct[task],
            "total": totals[task],
            "null_responses": null_responses[task],
            "empty_responses": empty_responses[task],
            "accuracy": accuracy,
        }

    categories: dict[str, dict[str, Any]] = {}
    for category, category_tasks in CATEGORY_TASKS.items():
        macro_accuracy = statistics.fmean(
            tasks[task]["accuracy"] for task in category_tasks
        )
        micro_correct = sum(correct[task] for task in category_tasks)
        micro_total = sum(totals[task] for task in category_tasks)
        categories[category] = {
            "tasks": list(category_tasks),
            "macro_accuracy": macro_accuracy,
            "micro_correct": micro_correct,
            "micro_total": micro_total,
            "micro_accuracy": 100.0 * micro_correct / micro_total,
        }

    official_score = statistics.fmean(
        categories[category]["macro_accuracy"] for category in CATEGORY_TASKS
    )
    total_correct = sum(correct.values())
    total_calls = sum(totals.values())
    return {
        "tasks": tasks,
        "categories": categories,
        "overall": {
            "official_three_category_macro_accuracy": official_score,
            "twelve_task_macro_accuracy": statistics.fmean(
                tasks[task]["accuracy"] for task in TASK_ORDER
            ),
            "micro_correct": total_correct,
            "micro_total": total_calls,
            "null_responses": sum(null_responses.values()),
            "empty_responses": sum(empty_responses.values()),
            "micro_accuracy": 100.0 * total_correct / total_calls,
        },
    }


def _aggregate_and_score(
    paths: Sequence[Path], annotations: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    expected_manifest = _annotation_manifest(annotations)
    _validate_current_edition(expected_manifest)
    merged, null_count = _merge_results(paths, annotations)
    observed_manifest = _observed_manifest(merged)
    if observed_manifest != expected_manifest:
        raise EvaluationError(
            f"observed manifest {observed_manifest!r} differs from "
            f"expected {expected_manifest!r}"
        )
    report = {
        "schema_version": 1,
        "manifest": {
            "expected": expected_manifest,
            "observed": observed_manifest,
        },
        **_score(merged, annotations),
    }
    if null_count != report["overall"]["null_responses"]:
        raise EvaluationError("internal null-response count mismatch")
    return merged, report


def _model_identity(model_path: Path) -> list[dict[str, Any]]:
    suffixes = {
        ".bin",
        ".json",
        ".model",
        ".pt",
        ".pth",
        ".safetensors",
        ".txt",
    }
    identity = []
    for path in sorted(model_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        stat = path.stat()
        identity.append(
            {
                "path": path.relative_to(model_path).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    if not identity:
        raise EvaluationError(f"no model files found under {model_path}")
    return identity


def _source_identity(root: Path) -> list[dict[str, str]]:
    sources = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        sources.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
            }
        )
    if not sources:
        raise EvaluationError(f"no Python sources found under {root}")
    return sources


def _chunk_identity(
    annotations: Sequence[Mapping[str, Any]], chunked_dir: Path
) -> dict[str, Any]:
    entries = []
    for annotation in annotations:
        annotation_id = annotation["id"]
        if annotation["task"] in CATEGORY_TASKS["forward"]:
            names = [
                f"{annotation_id}_{index}.mp4"
                for index in range(len(annotation["test_info"]))
            ]
        else:
            names = [f"{annotation_id}.mp4"]
        for name in names:
            path = chunked_dir / name
            if not path.is_file():
                raise EvaluationError(f"required chunk is missing: {path}")
            stat = path.stat()
            entries.append(
                {
                    "path": name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {
        "count": len(entries),
        "metadata_sha256": hashlib.sha256(_canonical_json(entries)).hexdigest(),
    }


def _build_run_manifest(
    *,
    annotations: Sequence[Mapping[str, Any]],
    annotations_path: Path,
    ovo_root: Path,
    dispider_root: Path,
    model_path: Path,
    data_root: Path,
    chunked_dir: Path,
    video_dir: Path,
    num_shards: int,
) -> dict[str, Any]:
    source_files = [
        ovo_root / "inference.py",
        ovo_root / "constant.py",
        ovo_root / "utils" / "OVOBench.py",
        ovo_root / "models" / "Dispider.py",
    ]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": "dispider-ovobench-offline-current-edition-v1",
        "annotations": {
            "path": str(annotations_path),
            "size": annotations_path.stat().st_size,
            "sha256": _sha256_file(annotations_path),
        },
        "ovo_root": str(ovo_root),
        "ovo_sources": [
            {
                "path": path.relative_to(ovo_root).as_posix(),
                "sha256": _sha256_file(_require_file(path, "OVO-Bench source")),
            }
            for path in source_files
        ],
        "dispider_root": str(dispider_root),
        "dispider_sources": _source_identity(dispider_root / "dispider"),
        "driver_sha256": _sha256_file(Path(__file__).resolve()),
        "model_path": str(model_path),
        "model_files": _model_identity(model_path),
        "data_root": str(data_root),
        "chunked_dir": str(chunked_dir),
        "chunked_videos": _chunk_identity(annotations, chunked_dir),
        "video_dir": str(video_dir),
        "num_shards": num_shards,
        "shard_algorithm": SHARD_ALGORITHM,
        "tasks": list(TASK_ORDER),
    }


def _worker_result(output_dir: Path, shard_id: int) -> Path:
    return output_dir / "workers" / f"shard_{shard_id:03d}" / "Dispider" / RESULT_NAME


def _worker_log(output_dir: Path, shard_id: int) -> Path:
    return output_dir / "logs" / f"shard_{shard_id:03d}.log"


def _shard_path(output_dir: Path, shard_id: int) -> Path:
    return output_dir / "shards" / f"shard_{shard_id:03d}.json"


def _worker_command(
    *,
    python: Path,
    ovo_root: Path,
    model_path: Path,
    video_dir: Path,
    chunked_dir: Path,
    output_dir: Path,
    shard_id: int,
) -> list[str]:
    return [
        str(python),
        str(ovo_root / "inference.py"),
        "--mode",
        "offline",
        "--model",
        "Dispider",
        "--model_path",
        str(model_path),
        "--anno_path",
        str(_shard_path(output_dir, shard_id)),
        "--video_dir",
        str(video_dir),
        "--chunked_dir",
        str(chunked_dir),
        "--result_dir",
        str(output_dir / "workers" / f"shard_{shard_id:03d}"),
        "--task",
        *TASK_ORDER,
    ]


def _validate_worker(path: Path, shard: Sequence[Mapping[str, Any]]) -> None:
    _, null_count = _merge_results([path], shard)
    if null_count:
        raise EvaluationError(f"{path} contains {null_count} null responses")


def _print_report(report: Mapping[str, Any]) -> None:
    print("task  correct/total  accuracy")
    for task in TASK_ORDER:
        score = report["tasks"][task]
        print(
            f"{task:<4}  {score['correct']:>4}/{score['total']:<4}  "
            f"{score['accuracy']:>7.2f}"
        )
    print()
    for category in CATEGORY_TASKS:
        score = report["categories"][category]
        print(f"{category:<8} macro: {score['macro_accuracy']:.2f}")
    overall = report["overall"]
    print(
        "official three-category macro: "
        f"{overall['official_three_category_macro_accuracy']:.2f}"
    )


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",")]
    if not gpus or any(not item for item in gpus):
        raise argparse.ArgumentTypeError(
            "GPUs must be a comma-separated non-empty list"
        )
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("GPU identifiers must be unique")
    return gpus


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Dispider on the pinned current OVO-Bench edition. The script "
            "runs one weighted shard per GPU, resumes complete shards, strictly "
            "merges every result, and reports the official three-category macro."
        )
    )
    parser.add_argument(
        "--ovo-root",
        type=Path,
        required=True,
        help="checkout of github.com/JoeLeelyf/OVO-Bench",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="dataset directory containing chunked_videos/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="default: <ovo-root>/data/ovo_bench_new.json",
    )
    parser.add_argument(
        "--chunked-dir",
        type=Path,
        help="default: <data-root>/chunked_videos",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help=(
            "source-video prefix recorded in result metadata; default: "
            "<data-root>/src_videos (the directory need not exist when using "
            "pre-chunked clips)"
        ),
    )
    parser.add_argument(
        "--gpus",
        type=_parse_gpus,
        default=_parse_gpus("0"),
        help="comma-separated CUDA device IDs; one weighted shard per GPU (default: 0)",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used by OVO-Bench workers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip only shards whose complete output passes strict validation",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--score-only",
        action="store_true",
        help="validate and score existing worker outputs without launching workers",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the shard plan and commands without writing",
    )
    return parser.parse_args(argv)


def _prepare_paths(args: argparse.Namespace) -> dict[str, Path]:
    dispider_root = Path(__file__).resolve().parents[1]
    ovo_root = _require_directory(args.ovo_root, "OVO-Bench root")
    model_path = _require_directory(args.model_path, "model path")
    data_root = _require_directory(args.data_root, "data root")
    annotations = _require_file(
        args.annotations or ovo_root / "data" / "ovo_bench_new.json",
        "annotations",
    )
    chunked_dir = _require_directory(
        args.chunked_dir or data_root / "chunked_videos", "chunked video directory"
    )
    video_dir = (args.video_dir or data_root / "src_videos").expanduser().resolve()
    python = _require_file(args.python, "Python executable")
    output_dir = args.output_dir.expanduser().resolve()
    return {
        "dispider_root": dispider_root,
        "ovo_root": ovo_root,
        "model_path": model_path,
        "data_root": data_root,
        "annotations": annotations,
        "chunked_dir": chunked_dir,
        "video_dir": video_dir,
        "python": python,
        "output_dir": output_dir,
    }


def _run(args: argparse.Namespace) -> int:
    paths = _prepare_paths(args)
    annotations = _load_json(paths["annotations"])
    if not isinstance(annotations, list):
        raise EvaluationError("annotations JSON must contain a list")
    expected = _annotation_manifest(annotations)
    _validate_current_edition(expected)
    shards = _make_shards(annotations, len(args.gpus))
    run_manifest = _build_run_manifest(
        annotations=annotations,
        annotations_path=paths["annotations"],
        ovo_root=paths["ovo_root"],
        dispider_root=paths["dispider_root"],
        model_path=paths["model_path"],
        data_root=paths["data_root"],
        chunked_dir=paths["chunked_dir"],
        video_dir=paths["video_dir"],
        num_shards=len(shards),
    )

    if args.dry_run:
        print(
            f"validated {expected['total_rows']} rows / "
            f"{expected['total_calls']} inference calls"
        )
        for shard_id, (gpu, shard) in enumerate(zip(args.gpus, shards)):
            command = _worker_command(
                python=paths["python"],
                ovo_root=paths["ovo_root"],
                model_path=paths["model_path"],
                video_dir=paths["video_dir"],
                chunked_dir=paths["chunked_dir"],
                output_dir=paths["output_dir"],
                shard_id=shard_id,
            )
            print(
                f"shard {shard_id}: GPU {gpu}, rows={len(shard)}, "
                f"calls={sum(_call_count(row) for row in shard)}"
            )
            print(f"  CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} {shlex.join(command)}")
        return 0

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".evaluate.lock"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise EvaluationError(
            f"another evaluator holds the output lock {lock_path}"
        ) from error

    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        existing_manifest = _load_json(manifest_path)
        if existing_manifest != run_manifest:
            raise EvaluationError(
                f"resume identity differs from {manifest_path}; "
                "use a new output directory"
            )
        if not (args.resume or args.score_only):
            raise EvaluationError(
                f"output already exists at {output_dir}; "
                "add --resume or use a new directory"
            )
    else:
        existing_results = list((output_dir / "workers").glob("**/*.json"))
        if existing_results:
            raise EvaluationError(
                "worker results exist without a run manifest; refusing unsafe resume"
            )
        if args.score_only:
            raise EvaluationError(f"score-only requires {manifest_path}")
        _write_json(manifest_path, run_manifest)

    for shard_id, shard in enumerate(shards):
        shard_path = _shard_path(output_dir, shard_id)
        if shard_path.exists():
            if _load_json(shard_path) != shard:
                raise EvaluationError(
                    f"stored shard differs from current plan: {shard_path}"
                )
        else:
            _write_json(shard_path, shard)

    pending: list[int] = []
    for shard_id, shard in enumerate(shards):
        result_path = _worker_result(output_dir, shard_id)
        if args.resume or args.score_only:
            try:
                _validate_worker(result_path, shard)
            except EvaluationError as error:
                if args.score_only:
                    raise EvaluationError(
                        f"shard {shard_id} is not complete: {error}"
                    ) from error
                print(f"rerun shard {shard_id}: {error}")
                pending.append(shard_id)
            else:
                print(f"resume shard {shard_id}: complete")
        elif result_path.exists():
            raise EvaluationError(
                f"worker output already exists: {result_path}; add --resume"
            )
        else:
            pending.append(shard_id)

    if not args.score_only and pending:
        processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
        for shard_id in pending:
            gpu = args.gpus[shard_id]
            log_path = _worker_log(output_dir, shard_id)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("ab")
            command = _worker_command(
                python=paths["python"],
                ovo_root=paths["ovo_root"],
                model_path=paths["model_path"],
                video_dir=paths["video_dir"],
                chunked_dir=paths["chunked_dir"],
                output_dir=output_dir,
                shard_id=shard_id,
            )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONUNBUFFERED"] = "1"
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = os.pathsep.join(
                item
                for item in (str(paths["dispider_root"]), existing_pythonpath)
                if item
            )
            process = subprocess.Popen(
                command,
                cwd=paths["ovo_root"],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((shard_id, process, log_handle))
            print(
                f"launched shard {shard_id} on GPU {gpu}: pid={process.pid}, "
                f"log={log_path}"
            )

        failures: list[str] = []
        active = list(processes)
        try:
            while active:
                remaining = []
                for shard_id, process, log_handle in active:
                    return_code = process.poll()
                    if return_code is None:
                        remaining.append((shard_id, process, log_handle))
                        continue
                    log_handle.close()
                    if return_code:
                        failures.append(
                            f"shard {shard_id} exited {return_code}; "
                            f"see {_worker_log(output_dir, shard_id)}"
                        )
                    else:
                        print(f"worker shard {shard_id} exited successfully")
                active = remaining
                if active:
                    time.sleep(1)
        except KeyboardInterrupt:
            for _, process, _ in active:
                process.terminate()
            for _, process, log_handle in active:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log_handle.close()
            raise EvaluationError(
                "interrupted; launched workers were stopped"
            ) from None
        if failures:
            raise EvaluationError("; ".join(failures))

    result_paths = []
    for shard_id, shard in enumerate(shards):
        result_path = _worker_result(output_dir, shard_id)
        _validate_worker(result_path, shard)
        result_paths.append(result_path)

    merged, report = _aggregate_and_score(result_paths, annotations)
    report["provenance"] = {
        "annotations": str(paths["annotations"]),
        "input_files": [str(path) for path in result_paths],
        "benchmark": "OVO-Bench current edition",
        "model": "Dispider",
    }
    merged_path = output_dir / "predictions.json"
    report_path = output_dir / "score_report.json"
    _write_json(merged_path, merged)
    _write_json(report_path, report)
    _print_report(report)
    print(f"merged predictions: {merged_path}")
    print(f"score report: {report_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(_parse_args(argv))
    except EvaluationError as error:
        print(f"evaluation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

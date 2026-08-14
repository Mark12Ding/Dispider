from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_evaluator() -> ModuleType:
    path = REPOSITORY_ROOT / "scripts" / "evaluate_ovobench.py"
    spec = importlib.util.spec_from_file_location("evaluate_ovobench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_evaluator()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _call_counts(rows: int, calls: int) -> list[int]:
    base, remainder = divmod(calls, rows)
    return [base + int(index < remainder) for index in range(rows)]


def _complete_current_edition() -> tuple[list[dict], dict[str, list[dict]]]:
    annotations = []
    results = {category: [] for category in EVALUATOR.CATEGORY_TASKS}
    next_id = 0

    for task in EVALUATOR.TASK_ORDER:
        category = EVALUATOR.TASK_CATEGORY[task]
        rows = EVALUATOR.CURRENT_EDITION_ROWS[task]
        calls = EVALUATOR.CURRENT_EDITION_CALLS[task]
        per_row_calls = _call_counts(rows, calls)
        for row_index, row_calls in enumerate(per_row_calls):
            annotation_id = next_id
            next_id += 1
            video = f"clips/{annotation_id}.mp4"
            if category != "forward":
                annotation = {
                    "id": annotation_id,
                    "task": task,
                    "video": video,
                    "question": f"Question {annotation_id}",
                    "gt": 0,
                }
                result = {
                    "id": annotation_id,
                    "video": f"/dataset/{video}",
                    "task": task,
                    "question": annotation["question"],
                    "response": "A.",
                    "ground_truth": "A",
                }
            else:
                test_info = []
                observed_test_info = []
                for call_index in range(row_calls):
                    if task == "REC":
                        expected = {"count": call_index}
                        response = f"There are {call_index}."
                    else:
                        expected = {"type": call_index % 2}
                        response = "Y" if expected["type"] else "N"
                    test_info.append(expected)
                    observed_test_info.append({**expected, "response": response})
                annotation = {
                    "id": annotation_id,
                    "task": task,
                    "video": video,
                    "test_info": test_info,
                    "row": row_index,
                }
                result = copy.deepcopy(annotation)
                result["video"] = f"/dataset/{video}"
                result["test_info"] = observed_test_info
            annotations.append(annotation)
            results[category].append(result)

    return annotations, results


def test_complete_prediction_file_is_validated_and_scored_offline(
    tmp_path: Path,
) -> None:
    annotations, predictions = _complete_current_edition()
    prediction_path = tmp_path / "complete_predictions.json"
    _write_json(prediction_path, predictions)

    merged, report = EVALUATOR._aggregate_and_score([prediction_path], annotations)

    expected_ids = {
        category: [
            annotation["id"]
            for annotation in annotations
            if EVALUATOR.TASK_CATEGORY[annotation["task"]] == category
        ]
        for category in EVALUATOR.CATEGORY_TASKS
    }
    assert {
        category: [result["id"] for result in merged[category]]
        for category in EVALUATOR.CATEGORY_TASKS
    } == expected_ids
    assert report["manifest"]["expected"] == report["manifest"]["observed"]
    assert report["manifest"]["observed"]["total_rows"] == 1640
    assert report["manifest"]["observed"]["total_calls"] == 3035
    assert report["overall"]["official_three_category_macro_accuracy"] == 100.0
    assert report["overall"]["null_responses"] == 0


def test_merge_restores_annotation_order_across_worker_files(
    tmp_path: Path,
) -> None:
    annotations, predictions = _complete_current_edition()
    even = {category: [] for category in EVALUATOR.CATEGORY_TASKS}
    odd = {category: [] for category in EVALUATOR.CATEGORY_TASKS}
    for category in EVALUATOR.CATEGORY_TASKS:
        for result in reversed(predictions[category]):
            target = even if result["id"] % 2 == 0 else odd
            target[category].append(result)
    even_path = tmp_path / "worker_even.json"
    odd_path = tmp_path / "worker_odd.json"
    _write_json(even_path, even)
    _write_json(odd_path, odd)

    merged, _ = EVALUATOR._aggregate_and_score([odd_path, even_path], annotations)

    assert [
        result["id"]
        for category in EVALUATOR.CATEGORY_TASKS
        for result in merged[category]
    ] == [annotation["id"] for annotation in annotations]


def test_worker_validator_rejects_null_responses(tmp_path: Path) -> None:
    annotations, predictions = _complete_current_edition()
    predictions["backward"][0]["response"] = None
    prediction_path = tmp_path / "null_response.json"
    _write_json(prediction_path, predictions)

    with pytest.raises(EVALUATOR.EvaluationError, match="1 null responses"):
        EVALUATOR._validate_worker(prediction_path, annotations)


def test_merge_rejects_changed_annotation_fields(tmp_path: Path) -> None:
    annotations, predictions = _complete_current_edition()
    predictions["forward"][0]["test_info"][0]["count"] = -1
    prediction_path = tmp_path / "changed_result.json"
    _write_json(prediction_path, predictions)

    with pytest.raises(EVALUATOR.EvaluationError, match="field 'count' differs"):
        EVALUATOR._merge_results([prediction_path], annotations)


def test_merge_rejects_duplicate_results(tmp_path: Path) -> None:
    annotations, predictions = _complete_current_edition()
    duplicate = {category: [] for category in EVALUATOR.CATEGORY_TASKS}
    duplicate["backward"].append(predictions["backward"][0])
    predictions_path = tmp_path / "predictions.json"
    duplicate_path = tmp_path / "duplicate.json"
    _write_json(predictions_path, predictions)
    _write_json(duplicate_path, duplicate)

    with pytest.raises(EVALUATOR.EvaluationError, match="duplicate result id"):
        EVALUATOR._merge_results([predictions_path, duplicate_path], annotations)

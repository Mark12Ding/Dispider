from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from dispider.model.checkpoint import (
    CHECKPOINT_MANIFEST,
    EMBEDDED_PERCEPTION_DECISION_PREFIX,
    load_checkpoint_layout,
    prepare_perception_decision_config,
    prepare_reaction_config,
    resolve_checkpoint_root,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _make_checkpoint(root: Path) -> Path:
    metadata = root / "perception_decision"
    vision = metadata / "vision_tower"
    vision.mkdir(parents=True)
    _write_json(
        root / "config.json",
        {
            "checkpoint_format_version": 2,
            "perception_decision": "perception_decision",
        },
    )
    _write_json(
        metadata / "config.json",
        {"perception_vision_tower": "vision_tower"},
    )
    _write_json(vision / "config.json", {"model_type": "clip_vision_model"})
    _write_json(vision / "preprocessor_config.json", {"size": 224})
    _write_json(
        root / CHECKPOINT_MANIFEST,
        {
            "format_version": 2,
            "embedded_perception_decision": {
                "state_dict_prefix": EMBEDDED_PERCEPTION_DECISION_PREFIX,
                "verified_key_count": 1,
            },
        },
    )
    save_file(
        {
            "model.layers.0.weight": torch.ones(1),
            EMBEDDED_PERCEPTION_DECISION_PREFIX + "model.layer.weight": torch.ones(1),
        },
        root / "model.safetensors",
    )
    return root


def test_current_checkpoint_layout_and_runtime_paths(tmp_path: Path) -> None:
    root = _make_checkpoint(tmp_path / "checkpoint")

    layout = load_checkpoint_layout(root)

    assert layout.root == root
    assert layout.perception_decision_dir == root / "perception_decision"
    reaction = SimpleNamespace()
    prepare_reaction_config(reaction, layout)
    assert reaction.perception_decision == str(root / "perception_decision")

    decision = SimpleNamespace(perception_vision_tower="vision_tower")
    prepare_perception_decision_config(decision, layout.perception_decision_dir)
    assert decision.perception_vision_tower == str(
        root / "perception_decision/vision_tower"
    )
    assert decision._dispider_embedded_weights is True


def test_hub_checkpoint_is_materialized_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    resolved = resolve_checkpoint_root(
        "organization/model",
        revision="revision",
        token="token",
        local_files_only=True,
    )

    assert resolved == snapshot
    assert calls == [
        {
            "repo_id": "organization/model",
            "cache_dir": None,
            "revision": "revision",
            "token": "token",
            "local_files_only": True,
        }
    ]


@pytest.mark.parametrize("missing", ["config.json", CHECKPOINT_MANIFEST])
def test_required_metadata_is_fail_closed(tmp_path: Path, missing: str) -> None:
    root = _make_checkpoint(tmp_path / "checkpoint")
    (root / missing).unlink()

    with pytest.raises(FileNotFoundError, match="metadata is missing"):
        load_checkpoint_layout(root)


def test_unsupported_checkpoint_version_is_rejected(tmp_path: Path) -> None:
    root = _make_checkpoint(tmp_path / "checkpoint")
    config = json.loads((root / "config.json").read_text())
    config["checkpoint_format_version"] = 1
    _write_json(root / "config.json", config)

    with pytest.raises(ValueError, match="Only Dispider checkpoint format v2"):
        load_checkpoint_layout(root)


@pytest.mark.parametrize("reference", ["/absolute/path", "../escape"])
def test_perception_decision_path_must_be_portable(
    tmp_path: Path,
    reference: str,
) -> None:
    root = _make_checkpoint(tmp_path / "checkpoint")
    config = json.loads((root / "config.json").read_text())
    config["perception_decision"] = reference
    _write_json(root / "config.json", config)

    with pytest.raises(ValueError, match="relative|escapes"):
        load_checkpoint_layout(root)


def test_duplicate_child_weights_are_rejected(tmp_path: Path) -> None:
    root = _make_checkpoint(tmp_path / "checkpoint")
    save_file(
        {"weight": torch.ones(1)},
        root / "perception_decision/model.safetensors",
    )

    with pytest.raises(ValueError, match="embedded only once"):
        load_checkpoint_layout(root)


def test_missing_embedded_weights_are_rejected(tmp_path: Path) -> None:
    root = _make_checkpoint(tmp_path / "checkpoint")
    save_file({"model.weight": torch.ones(1)}, root / "model.safetensors")

    with pytest.raises(ValueError, match="no embedded Perception/Decision"):
        load_checkpoint_layout(root)


def test_manifest_key_count_is_verified(tmp_path: Path) -> None:
    root = _make_checkpoint(tmp_path / "checkpoint")
    manifest = json.loads((root / CHECKPOINT_MANIFEST).read_text())
    manifest["embedded_perception_decision"]["verified_key_count"] = 2
    _write_json(root / CHECKPOINT_MANIFEST, manifest)

    with pytest.raises(ValueError, match="does not match weights"):
        load_checkpoint_layout(root)

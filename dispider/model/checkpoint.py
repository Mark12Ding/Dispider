"""Validation and path resolution for the portable Dispider checkpoint."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union


CHECKPOINT_FORMAT_VERSION = 2
CHECKPOINT_MANIFEST = "dispider_checkpoint_manifest.json"
PERCEPTION_DECISION_DIR = "perception_decision"
EMBEDDED_PERCEPTION_DECISION_PREFIX = "model.compressor.compressor."

PathLike = Union[str, os.PathLike]


@dataclass(frozen=True)
class CheckpointLayout:
    root: Path
    perception_decision_dir: Path
    manifest: Mapping[str, Any]


def resolve_checkpoint_root(
    model_path: PathLike,
    *,
    cache_dir: Optional[PathLike] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local checkpoint or materialize one Hub snapshot."""

    candidate = Path(model_path).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"Dispider checkpoint must be a directory: {candidate}")
        return candidate.resolve()

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=os.fspath(model_path),
        cache_dir=os.fspath(cache_dir) if cache_dir is not None else None,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )
    return Path(snapshot).resolve()


def load_checkpoint_layout(root: PathLike) -> CheckpointLayout:
    """Validate the current portable, deduplicated checkpoint layout."""

    root_path = Path(root).expanduser().resolve()
    config = _read_json(root_path / "config.json")
    manifest = _read_json(root_path / CHECKPOINT_MANIFEST)

    config_version = config.get("checkpoint_format_version")
    manifest_version = manifest.get("format_version")
    if (
        config_version != CHECKPOINT_FORMAT_VERSION
        or manifest_version != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            "Only Dispider checkpoint format v2 is supported; "
            f"found config={config_version!r}, manifest={manifest_version!r}"
        )

    reference = config.get("perception_decision")
    if not isinstance(reference, str) or not reference:
        raise ValueError("Checkpoint requires a `perception_decision` directory")
    perception_decision_dir = resolve_relative_resource(root_path, reference)
    if not perception_decision_dir.is_dir():
        raise FileNotFoundError(
            "Perception/Decision metadata directory does not exist: "
            f"{perception_decision_dir}"
        )

    _validate_metadata_only(perception_decision_dir)
    embedded_key_count = _validate_embedded_weights(root_path)
    _validate_manifest(manifest, embedded_key_count)
    return CheckpointLayout(root_path, perception_decision_dir, manifest)


def prepare_reaction_config(config: Any, layout: CheckpointLayout) -> Any:
    config.perception_decision = os.fspath(layout.perception_decision_dir)
    config.checkpoint_format_version = CHECKPOINT_FORMAT_VERSION
    config._dispider_checkpoint_root = os.fspath(layout.root)
    return config


def prepare_perception_decision_config(config: Any, metadata_dir: PathLike) -> Any:
    metadata_path = Path(metadata_dir).expanduser().resolve()
    reference = getattr(config, "perception_vision_tower", None)
    if not isinstance(reference, str) or not reference:
        raise ValueError(
            "Perception/Decision config requires `perception_vision_tower`"
        )

    vision_path = resolve_relative_resource(metadata_path, reference)
    if not vision_path.is_dir():
        raise FileNotFoundError(
            f"Perception metadata directory does not exist: {vision_path}"
        )
    config.perception_vision_tower = os.fspath(vision_path)
    config._dispider_embedded_weights = True
    return config


def resolve_relative_resource(base_dir: PathLike, reference: PathLike) -> Path:
    base = Path(base_dir).expanduser().resolve()
    resource = Path(reference).expanduser()
    if resource.is_absolute():
        raise ValueError(f"Checkpoint resource must be relative: {resource}")
    resolved = (base / resource).resolve()
    if not _is_relative_to(resolved, base):
        raise ValueError(f"Checkpoint resource escapes its repository: {reference}")
    return resolved


def _validate_metadata_only(directory: Path) -> None:
    forbidden = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
            forbidden.append(path.relative_to(directory))
        elif path.name.endswith(".index.json"):
            forbidden.append(path.relative_to(directory))
    if forbidden:
        files = ", ".join(map(os.fspath, forbidden))
        raise ValueError(
            "Perception/Decision weights must be embedded only once in the "
            f"composite checkpoint; found {files}"
        )


def _validate_embedded_weights(root: Path) -> int:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = _read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid safetensors index: {index_path}")
        if not all(
            isinstance(key, str) and isinstance(filename, str)
            for key, filename in weight_map.items()
        ):
            raise ValueError(f"Invalid safetensors weight map: {index_path}")
        missing_files = sorted(
            {
                filename
                for filename in weight_map.values()
                if not (root / filename).is_file()
            }
        )
        if missing_files:
            raise FileNotFoundError(
                "Safetensors index references missing files: "
                + ", ".join(missing_files)
            )
        keys = weight_map
    else:
        model_path = root / "model.safetensors"
        if not model_path.is_file():
            raise FileNotFoundError(
                "Checkpoint requires model.safetensors or "
                "model.safetensors.index.json"
            )
        from safetensors import safe_open

        try:
            with safe_open(model_path, framework="pt", device="cpu") as handle:
                keys = tuple(handle.keys())
        except Exception as error:
            raise ValueError(f"Invalid safetensors checkpoint: {model_path}") from error

    key_count = sum(key.startswith(EMBEDDED_PERCEPTION_DECISION_PREFIX) for key in keys)
    if not key_count:
        raise ValueError("Checkpoint has no embedded Perception/Decision weights")
    return key_count


def _validate_manifest(manifest: Mapping[str, Any], key_count: int) -> None:
    section = manifest.get("embedded_perception_decision")
    if not isinstance(section, Mapping):
        raise ValueError("Checkpoint manifest has no Perception/Decision section")
    if section.get("state_dict_prefix") != EMBEDDED_PERCEPTION_DECISION_PREFIX:
        raise ValueError("Checkpoint manifest has an incompatible state-dict prefix")
    expected_count = section.get("verified_key_count")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count != key_count
    ):
        raise ValueError(
            "Checkpoint manifest embedded key count does not match weights: "
            f"{expected_count!r} != {key_count}"
        )


def _read_json(path: Path) -> MutableMapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint metadata is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid checkpoint metadata: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint metadata must be a JSON object: {path}")
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True

"""Canonical Dispider model API."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Decision": (".decision", "Decision"),
    "DecisionConfig": (".decision", "DecisionConfig"),
    "Perception": (".perception", "Perception"),
    "PerceptionDecision": (".perception_decision", "PerceptionDecision"),
    "Reaction": (".reaction", "Reaction"),
    "ReactionConfig": (".reaction", "ReactionConfig"),
    "load_pretrained_model": (".builder", "load_pretrained_model"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name, package=__name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

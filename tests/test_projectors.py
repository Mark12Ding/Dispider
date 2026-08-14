from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dispider.model.projectors import (
    PoolProjector,
    build_vision_projector,
)


def _reference_mlp(input_size, hidden_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )


def _reference_projector(config):
    mlp = _reference_mlp(4 * config.perception_hidden_size, config.hidden_size)
    return PoolProjector(mlp, config.resolution, config.pool_num)


def _config(projector_type):
    return SimpleNamespace(
        perception_projector_type=projector_type,
        perception_hidden_size=2,
        hidden_size=3,
        pool_num=4,
        resolution=4,
    )


def test_released_pool_projector_state_dict_and_output():
    config = _config("pool")
    expected_keys = [
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.2.weight",
        "mlp.2.bias",
    ]

    torch.manual_seed(7)
    reference = _reference_projector(config).eval()
    torch.manual_seed(7)
    projector = build_vision_projector(config).eval()

    reference_state = reference.state_dict()
    state = projector.state_dict()
    assert type(projector) is type(reference)
    assert [type(module) for module in projector.modules()] == [
        type(module) for module in reference.modules()
    ]
    assert list(reference_state) == expected_keys
    assert list(state) == expected_keys
    for key in expected_keys:
        assert torch.equal(state[key], reference_state[key])

    inputs = torch.randn(2, 8, 4 * config.perception_hidden_size)

    reference_output = reference(inputs)
    output = projector(inputs)
    assert torch.equal(output, reference_output)


def test_decision_rejects_non_pool_projector():
    with pytest.raises(ValueError, match="requires perception_projector_type='pool'"):
        build_vision_projector(_config("mlp2x_gelu"))

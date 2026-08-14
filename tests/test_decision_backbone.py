import pytest
import torch

from dispider.model.decision_backbone.adapter import _validate_cache_position


def _tuple_cache(length):
    key = torch.zeros(1, 2, length, 4)
    value = torch.zeros_like(key)
    return ((key, value),)


def test_cache_position_continues_tuple_kv_cache():
    _validate_cache_position(
        torch.tensor([3, 4]),
        _tuple_cache(3),
        sequence_length=2,
    )


@pytest.mark.parametrize("position", [torch.tensor([2, 3]), torch.tensor([3])])
def test_cache_position_rejects_non_contiguous_values(position):
    with pytest.raises(ValueError, match="contiguous cache_position"):
        _validate_cache_position(position, _tuple_cache(3), sequence_length=2)

from types import SimpleNamespace

import torch
import torch.nn as nn

from dispider.constants import IMAGE_TOKEN_INDEX
from dispider.model.reaction import _ReactionMultimodalMixin


class _EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 2)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.arange(32, dtype=torch.float32).view(16, 2)
            )


class _ReactionHarness(_ReactionMultimodalMixin):
    def __init__(self, padding_side="right"):
        self.model = _EmbeddingModel()
        self.config = SimpleNamespace(tokenizer_padding_side=padding_side)

    def get_model(self):
        return self.model


def test_interleave_image_features_preserves_order():
    harness = _ReactionHarness()
    input_ids = [
        torch.tensor([1, 2]),
        torch.tensor([3, IMAGE_TOKEN_INDEX, 4]),
    ]
    image_features = [
        torch.tensor([[100.0, 101.0]]),
        torch.tensor([[200.0, 201.0], [202.0, 203.0]]),
    ]

    embeds = harness._interleave_image_features(
        input_ids,
        image_features,
    )

    assert torch.equal(embeds[0], harness.model.embed_tokens(input_ids[0]))
    assert torch.equal(
        embeds[1],
        torch.tensor(
            [
                [6.0, 7.0],
                [200.0, 201.0],
                [202.0, 203.0],
                [8.0, 9.0],
            ]
        ),
    )


def test_pad_multimodal_batch_supports_both_padding_sides():
    embeds = [torch.tensor([[1.0, 2.0]]), torch.ones(3, 2)]
    attention_mask = torch.ones(2, 3, dtype=torch.bool)
    position_ids = torch.arange(3)

    right = _ReactionHarness("right")._pad_multimodal_batch(
        embeds,
        attention_mask,
        position_ids,
    )
    left = _ReactionHarness("left")._pad_multimodal_batch(
        embeds,
        attention_mask,
        position_ids,
    )

    assert right[0].shape == (2, 3, 2)
    assert torch.equal(right[1][0], torch.tensor([True, False, False]))
    assert torch.equal(right[2][0], torch.tensor([0, 0, 0]))
    assert torch.equal(left[1][0], torch.tensor([False, False, True]))
    assert torch.equal(left[2][0], torch.tensor([0, 0, 0]))

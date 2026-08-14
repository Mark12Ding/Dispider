from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dispider.constants import IMAGE_TOKEN_INDEX
from dispider.model.decision_inputs import (
    DecisionInputsMixin,
    _encode_in_chunks,
    _interleave_text_and_images,
    _pad_multimodal_batch,
    _split_text_segments,
)


def test_chunked_vision_encoding_preserves_order() -> None:
    calls = []

    def encoder(batch):
        calls.append(batch.shape[0])
        return batch * 2

    images = torch.arange(35, dtype=torch.float32).unsqueeze(1)

    output = _encode_in_chunks(images, encoder)

    assert calls == [16, 16, 3]
    torch.testing.assert_close(output, images * 2)


def test_text_image_interleave_preserves_order_and_indicators() -> None:
    input_ids = torch.tensor([5, IMAGE_TOKEN_INDEX, 6, 7, IMAGE_TOKEN_INDEX, 8])
    indicators = torch.zeros_like(input_ids)

    def embed_tokens(token_ids):
        values = token_ids.to(torch.float32)
        return torch.stack((values, -values), dim=-1)

    text_embeddings, text_indicators = _split_text_segments(
        input_ids,
        indicators,
        embed_tokens,
    )
    images = (
        torch.arange(12, dtype=torch.float32).view(6, 2),
        torch.arange(12, 24, dtype=torch.float32).view(6, 2),
    )
    projector = SimpleNamespace(resolution=4, pool_num=1)

    output, output_indicators, image_index = _interleave_text_and_images(
        text_embeddings,
        text_indicators,
        images,
        0,
        projector,
    )

    expected = torch.cat(
        (
            embed_tokens(torch.tensor([5])),
            images[0],
            embed_tokens(torch.tensor([6, 7])),
            images[1],
            embed_tokens(torch.tensor([8])),
        )
    )
    torch.testing.assert_close(output, expected)
    assert output_indicators.tolist() == (
        [0] + [2, 2, 2, 2, 100, 200] + [0, 0] + [2, 2, 2, 2, 100, 200] + [0]
    )
    assert image_index == 2


@pytest.mark.parametrize(
    ("padding_side", "first_attention", "first_positions"),
    (
        ("right", [True, True, False], [0, 1, 0]),
        ("left", [False, True, True], [0, 0, 1]),
    ),
)
def test_multimodal_padding_preserves_side_and_positions(
    padding_side,
    first_attention,
    first_positions,
) -> None:
    embeddings = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
    )
    indicators = (torch.tensor([100, 200]), torch.tensor([2, 100, 200]))

    padded = _pad_multimodal_batch(
        embeddings,
        indicators,
        torch.ones((2, 3), dtype=torch.bool),
        torch.arange(3),
        padding_side,
    )
    padded_embeddings, padded_indicators, attention, positions = padded

    assert padded_embeddings.shape == (2, 3, 2)
    assert attention[0].tolist() == first_attention
    assert positions[0].tolist() == first_positions
    assert padded_indicators[1].tolist() == [2, 100, 200]
    assert positions[1].tolist() == [0, 1, 2]


class _TinyVision(nn.Module):
    def forward(self, images):
        values = torch.arange(
            images.shape[0] * 4,
            dtype=images.dtype,
            device=images.device,
        )
        return values.view(images.shape[0], 4, 1)


class _TinyPoolProjector(nn.Module):
    resolution = 1
    pool_num = 1

    def forward(self, features):
        return torch.cat((features, features.mean(dim=1, keepdim=True)), dim=1)


class _TinyDecisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 4)
        self.mm_projector = _TinyPoolProjector()
        self.vision_tower = _TinyVision()

    def get_vision_tower(self):
        return self.vision_tower


class _TinyDecisionInputAdapter(DecisionInputsMixin):
    def __init__(self) -> None:
        self.model = _TinyDecisionModel()
        self.config = SimpleNamespace(tokenizer_padding_side="right")
        self.device = torch.device("cpu")

    def get_model(self):
        return self.model


def test_prepare_multimodal_keeps_eleven_item_decision_contract() -> None:
    adapter = _TinyDecisionInputAdapter()
    prepared = adapter.prepare_inference_inputs(
        input_ids=torch.tensor([[5, IMAGE_TOKEN_INDEX, 6]]),
        position_ids=None,
        attention_mask=None,
        question_ids=torch.tensor([[7, 8]]),
        question_mask=torch.ones((1, 2), dtype=torch.long),
        past_key_values=None,
        images=torch.ones((1, 2, 1, 1, 1)),
        answer_token=torch.tensor([[9]]),
        todo_token=torch.tensor([[10]]),
    )

    assert len(prepared) == 11
    assert prepared[4].shape == (1, 5, 4)
    assert prepared[8].tolist() == [[0, 2, 100, 200, 0]]
    assert prepared[9].shape == (1, 1, 4)
    assert prepared[10].shape == (1, 1, 4)

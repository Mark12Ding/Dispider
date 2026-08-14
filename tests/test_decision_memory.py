import torch

from dispider.model.decision_backbone.constants import (
    GLOBAL_MEMORY_TOKEN,
    MEMORY_TOKEN,
    QUERY_TOKEN,
    TIME_TOKEN,
)
from dispider.model.decision_backbone.memory import (
    QueryBatch,
    pad_and_concat,
    run_decoder,
)


class _RecordingDecoder:
    def __init__(self):
        self.arguments = None

    def __call__(self, input_ids, **kwargs):
        self.arguments = {"input_ids": input_ids, **kwargs}
        return (kwargs["inputs_embeds"],)


def test_run_decoder_disables_implicit_transformers_cache():
    decoder = _RecordingDecoder()
    owner = type("Owner", (), {"model": decoder})()
    embeddings = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)
    indicators = torch.zeros(2, 3, dtype=torch.long)

    output = run_decoder(owner, None, embeddings, mask, indicators, 7)

    assert output[0] is embeddings
    assert set(decoder.arguments) == {
        "input_ids",
        "inputs_embeds",
        "past_key_values",
        "attention_mask",
        "indicators",
        "select_layer",
        "use_cache",
    }
    assert decoder.arguments["input_ids"] is None
    assert decoder.arguments["inputs_embeds"] is embeddings
    assert decoder.arguments["past_key_values"] is None
    assert decoder.arguments["attention_mask"] is mask
    assert decoder.arguments["indicators"] is indicators
    assert decoder.arguments["select_layer"] == 7
    assert decoder.arguments["use_cache"] is False


def test_decision_token_roles_are_stable():
    assert QUERY_TOKEN == 1
    assert MEMORY_TOKEN == 100
    assert GLOBAL_MEMORY_TOKEN == 150
    assert TIME_TOKEN == 200


def test_pad_and_concat_preserves_values_metadata_and_device():
    batches = (
        QueryBatch(
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64),
            torch.tensor([[True, True]]),
            torch.tensor([[100, 200]], dtype=torch.int32),
        ),
        QueryBatch(
            torch.tensor(
                [[[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]],
                dtype=torch.float64,
            ),
            torch.tensor([[True, False, True]]),
            torch.tensor([[0, 1, 0]], dtype=torch.int32),
        ),
    )

    padded = pad_and_concat(batches)

    assert torch.equal(
        padded.embeddings,
        torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
                [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            ],
            dtype=torch.float64,
        ),
    )
    assert torch.equal(
        padded.mask,
        torch.tensor([[True, True, False], [True, False, True]]),
    )
    assert torch.equal(
        padded.indicators,
        torch.tensor([[100, 200, 0], [0, 1, 0]], dtype=torch.int32),
    )


def test_pad_and_concat_preserves_embedding_gradients():
    short = torch.tensor([[[1.0], [2.0]]], dtype=torch.float64, requires_grad=True)
    long = torch.tensor(
        [[[3.0], [4.0], [5.0]], [[6.0], [7.0], [8.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batches = (
        QueryBatch(short, torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.long)),
        QueryBatch(long, torch.ones(2, 3), torch.zeros(2, 3, dtype=torch.long)),
    )

    padded = pad_and_concat(batches)
    weights = torch.arange(1, 10, dtype=torch.float64).view(3, 3, 1)
    (padded.embeddings * weights).sum().backward()

    assert torch.equal(
        short.grad,
        torch.tensor([[[1.0], [2.0]]], dtype=torch.float64),
    )
    assert torch.equal(
        long.grad,
        torch.tensor(
            [[[4.0], [5.0], [6.0]], [[7.0], [8.0], [9.0]]],
            dtype=torch.float64,
        ),
    )

"""Small runtime helpers required by the pinned OVO-Bench adapter."""


def disable_torch_init():
    """Skip initialization because released weights overwrite every parameter."""

    import torch

    def skip_initialization(_module):
        return None

    torch.nn.Linear.reset_parameters = skip_initialization
    torch.nn.LayerNorm.reset_parameters = skip_initialization


__all__ = ["disable_torch_init"]

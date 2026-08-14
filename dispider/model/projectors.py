"""Projectors used by the released Dispider checkpoints."""

import math

import torch
from torch import nn


_POOL_PROJECTOR = "pool"


def build_mlp_projector(input_size, hidden_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )


class PoolProjector(nn.Module):
    def __init__(self, mlp, resolution, pool_num):
        super().__init__()
        self.mlp = mlp
        self.pool_num = pool_num
        self.resolution = resolution

    def forward(self, inputs):
        batch_size, token_count, channels = inputs.shape
        height = math.isqrt(self.resolution)
        grid_size = math.isqrt(self.pool_num)
        maps = inputs.view(
            batch_size,
            token_count // self.resolution,
            height,
            height,
            channels,
        )
        maps = maps.view(
            batch_size,
            token_count // self.resolution,
            grid_size,
            height // grid_size,
            grid_size,
            height // grid_size,
            channels,
        )
        maps = maps.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        maps = maps.view(
            batch_size,
            token_count // self.resolution,
            self.pool_num,
            self.resolution // self.pool_num,
            channels,
        )

        pooled = torch.mean(maps, dim=-2).view(
            batch_size,
            token_count // self.resolution * self.pool_num,
            channels,
        )
        global_pool = torch.mean(inputs, dim=1, keepdim=True)
        return self.mlp(torch.cat([inputs, pooled, global_pool], dim=1))


def build_vision_projector(config):
    projector_type = config.perception_projector_type
    if projector_type != _POOL_PROJECTOR:
        raise ValueError(
            "Released Decision requires perception_projector_type='pool'; "
            f"found {projector_type!r}"
        )
    mlp = build_mlp_projector(
        4 * config.perception_hidden_size,
        config.hidden_size,
    )
    return PoolProjector(
        mlp,
        config.resolution,
        getattr(config, "pool_num", 1),
    )


__all__ = ["PoolProjector", "build_mlp_projector", "build_vision_projector"]

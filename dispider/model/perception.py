"""CLIP perception used by Dispider's Perception/Decision module."""

import os

import torch
import torch.nn as nn

from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel


class Perception(nn.Module):
    def __init__(self, vision_tower, config):
        super().__init__()
        self.vision_tower_name = vision_tower
        self.select_layer = config.perception_select_layer
        self.select_feature = getattr(config, "perception_select_feature", "patch")
        self.load_model()

    def load_model(self):
        if not os.path.isdir(self.vision_tower_name):
            raise FileNotFoundError(
                "Perception metadata directory does not exist: "
                f"{self.vision_tower_name}"
            )
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name,
            local_files_only=True,
        )
        vision_config = CLIPVisionConfig.from_pretrained(
            self.vision_tower_name,
            local_files_only=True,
        )
        self.vision_tower = CLIPVisionModel(vision_config)
        self.vision_tower.requires_grad_(False)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.no_grad()
    def forward(self, images):
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
        )
        return self.feature_select(image_forward_outs).to(images.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size


def build_perception(config):
    """Build the CLIP perception tower described by a model config."""

    vision_tower = getattr(config, "perception_vision_tower", None)
    if not vision_tower:
        raise ValueError("Model config does not define `perception_vision_tower`")
    return Perception(vision_tower, config)


__all__ = ["Perception", "build_perception"]

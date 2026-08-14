"""Token and image helpers used by Dispider and the pinned OVO adapter."""

import torch
from PIL import Image

from dispider.constants import IMAGE_TOKEN_INDEX


def _expand_square(image, background_color):
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    result = Image.new(image.mode, (size, size), background_color)
    result.paste(image, ((size - width) // 2, (size - height) // 2))
    return result


def process_images(images, image_processor, model_config):
    """Preprocess images for the OVO adapter's public import contract."""

    if getattr(model_config, "image_aspect_ratio", None) != "pad":
        return image_processor(images, return_tensors="pt")["pixel_values"]

    background = tuple(int(channel * 255) for channel in image_processor.image_mean)
    tensors = [
        image_processor.preprocess(
            _expand_square(image, background),
            return_tensors="pt",
        )["pixel_values"][0]
        for image in images
    ]
    if all(tensor.shape == tensors[0].shape for tensor in tensors):
        return torch.stack(tensors)
    return tensors


def tokenizer_image_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    return_tensors=None,
):
    chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]
    offset = int(bool(chunks[0]) and chunks[0][0] == tokenizer.bos_token_id)
    input_ids = chunks[0][:offset]
    for index, chunk in enumerate(chunks):
        if index:
            input_ids.append(image_token_index)
        input_ids.extend(chunk[offset:])

    if return_tensors is None:
        return input_ids
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    raise ValueError(f"Unsupported tensor type: {return_tensors}")


def get_model_name_from_path(model_path):
    parts = model_path.strip("/").split("/")
    if parts[-1].startswith("checkpoint-"):
        return parts[-2] + "_" + parts[-1]
    return parts[-1]


__all__ = ["get_model_name_from_path", "process_images", "tokenizer_image_token"]

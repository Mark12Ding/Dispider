---
license: apache-2.0
library_name: transformers
pipeline_tag: video-text-to-text
inference: false
tags:
  - video
  - streaming-video
  - multimodal
  - qwen2
  - cvpr-2025
---

# Dispider

Official checkpoint for **Dispider: Enabling Video LLMs with Active Real-Time
Interaction via Disentangled Perception, Decision, and Reaction** (CVPR 2025).

- [Code](https://github.com/Mark12Ding/Dispider)
- [Paper](https://arxiv.org/abs/2501.03218)

Dispider continuously perceives a video stream, uses a compact Decision model
to determine when a response is needed, and invokes the larger Reaction model
only after a trigger.

## Architecture

| Component | Purpose | Backbone |
| --- | --- | --- |
| Perception | Encode each sampled 16-frame clip | CLIP vision tower |
| Perception-Decision | Build temporal memory | Compact Qwen2-1.5B |
| Decision | Trigger or remain silent | Decision head on the compact model |
| Reaction | Generate the response | Qwen2-7B |

## Checkpoint layout

The composite shards contain all model weights exactly once. The nested
`perception_decision/` directory contains metadata only.

```text
Dispider/
├── config.json
├── dispider_checkpoint_manifest.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
└── perception_decision/
    ├── config.json
    ├── tokenizer metadata
    └── vision_tower/
        ├── config.json
        └── preprocessor_config.json
```

Download it directly:

```bash
huggingface-cli download Mar2Ding/Dispider \
  --local-dir checkpoints/Dispider
```

Use the code repository with Python 3.10, PyTorch 2.2.0, FlashAttention
2.5.9.post1, and Transformers 4.41.2. Load either the repository ID or a local
snapshot through `dispider.model.load_pretrained_model`; generic
`AutoModel.from_pretrained()` does not assemble the video components.
The accompanying code is an inference-only runtime.

## Offline inference

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model_path Mar2Ding/Dispider \
  --video_path /path/to/video.mp4 \
  --prompt "What is happening in the video?"
```

## Streaming inference

```bash
CUDA_VISIBLE_DEVICES=0 python stream_inference.py /path/to/video.mp4 \
  --model Mar2Ding/Dispider \
  --prompt "Tell me when the person starts performing." \
  --decision-kv-cache auto
```

The command emits timestamped JSON responses as Decision triggers Reaction.
The optional Decision KV cache stores only stable prefixes and automatically
falls back to full computation near the trigger threshold.

## OVO-Bench example

From the code repository:

```bash
python scripts/evaluate_ovobench.py \
  --ovo-root ../OVO-Bench \
  --model-path checkpoints/Dispider \
  --data-root /path/to/ovo-data \
  --output-dir outputs/ovobench \
  --gpus 0,1,2,3 \
  --resume
```

The evaluator validates complete coverage, rejects missing or null outputs,
and computes the official category metrics. Generated results remain local.

## Citation

```bibtex
@inproceedings{qian2025dispider,
  title={Dispider: Enabling Video {LLMs} with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction},
  author={Qian, Rui and Ding, Shuangrui and Dong, Xiaoyi and Zhang, Pan and Zang, Yuhang and Cao, Yuhang and Lin, Dahua and Wang, Jiaqi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  pages={24045--24055},
  month={June},
  year={2025}
}
```

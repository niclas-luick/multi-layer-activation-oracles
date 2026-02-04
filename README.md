This repository is a personal research fork and heavily builds upon the [Activation Oracles](https://github.com/adamkarvonen/activation_oracles) project by Karvonen et al.

# Multi-Layer Activation Oracles (MLAOs)

## Overview
Standard Activation Oracles (AOs) interpret model activations from a single layer at a time. **Multi-Layer Activation Oracles (MLAOs)** promise to improve generalizability by processing activations from multiple layers **simultaneously**.

This repository contains the code to:
1.  **Train** and **Evaluate** MLAOs on the [Activation Oracles datasets](https://github.com/adamkarvonen/activation_oracles).
2.  **Visualize** the performance gains across models and evaluation tasks.

## Installation

```bash
uv sync
source .venv/bin/activate
huggingface-cli login --token <your_token>
```

## Quick Start: Demo

The easiest way to get started is with the demo notebook ([Colab](https://colab.research.google.com/drive/1NwWDkj6Daju-0_1Nvj7ScYIg62nR8SfB?usp=sharing) | [Local](experiments/multi_layer_AO_demo.ipynb)), which compares activation oracles and their multi-layer variants:
1. in a classification task setting, i.e. language identification
2. in detecting a misaligned model

## Pre-trained Models

Pre-trained oracle weights for Qwen3-4B and Qwen3-8B are available on Hugging Face: [Activation Oracles Collection](https://huggingface.co/collections/nluick/multi-layer-activation-oracles
)

## Training

To train MLAOs, use the training script with `torchrun`:

```bash
torchrun --nproc_per_node=<NUM_GPUS> nl_probes/sft.py
```

By default, this trains a three-layer MLAO-Qwen3-8B-3L-1N based on Qwen3-8B using a diverse mixture of training tasks:
- System prompt question-answering (LatentQA)
- Binary classification tasks
- Self-supervised context prediction

Automated selection of dataset mixtures is not implemented yet. To train the corresponding MLAO-Qwen3-8B-3L-3N variant you have to manually uncomment the loops "for _ in [25, 50, 75]" in [nl_probes/dataset_classes/classification](nl_probes/dataset_classes/classification)

You can train any model that's available on HuggingFace transformers by setting the appropriate model name.

Training configuration can be modified in `nl_probes/configs/sft_config.py`.



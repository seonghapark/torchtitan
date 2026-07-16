# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config for instruction fine-tuning gemma-7b via HF Transformers."""

from dataclasses import dataclass


@dataclass
class GemmaSFTConfig:
    # ---- Model ----
    # Local path (preferred) or HF hub id. Local path avoids the HF auth flow.
    model_name_or_path: str = "./assets/hf/gemma-7b"
    # HF revision used only when `model_name_or_path` is an HF hub id.
    model_revision: str = "main"
    # dtype used to load pretrained weights and cast the model.
    dtype: str = "bfloat16"
    # Enable HF gradient checkpointing (activation recomputation).
    activation_checkpoint: bool = True

    # ---- Data ----
    # HF hub dataset id. Ignored if `dataset_local_path` is set.
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_config_name: str | None = None
    dataset_split: str = "train"
    # Local JSON/JSONL file with a list of {instruction, input, output} records.
    dataset_local_path: str | None = None
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"
    max_seq_len: int = 2048
    # If True, tokens belonging to the user turn (prompt) get label -100 so
    # loss is computed only on the assistant response.
    mask_instruction: bool = True

    # ---- Training ----
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    seed: int = 42
    num_workers: int = 2

    # ---- Mixed precision (FSDP2) ----
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"

    # ---- IO ----
    output_dir: str = "outputs/gemma-7b-sft"
    log_interval: int = 10
    # Save an intermediate HF-format checkpoint every `save_interval` optimizer
    # steps. Set to 0 to disable intermediate saves.
    save_interval: int = 0

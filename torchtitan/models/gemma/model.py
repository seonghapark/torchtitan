# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thin wrapper around HuggingFace's GemmaForCausalLM for SFT."""

import torch
import torch.nn as nn
from transformers import GemmaForCausalLM


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{name}'. Valid: {sorted(_DTYPE_MAP.keys())}"
        )
    return _DTYPE_MAP[name]


class GemmaModel(nn.Module):
    """Instruction-tuning wrapper around ``GemmaForCausalLM``.

    Loads pretrained weights via ``from_pretrained`` and exposes a plain
    ``forward(input_ids, labels, attention_mask) -> loss`` interface so the
    training loop stays framework-agnostic.
    """

    def __init__(self, args):
        super().__init__()
        dtype = resolve_dtype(args.dtype)

        self.model = GemmaForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision=getattr(args, "model_revision", "main"),
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        # Cache is incompatible with training / gradient checkpointing.
        self.model.config.use_cache = False

        if getattr(args, "activation_checkpoint", False):
            # Non-reentrant checkpointing is required for FSDP2 compatibility.
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss

    def save_pretrained(self, save_directory: str) -> None:
        """Save the underlying HF model to `save_directory` (rank-0 only)."""
        self.model.save_pretrained(save_directory)

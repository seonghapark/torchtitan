# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Instruction fine-tune ``google/gemma-7b`` with FSDP2 + HF Transformers.

Launch (single node, 8 devices):
    torchrun --nproc_per_node=8 -m torchtitan.models.gemma.train \
        --model_name_or_path ./assets/hf/gemma-7b \
        --dataset_name tatsu-lab/alpaca \
        --output_dir outputs/gemma-7b-sft

Or use ``run_sft.sh``.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import tyro
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, DistributedSampler

from .config import GemmaSFTConfig
from .model import GemmaModel
from .parallelize import parallelize_gemma
from .sftdataset import SFTCollator, SFTDataset, load_sft_records
from .tokenizer import build_tokenizer


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _pick_backend_and_device() -> tuple[str, torch.device]:
    """Pick the right dist backend and per-rank device for CUDA or XPU."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return "nccl", torch.device(f"cuda:{local_rank}")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(local_rank)
        return "ccl", torch.device(f"xpu:{local_rank}")
    return "gloo", torch.device("cpu")


def _setup_distributed() -> tuple[torch.device, int, int]:
    backend, device = _pick_backend_and_device()
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return device, rank, world_size


def _is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _log(msg: str) -> None:
    if _is_rank0():
        print(msg, flush=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup + linear decay to 0."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    remain = max(0, total_steps - step)
    return remain / float(max(1, total_steps - warmup_steps))


# ---------------------------------------------------------------------------
# Checkpointing (HF format, rank-0 only)
# ---------------------------------------------------------------------------


def _save_hf_checkpoint(
    model: GemmaModel,
    tokenizer,
    save_dir: str,
) -> None:
    """Save an HF-format checkpoint by gathering full state on rank 0.

    Uses FSDP2's ``full_state_dict`` semantics: each parameter's ``full_tensor``
    is materialized on rank 0 while other ranks contribute their shards.
    """
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        StateDictOptions,
    )

    os.makedirs(save_dir, exist_ok=True) if _is_rank0() else None

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_sd = get_model_state_dict(model, options=options)

    if _is_rank0():
        # `full_sd` keys are prefixed with `model.` (the HF module inside
        # GemmaModel). Strip that prefix so HF's `from_pretrained` can consume
        # the checkpoint directly.
        prefix = "model."
        hf_sd = {}
        for k, v in full_sd.items():
            new_k = k[len(prefix):] if k.startswith(prefix) else k
            hf_sd[new_k] = v.detach().to("cpu")

        # Load into a lightweight (meta) copy for save_pretrained sharding.
        model.model.save_pretrained(save_dir, state_dict=hf_sd)
        tokenizer.save_pretrained(save_dir)
        print(f"[rank0] saved HF checkpoint to {save_dir}", flush=True)

    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: GemmaSFTConfig) -> None:
    device, rank, world_size = _setup_distributed()
    _set_seed(cfg.seed + rank)

    _log(f"[config]\n{cfg}")
    _log(f"[dist] world_size={world_size} device={device}")

    # ------------------------ Tokenizer + data ------------------------
    tokenizer = build_tokenizer(cfg.model_name_or_path)

    records = load_sft_records(
        dataset_name=cfg.dataset_name,
        dataset_config_name=cfg.dataset_config_name,
        dataset_split=cfg.dataset_split,
        dataset_local_path=cfg.dataset_local_path,
    )
    _log(f"[data] loaded {len(records)} records")

    dataset = SFTDataset(
        records,
        tokenizer,
        max_seq_len=cfg.max_seq_len,
        instruction_key=cfg.instruction_key,
        input_key=cfg.input_key,
        output_key=cfg.output_key,
        mask_instruction=cfg.mask_instruction,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.seed,
        drop_last=True,
    )
    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=(device.type in {"cuda", "xpu"}),
        drop_last=True,
    )

    # ------------------------ Model + parallelism ---------------------
    _log(f"[model] loading {cfg.model_name_or_path}")
    model = GemmaModel(cfg)
    # ``from_pretrained`` loads on CPU. Move to the per-rank device BEFORE
    # FSDP2 shards so each rank ends up holding only its shard on-device.
    model.to(device)

    mesh = init_device_mesh(
        device_type=device.type, mesh_shape=(world_size,), mesh_dim_names=("fsdp",)
    )
    model = parallelize_gemma(
        model,
        mesh["fsdp"],
        param_dtype=cfg.param_dtype,
        reduce_dtype=cfg.reduce_dtype,
    )
    model.train()

    # ------------------------ Optimizer + schedule --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        foreach=True,
    )

    steps_per_epoch = math.ceil(len(loader) / cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, warmup_steps, total_steps)
    )
    _log(
        f"[schedule] total_steps={total_steps} warmup_steps={warmup_steps} "
        f"steps_per_epoch={steps_per_epoch}"
    )

    # ------------------------ Training loop ---------------------------
    global_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    running_loss = 0.0

    for epoch in range(cfg.num_epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )
            (loss / cfg.gradient_accumulation_steps).backward()
            running_loss += loss.detach().float().item()
            micro_step += 1

            if micro_step % cfg.gradient_accumulation_steps != 0:
                continue

            # Optimizer step boundary.
            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.max_grad_norm
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % cfg.log_interval == 0:
                avg_loss = running_loss / (
                    cfg.log_interval * cfg.gradient_accumulation_steps
                )
                elapsed = time.perf_counter() - t0
                lr = scheduler.get_last_lr()[0]
                _log(
                    f"[step {global_step}/{total_steps}] "
                    f"epoch={epoch} loss={avg_loss:.4f} lr={lr:.3e} "
                    f"elapsed={elapsed:.1f}s"
                )
                running_loss = 0.0

            if (
                cfg.save_interval
                and global_step % cfg.save_interval == 0
                and global_step > 0
            ):
                _save_hf_checkpoint(
                    model,
                    tokenizer,
                    os.path.join(cfg.output_dir, f"step-{global_step}"),
                )

    # Final save.
    _save_hf_checkpoint(
        model, tokenizer, os.path.join(cfg.output_dir, "final")
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    cfg = tyro.cli(GemmaSFTConfig)
    try:
        main(cfg)
    except Exception:
        # Ensure non-zero exit propagates through torchrun cleanly.
        import traceback

        traceback.print_exc()
        sys.exit(1)

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


# Fallback lookup tables for env vars set by different launchers. torchrun
# populates ``RANK/WORLD_SIZE/LOCAL_RANK`` directly; MPI-style launchers
# (``mpiexec`` on Aurora / PALS / OpenMPI) use different names.
_RANK_ENV_KEYS = ("RANK", "PMI_RANK", "PALS_RANKID", "OMPI_COMM_WORLD_RANK")
_SIZE_ENV_KEYS = (
    "WORLD_SIZE",
    "PMI_SIZE",
    "PALS_NRANKS",
    "OMPI_COMM_WORLD_SIZE",
)
_LOCAL_RANK_ENV_KEYS = (
    "LOCAL_RANK",
    "PALS_LOCAL_RANKID",
    "PMI_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
)


def _first_env(keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = os.environ.get(k)
        if v is not None and v != "":
            return v
    return None


def _set_master_addr_via_mpi(comm=None) -> None:
    """Broadcast rank 0's hostname over MPI to populate ``MASTER_ADDR``.

    Required for multi-node ``mpiexec`` launches where every rank must agree
    on the same address. Falls back to the local hostname if ``mpi4py`` is
    unavailable (single-node case).
    """
    if comm is None:
        try:
            from mpi4py import MPI  # type: ignore
        except ImportError:
            import socket

            os.environ.setdefault("MASTER_ADDR", socket.gethostname())
            return
        comm = MPI.COMM_WORLD
    else:
        from mpi4py import MPI  # type: ignore # noqa: F401

    hostname = None
    if comm.Get_rank() == 0:
        # Prefer a routable hostname; fall back to `socket.gethostname()`.
        import socket

        hostname = socket.gethostname()
    master = comm.bcast(hostname, root=0)
    os.environ["MASTER_ADDR"] = str(master)


def _bootstrap_dist_env() -> None:
    """Ensure ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` / ``MASTER_ADDR`` /
    ``MASTER_PORT`` are all set before ``init_process_group`` is called.

    Priority for rank / world_size:
      1. Existing torchrun-style env vars (already set)
      2. ``mpi4py.COMM_WORLD`` (authoritative under ``mpiexec`` / ezpz)
      3. Individual MPI-style env vars (``PMI_*``, ``PALS_*``, ``OMPI_*``)
      4. Single-process fallback (RANK=0, WORLD_SIZE=1)

    We reach for ``mpi4py`` when torchrun hasn't populated the env because
    Aurora / PALS does not always export a ``WORLD_SIZE``-equivalent variable
    (only per-rank ids), whereas ``mpi4py`` can always ask MPI directly.
    """
    torchrun_ok = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    rank_val: str | None = None
    size_val: str | None = None
    mpi_comm = None

    if not torchrun_ok:
        # Try mpi4py -- authoritative source under any MPI launcher.
        try:
            from mpi4py import MPI  # type: ignore

            mpi_comm = MPI.COMM_WORLD
            rank_val = str(mpi_comm.Get_rank())
            size_val = str(mpi_comm.Get_size())
        except ImportError:
            pass

        # Fall back to launcher-set env vars.
        if rank_val is None:
            rank_val = _first_env(_RANK_ENV_KEYS)
        if size_val is None:
            size_val = _first_env(_SIZE_ENV_KEYS)

        # Final fallback: single-process run.
        if rank_val is None or size_val is None:
            rank_val = "0"
            size_val = "1"

        os.environ["RANK"] = rank_val
        os.environ["WORLD_SIZE"] = size_val

    # LOCAL_RANK: honor any explicit setting, else consult launcher vars,
    # else default to 0 (correct for single-process).
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = _first_env(_LOCAL_RANK_ENV_KEYS) or "0"

    if "MASTER_ADDR" not in os.environ:
        _set_master_addr_via_mpi(mpi_comm)
    os.environ.setdefault("MASTER_PORT", "29500")

    _bootstrap_ccl_env(mpi_comm)


def _bootstrap_ccl_env(mpi_comm=None) -> None:
    """Populate CCL-facing env vars so oneCCL doesn't fall back to ATL probing.

    On Aurora, if ``CCL_LOCAL_RANK`` / ``CCL_LOCAL_SIZE`` are unset, oneCCL
    prints ``could not get local_idx/count from environment variables,
    trying to get them from ATL`` and can then hang in the ATL probe,
    especially when the FI provider is misconfigured. Setting them
    explicitly avoids the fallback entirely.
    """
    local_rank = os.environ["LOCAL_RANK"]
    os.environ.setdefault("CCL_LOCAL_RANK", local_rank)
    os.environ.setdefault("CCL_LOCAL_IDX", local_rank)

    # Node-local process count: prefer explicit launcher vars, then compute
    # via mpi4py's shared-memory split.
    local_size = (
        os.environ.get("PALS_LOCAL_SIZE")
        or os.environ.get("MPI_LOCALNRANKS")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE")
    )
    if local_size is None and mpi_comm is not None:
        try:
            from mpi4py import MPI  # type: ignore

            node_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)
            local_size = str(node_comm.Get_size())
        except Exception:
            pass
    if local_size:
        os.environ.setdefault("CCL_LOCAL_SIZE", local_size)
        os.environ.setdefault("CCL_LOCAL_COUNT", local_size)


def _pick_xpu_backend() -> str:
    """Pick a distributed backend that can run on Intel XPU.

    Priority:
      1. ``TORCH_XPU_BACKEND`` env override (advanced users).
      2. ``xccl`` if PyTorch reports it as available (native support in
         recent nightly XPU builds).
      3. ``ccl`` if ``oneccl_bindings_for_pytorch`` can be imported (this
         registers the backend as a side effect).
    Raises with an actionable message if none work.
    """
    override = os.environ.get("TORCH_XPU_BACKEND")
    if override:
        return override

    # xccl (native) -- available in recent nightly XPU builds.
    try:
        from torch.distributed import is_backend_available  # type: ignore

        if is_backend_available("xccl"):
            return "xccl"
    except (ImportError, AttributeError):
        pass

    # ccl (via oneCCL bindings) -- importing the module registers the backend.
    try:
        import oneccl_bindings_for_pytorch  # type: ignore  # noqa: F401

        return "ccl"
    except ImportError:
        pass

    raise RuntimeError(
        "XPU device is available but no distributed backend is registered.\n"
        "Fix by ONE of:\n"
        "  * Use a PyTorch nightly XPU build with native `xccl` support, or\n"
        "  * pip install oneccl_bind_pt --extra-index-url "
        "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/\n"
        "Or set TORCH_XPU_BACKEND=<backend_name> to force a choice."
    )


def _pick_backend_and_device() -> tuple[str, torch.device]:
    """Pick the right dist backend and per-rank device.

    Order: XPU (Aurora / Intel) first, then CUDA (NVIDIA), then CPU. XPU is
    checked first so that on Aurora we pick an XPU backend even in the rare
    case where a stub CUDA install reports as available.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(local_rank)
        backend = _pick_xpu_backend()
        return backend, torch.device(f"xpu:{local_rank}")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return "nccl", torch.device(f"cuda:{local_rank}")

    return "gloo", torch.device("cpu")


def _setup_distributed() -> tuple[torch.device, int, int]:
    _bootstrap_dist_env()
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
    # `use_underscores=True` keeps CLI flags matching the dataclass field names
    # (e.g. `--num_epochs` instead of tyro's default `--num-epochs`).
    cfg = tyro.cli(GemmaSFTConfig, use_underscores=True)
    try:
        main(cfg)
    except Exception:
        # Ensure non-zero exit propagates through torchrun cleanly.
        import traceback

        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Instruction fine-tune google/gemma-7b via HF Transformers + FSDP2.
#
# The launcher auto-detects the device count (XPU on Aurora, else CUDA) and
# picks a launcher:
#   * If `ezpz` is on PATH AND a PBS/SLURM job env is present -> `ezpz launch`
#     (correct for multi-node Aurora / ALCF).
#   * Otherwise -> `torchrun --nproc_per_node=<detected>` (single-node).
#
# Overrides:
#   NGPU        Force the per-node process count.
#   LAUNCHER    "ezpz" | "torchrun" | "auto" (default "auto").
#   LOG_RANK    Ranks whose stdout to tee to the terminal (default "0").
#   MODEL_PATH  HF model path or hub id (default: ./assets/hf/gemma-7b[/main]).
#
# Examples:
#   ./torchtitan/models/gemma/run_sft.sh \
#       --dataset_name AI-MO/NuminaMath-CoT \
#       --instruction_key problem --output_key solution \
#       --output_dir outputs/gemma-7b-numina
#
#   NGPU=12 LAUNCHER=torchrun ./torchtitan/models/gemma/run_sft.sh ...
set -eo pipefail

LOG_RANK=${LOG_RANK:-0}
LAUNCHER=${LAUNCHER:-auto}

# ---------------------------------------------------------------------------
# Auto-detect device count (XPU on Aurora, else CUDA)
# ---------------------------------------------------------------------------
_detect_ngpu() {
    python3 - <<'PY' 2>/dev/null
try:
    import torch
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print(torch.xpu.device_count())
    elif torch.cuda.is_available():
        print(torch.cuda.device_count())
    else:
        print(0)
except Exception:
    print(0)
PY
}

if [[ -z "${NGPU:-}" ]]; then
    NGPU="$(_detect_ngpu)"
fi
if [[ -z "${NGPU}" || "${NGPU}" -le 0 ]]; then
    echo "[run_sft.sh] No XPU/CUDA devices detected. Set NGPU=<n> manually." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Choose launcher
# ---------------------------------------------------------------------------
_have_ezpz() { command -v ezpz >/dev/null 2>&1; }
_in_job() { [[ -n "${PBS_NODEFILE:-}" || -n "${SLURM_JOB_ID:-}" ]]; }

if [[ "${LAUNCHER}" == "auto" ]]; then
    if _have_ezpz && _in_job; then
        LAUNCHER=ezpz
    else
        LAUNCHER=torchrun
    fi
fi

# ---------------------------------------------------------------------------
# Default model path (prefer locally-downloaded assets)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH="./assets/hf/gemma-7b"
if [[ -d "${DEFAULT_MODEL_PATH}/main" ]]; then
    DEFAULT_MODEL_PATH="${DEFAULT_MODEL_PATH}/main"
fi
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"

echo "[run_sft.sh] NGPU=${NGPU}  LAUNCHER=${LAUNCHER}  MODEL_PATH=${MODEL_PATH}"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if [[ "${LAUNCHER}" == "ezpz" ]]; then
    # ezpz reads PBS_NODEFILE / SLURM env and drives mpiexec across all nodes.
    # NGPU is exported for consumers that read it (informational only here --
    # ezpz decides ranks per node from the job allocation).
    export NGPU
    ezpz launch python3 -m torchtitan.models.gemma.train \
        --model_name_or_path "${MODEL_PATH}" \
        "$@"
else
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    torchrun \
        --nproc_per_node="${NGPU}" \
        --rdzv_backend c10d \
        --rdzv_endpoint="localhost:0" \
        --local-ranks-filter "${LOG_RANK}" \
        --role rank --tee 3 \
        -m torchtitan.models.gemma.train \
        --model_name_or_path "${MODEL_PATH}" \
        "$@"
fi

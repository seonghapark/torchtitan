#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Instruction fine-tune google/gemma-7b via HF Transformers + FSDP2.
#
# Example:
#   NGPU=8 ./torchtitan/models/gemma/run_sft.sh \
#       --dataset_name tatsu-lab/alpaca \
#       --output_dir outputs/gemma-7b-alpaca
set -eo pipefail

NGPU=${NGPU:-8}
LOG_RANK=${LOG_RANK:-0}

# Default to the locally downloaded weights so we skip HF auth.
DEFAULT_MODEL_PATH="./assets/hf/gemma-7b"
if [[ -d "${DEFAULT_MODEL_PATH}/main" ]]; then
    DEFAULT_MODEL_PATH="${DEFAULT_MODEL_PATH}/main"
fi

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter "${LOG_RANK}" \
    --role rank --tee 3 \
    -m torchtitan.models.gemma.train \
    --model_name_or_path "${MODEL_PATH:-${DEFAULT_MODEL_PATH}}" \
    "$@"

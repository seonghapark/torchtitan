module load python
alias uvi='uv pip install --no-cache --link-mode=copy'

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov"

uv venv \
    --system-site-packages \
    --relocatable \
    --no-cache \
    --link-mode=copy \
    --python=$(which python3)
source .venv/bin/activate

gh repo clone saforem2/torchtitan -- --branch ezpz
cd torchtitan

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job && ezpz_load_modules

# Required for per-node XPU power / util sampling (see _ResourceMonitor in
# torchtitan/models/gemma/train.py). Without this, `xpu-smi` is not on PATH
# on the compute nodes and the resource monitor falls back to torch memory
# stats only. Version pinned to the current default; drop the "/1.2.42" to
# always take the latest.
module load xpu-smi/1.2.42
# Weights & Biases: for logging train/val loss+accuracy and XPU util/power.
pip install --no-cache wandb



###########

torch==2.13.0.dev20260531+xpu
torchvision==0.28.0.dev20260601+xpu
torchaudio==2.11.0.dev20260531+xpu
torchdata
--index-url https://download.pytorch.org/whl/nightly/xpu


pip install --no-cache --pre \
  torch==2.13.0.dev20260530+xpu \
  torchvision==0.28.0.dev20260519+xpu \
  torchaudio==2.11.0.dev20260519+xpu \
  torchdata==0.12.0.dev20250220 \
  --index-url https://download.pytorch.org/whl/nightly/xpu
pip index versions torchdata --no-cache --pre \
  --index-url https://download.pytorch.org/whl/nightly/xpu
###########


uvi spmd_types torchcomms tyro tensorboard deepspeed mpi4py
uvi "git+https://github.com/zhenghh04/blendcorpus"
uvi "git+https://github.com/saforem2/ezpz"

uv pip uninstall impi-rt
## then download model from hf, using the download_all_models.sh

export HF_HOME=/lus/flare/projects/datascience/seonghapark/torchtitan/datasets
export HF_DATASETS_CACHE="${HF_HOME}/.cache"

# ---------------------------------------------------------------------------
# Weights & Biases config
# ---------------------------------------------------------------------------
# Login once on the login node before submitting:  wandb login <token>
# OR export WANDB_API_KEY here (do NOT commit the key to git).
# On offline compute nodes, set WANDB_MODE=offline and `wandb sync` later.
export WANDB_PROJECT="torchtitan-gemma-sft"
export WANDB_DIR="/lus/flare/projects/datascience/seonghapark/torchtitan/outputs/wandb"
export WANDB_CACHE_DIR="${WANDB_DIR}/.cache"
export WANDB_CONFIG_DIR="${WANDB_DIR}/.config"
mkdir -p "${WANDB_DIR}"
# Uncomment if compute nodes lack outbound HTTPS to api.wandb.ai:
# export WANDB_MODE=offline
# export WANDB_API_KEY="<your-key-here>"   # or use `wandb login` beforehand

export start=$(date -d "5 hours ago" '+%Y-%m-%d %H:%M:%S')

./torchtitan/models/gemma/run_sft.sh \
    --dataset_name AI-MO/NuminaMath-CoT \
    --dataset_split train \
    --instruction_key problem \
    --output_key solution \
    --output_dir outputs/gemma-7b-numina \
    --num_epochs 10 \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --lr 1e-5 \
    --warmup_ratio 0.03 \
    --max_seq_len 2048 \
    --save_interval 2000 \
    --log_interval 20 \
    --enable_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name gemma7b-numina-bs1-seq2048 \
    --wandb_mode "${WANDB_MODE:-online}" \
    --validation_split 0.02 \
    --eval_interval 200 \
    --eval_max_batches 50 \
    --resource_log_interval_sec 30

## tyro가 bool 옵션을 처리하는 방식 때문입니다.
## bool 타입 필드가 기본값 True일 때, 이를 False로 만들려면
## --no_<필드명> 이라는 별도 flag를 씁니다.
## --activation_checkpoint False처럼 값으로 전달하는 방식은 지원 안 함 → False가 unrecognized argument로 잡힌 것.

./torchtitan/models/gemma/run_sft.sh \
    --dataset_name AI-MO/NuminaMath-CoT \
    --dataset_split train \
    --instruction_key problem \
    --output_key solution \
    --output_dir outputs/gemma-7b-numina \
    --num_epochs 1 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr 1e-5 \
    --warmup_ratio 0.03 \
    --max_seq_len 1024 \
    --no_activation_checkpoint \
    --save_interval 2000 \
    --log_interval 20 \
    --enable_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name gemma7b-numina-bs2-seq1024 \
    --wandb_mode "${WANDB_MODE:-online}" \
    --validation_split 0.02 \
    --eval_interval 200 \
    --eval_max_batches 50 \
    --resource_log_interval_sec 30



MODULE=ezpz.agpt
CONFIG=agpt_2b
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module="${MODULE}" \
    --config="${CONFIG}" \
    --training.steps=10 \
    --checkpoint.no-enable \
    --training.local-batch-size=2

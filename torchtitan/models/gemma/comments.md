HF Transformers `GemmaForCausalLM`을 직접 사용하는 자체 완결형 SFT 파이프라인을 작성했습니다.

## 최종 파일 구성

| 파일 | 역할 |
| ---- | ---- |
| config.py | `GemmaSFTConfig` — 모델/데이터/학습/FSDP 옵션 dataclass |
| model.py | `GemmaModel` — `GemmaForCausalLM` 래퍼 (SDPA, gradient checkpointing) |
| tokenizer.py | `build_tokenizer` — PAD/EOS 설정 + Gemma chat template 자동 주입 |
| sftdataset.py | Alpaca 형식 `SFTDataset` + collator, 응답 부분만 loss 계산 (`IGNORE_INDEX=-100`) |
| parallelize.py | `parallelize_gemma` — FSDP2 + `MixedPrecisionPolicy` (bf16 param / fp32 reduce) |
| train.py | 메인 `torchrun` 엔트리 (dist init, AdamW, warmup+linear decay, grad clip, HF 포맷 저장) |
| run_sft.sh | 편의 런처 (기본적으로 `./assets/hf/gemma-7b/main` 로컬 가중치 사용) |
| README.md | 사용법 |
| __init__.py | 패키지 exports |

## 실행 방법

기본 실행 (Alpaca):
```bash
NGPU=8 ./torchtitan/models/gemma/run_sft.sh \
    --dataset_name tatsu-lab/alpaca \
    --output_dir outputs/gemma-7b-alpaca
```

로컬 JSONL:
```bash
NGPU=8 ./torchtitan/models/gemma/run_sft.sh \
    --dataset_local_path ./my_sft.jsonl \
    --output_dir outputs/gemma-7b-custom \
    --num_epochs 3 --lr 2e-5 --per_device_batch_size 1 \
    --gradient_accumulation_steps 8
```

모든 `GemmaSFTConfig` 필드는 `tyro`를 통해 자동으로 CLI 플래그로 노출됩니다. 저장 결과물은 표준 HF 디렉터리(`config.json`, `model.safetensors*`, tokenizer 파일)라서 이후 `GemmaForCausalLM.from_pretrained(save_dir)`로 바로 로딩 가능합니다.

## 주요 설계 포인트

- **응답 전용 loss**: chat template로 user turn까지의 토큰을 렌더링한 뒤 길이를 재서, 해당 prefix 토큰의 label을 `-100`으로 마스킹합니다.
- **Chat template**: `google/gemma-7b`(base)에는 chat template이 없어 gemma-7b-it과 동일한 템플릿을 tokenizer에 자동으로 심어 downstream 호환성을 유지합니다.
- **FSDP2**: 각 `GemmaDecoderLayer` → HF 모델 → 래퍼 순서로 `fully_shard`. bf16 param / fp32 grad reduce.
- **디바이스**: CUDA/XPU 자동 감지 (NCCL/CCL 백엔드 자동 선택) — Aurora XPU에서도 그대로 동작.
- **체크포인트 저장**: FSDP2의 `get_model_state_dict(full_state_dict=True, cpu_offload=True)`로 rank 0에 전체 가중치를 모아 `model.` prefix를 벗겨 HF 표준 포맷으로 저장.

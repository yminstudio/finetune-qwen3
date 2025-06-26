"""
QLora 파인튜닝 → 서비스 배포 프로젝트 설정
"""
import os
from pathlib import Path

# 모델 설정
MODEL_BASE_NAME = "Qwen3-4B"
MODEL_NAME = f"Qwen/{MODEL_BASE_NAME}"

# 모델별 메모리 요구사항 (GB 단위)
MODEL_MEMORY_REQUIREMENTS = {
    "Qwen3-4B": {
        "training_ram": 16,      # 학습 시 최소 RAM
        "merge_ram": 12,         # 병합 시 최소 RAM  
        "inference_ram": 8,      # 추론 시 최소 RAM
        "disk_space": 15,        # 필요 디스크 공간
        "gpu_vram": 12          # 필요 GPU VRAM
    },
    "Qwen3-8B": {
        "training_ram": 32,
        "merge_ram": 24,
        "inference_ram": 16,
        "disk_space": 30,
        "gpu_vram": 24
    },
    "Qwen3-14B": {
        "training_ram": 48,
        "merge_ram": 36,
        "inference_ram": 24,
        "disk_space": 50,
        "gpu_vram": 36
    }
}

def get_memory_requirements():
    """현재 모델의 메모리 요구사항 반환"""
    return MODEL_MEMORY_REQUIREMENTS.get(MODEL_BASE_NAME, MODEL_MEMORY_REQUIREMENTS["Qwen3-4B"])

# 프로젝트 기본 경로
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / f"outputs/{MODEL_BASE_NAME}/"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# 필요한 디렉토리 생성
for dir_path in [CACHE_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# QLora 설정
QLORA_CONFIG = {
    "r": 16,                    # LoRA rank
    "lora_alpha": 16,          # LoRA alpha parameter
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.1,       # LoRA dropout
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# 학습 설정
TRAINING_CONFIG = {
    "output_dir": str(OUTPUT_DIR / "qlora_checkpoints"),
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "optim": "adamw_8bit",
    "save_steps": 50,
    "logging_steps": 10,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "fp16": False,
    "bf16": True,
    "max_grad_norm": 0.3,
    "max_steps": 100,
    "warmup_ratio": 0.03,
    "group_by_length": True,
    "lr_scheduler_type": "linear",
    "report_to": [],  # wandb 비활성화 (빈 리스트)
    "eval_strategy": "no"
}

# 4-bit 양자화 설정
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": False,
}

# 데이터 설정
MAX_SEQ_LENGTH = 2048
CUTOFF_LEN = 512

# GPU 설정
GPU_CONFIG = {
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "device_map": "auto"
}

# GGUF 변환 설정
GGUF_CONFIG = {
    "quantization_type": "q4_k_m",  # 권장 옵션: 속도와 품질 균형
    "context_length": 4096,
    "output_filename": f"{MODEL_BASE_NAME.lower()}-finetune-q4_k_m.gguf"
}

# Ollama 설정
OLLAMA_CONFIG = {
    "model_name": f"{MODEL_BASE_NAME.lower()}-finetune",  # 간단한 형식으로 변경
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "stop_tokens": ["<|im_end|>", "</s>"],
    "num_ctx": 4096,
    "num_keep": 24
}

# API 서버 설정
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1,
    "timeout": 300
}

# 로깅 설정
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": str(LOGS_DIR / "pipeline.log")
}

# 성능 모니터링 설정
MONITORING_CONFIG = {
    "log_memory_usage": True,
    "log_gpu_usage": True,
    "benchmark_inference": True,
    "save_metrics": True
}

LLAMA_CPP_DIR = Path("/root/llama.cpp")

print(f"✅ 설정 로드 완료")
print(f"📁 프로젝트 루트: {PROJECT_ROOT}")
print(f"🤖 모델: {MODEL_NAME}")
print(f"💾 출력 디렉토리: {OUTPUT_DIR}") 
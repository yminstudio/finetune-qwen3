#!/usr/bin/env python3
"""
🔧 2단계: QLora 파인튜닝 실행 (2-3시간)

목적:
- 한국어 기술 QA 데이터로 모델 특화
- LoRA 어댑터 생성 및 학습
- 학습 과정 모니터링 및 최적화
"""

import os
import sys
import time
import json
import torch
import gc
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import psutil
import GPUtil
from config import PROJECT_ROOT

# 설정 파일 import
from config import *

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'step2_train_qlora.log')
os.makedirs(LOG_DIR, exist_ok=True)
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        print(*args, **kwargs, file=f)

def monitor_resources():
    """시스템 리소스 모니터링"""
    # GPU 메모리
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_cached = torch.cuda.memory_reserved() / (1024**3)
        
        # GPUtil로 GPU 사용률 확인
        gpus = GPUtil.getGPUs()
        gpu_usage = (gpus[0].memoryUtil * 100) if gpus else 0
        
        log_print(f"   GPU: {gpu_memory:.1f}GB 할당, {gpu_cached:.1f}GB 캐시, 사용률: {gpu_usage:.1f}%")
    
    # 시스템 RAM
    memory = psutil.virtual_memory()
    log_print(f"   RAM: {memory.used/(1024**3):.1f}GB / {memory.total/(1024**3):.1f}GB ({memory.percent:.1f}%)")

def load_and_prepare_data():
    """학습 데이터 로드 및 전처리"""
    log_print("📊 [1/6] 학습 데이터 준비 시작...")
    start_time = time.time()
    
    # 절대 경로로 변경
    data_path = PROJECT_ROOT / "data" / "korean_tech_qa.json"
    if not data_path.exists():
        log_print("❌ 학습 데이터를 찾을 수 없습니다.")
        log_print(f"   파일 경로: {data_path}")
        log_print("   다음 명령으로 데이터를 생성하세요: python create_sample_data.py")
        return None
    
    # JSON 데이터 로드
    log_print("   📁 JSON 파일 로드 중...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    log_print(f"   ✅ 로드된 샘플 수: {len(data)}개")
    
    # ChatML 형식을 Qwen3 chat template로 변환하는 함수
    def format_chat_template(example):
        messages = example['messages']
        return {"messages": messages}
    
    # Dataset 객체 생성
    log_print("   🔄 데이터셋 포맷팅 중...")
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_chat_template)
    
    elapsed_time = time.time() - start_time
    log_print(f"   ✅ 데이터 준비 완료! (소요시간: {elapsed_time:.1f}초)")
    log_print(f"   📋 데이터셋 크기: {len(dataset)}")
    log_print(f"   📋 첫 번째 샘플 메시지 수: {len(dataset[0]['messages'])}")
    
    return dataset

def setup_model_and_tokenizer():
    """모델과 토크나이저 설정"""
    log_print(f"\n🤖 [2/6] 모델 로드 시작: {MODEL_NAME}")
    start_time = time.time()
    
    # BitsAndBytesConfig 설정
    log_print("   ⚙️ 4-bit 양자화 설정 중...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    
    # 토크나이저 로드
    log_print("   📝 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    log_print("   ✅ 토크나이저 로드 완료")
    
    # 모델 로드
    log_print("   🔄 베이스 모델 로드 중 (시간이 걸릴 수 있습니다)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model_load_time = time.time() - start_time
    log_print(f"   ✅ 베이스 모델 로드 완료! (소요시간: {model_load_time:.1f}초)")
    log_print(f"   📊 모델 타입: {type(model)}")
    log_print(f"   📊 최대 시퀀스 길이: {MAX_SEQ_LENGTH}")
    
    # 현재 메모리 사용량 출력
    monitor_resources()
    
    # LoRA 설정
    log_print("\n🔧 [3/6] LoRA 어댑터 설정 시작...")
    lora_start_time = time.time()
    
    lora_config = LoraConfig(
        r=QLORA_CONFIG["r"],
        lora_alpha=QLORA_CONFIG["lora_alpha"],
        target_modules=QLORA_CONFIG["target_modules"],
        lora_dropout=QLORA_CONFIG["lora_dropout"],
        bias=QLORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    log_print(f"   ⚙️ LoRA 설정: rank={QLORA_CONFIG['r']}, alpha={QLORA_CONFIG['lora_alpha']}")
    log_print(f"   🎯 Target modules: {QLORA_CONFIG['target_modules']}")
    
    # PEFT 모델로 변환
    log_print("   🔄 PEFT 모델 변환 중...")
    model = get_peft_model(model, lora_config)
    
    # 학습 가능한 파라미터 수 출력
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable_params / total_params
    
    lora_time = time.time() - lora_start_time
    total_time = time.time() - start_time
    
    log_print(f"   ✅ LoRA 어댑터 설정 완료! (소요시간: {lora_time:.1f}초)")
    log_print(f"   📊 학습 가능한 파라미터: {trainable_params:,} ({percentage:.3f}%)")
    log_print(f"   📊 전체 파라미터: {total_params:,}")
    log_print(f"   ⏱️ 총 모델 준비 시간: {total_time:.1f}초")
    
    # LoRA 설정 후 메모리 사용량
    monitor_resources()
    
    return model, tokenizer

def setup_trainer(model, tokenizer, dataset):
    """SFTTrainer 설정"""
    log_print("\n🏋️ [4/6] 트레이너 설정 시작...")
    start_time = time.time()
    
    # 학습 인자 설정
    log_print("   ⚙️ 학습 매개변수 설정 중...")
    training_args = TrainingArguments(**TRAINING_CONFIG)
    
     # chat template 적용 함수 (단일 예제 처리)
    def formatting_prompts_func(example):
        # example은 단일 딕셔너리 {"messages": [...]}
        messages = example["messages"]
        
        # 직접 텍스트 포맷팅 (chat template 대신)
        formatted_text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        return {"text": formatted_text}
    
    log_print("   🔄 SFTTrainer 초기화 중...")
    # SFTTrainer 생성
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )
    
    elapsed_time = time.time() - start_time
    log_print(f"   ✅ 트레이너 설정 완료! (소요시간: {elapsed_time:.1f}초)")
    log_print(f"   📊 배치 크기: {training_args.per_device_train_batch_size}")
    log_print(f"   📊 gradient accumulation: {training_args.gradient_accumulation_steps}")
    log_print(f"   📊 실효 배치 크기: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    log_print(f"   📊 최대 스텝: {training_args.max_steps}")
    log_print(f"   📊 학습률: {training_args.learning_rate}")
    
    return trainer

def train_model(trainer):
    """모델 학습 실행"""
    log_print("\n🚀 [5/6] 학습 시작!")
    log_print("=" * 60)
    
    start_time = time.time()
    
    # 초기 리소스 상태
    log_print("📊 학습 시작 전 리소스 상태:")
    monitor_resources()
    log_print()
    
    log_print("🔥 QLora 파인튜닝 실행 중...")
    log_print("   - 예상 소요시간: 20-30분")
    log_print("   - 100 스텝 학습 진행")
    log_print("   - 50스텝, 100스텝에서 체크포인트 저장")
    log_print()
    
    # 학습 실행
    try:
        trainer.train()
    except Exception as e:
        log_print(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    elapsed_time = time.time() - start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    
    log_print("=" * 60)
    log_print(f"✅ 학습 완료! 소요 시간: {int(hours)}시간 {int(minutes)}분")
    
    # 최종 리소스 상태
    log_print("\n📊 학습 완료 후 리소스 상태:")
    monitor_resources()
    
    return True

def save_model_and_adapters(model, tokenizer):
    """모델과 어댑터 저장"""
    log_print("\n💾 [6/6] 모델 저장 시작...")
    start_time = time.time()
    
    # LoRA 어댑터만 저장
    lora_output_dir = OUTPUT_DIR / "qwen3_lora_adapters"
    lora_output_dir.mkdir(exist_ok=True)
    
    log_print("   📁 LoRA 어댑터 저장 중...")
    model.save_pretrained(str(lora_output_dir))
    
    log_print("   📝 토크나이저 저장 중...")
    tokenizer.save_pretrained(str(lora_output_dir))
    
    save_time = time.time() - start_time
    log_print(f"   ✅ LoRA 어댑터 저장 완료: {lora_output_dir} (소요시간: {save_time:.1f}초)")
    
    # 저장된 파일 크기 확인
    adapter_file = lora_output_dir / "adapter_model.safetensors"
    if adapter_file.exists():
        size_gb = adapter_file.stat().st_size / (1024**3)
        log_print(f"   📊 어댑터 파일 크기: {size_gb:.2f}GB")
    
    # 학습 결과 요약 저장
    log_print("   📋 학습 요약 저장 중...")
    summary = {
        "model_name": MODEL_NAME,
        "lora_config": QLORA_CONFIG,
        "training_config": TRAINING_CONFIG,
        "output_dir": str(lora_output_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_file = lora_output_dir / "training_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    total_save_time = time.time() - start_time
    log_print(f"   ✅ 학습 요약 저장 완료: {summary_file} (총 저장시간: {total_save_time:.1f}초)")
    
    return lora_output_dir

def cleanup_memory():
    """메모리 정리"""
    log_print("🧹 메모리 정리 중...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    log_print("   메모리 정리 완료")

def main():
    """메인 실행 함수"""
    overall_start_time = time.time()
    
    log_print("🔧 2단계: QLora 파인튜닝 시작")
    log_print("=" * 60)
    log_print("🎯 목표: Qwen3 모델을 한국어 기술 QA로 파인튜닝")
    log_print("⏱️ 예상 총 소요시간: 30-40분")
    log_print("=" * 60)
    
    try:
        # 1. 데이터 준비
        dataset = load_and_prepare_data()
        if dataset is None:
            log_print("❌ 1단계 실패: 데이터 준비")
            return False
        
        # 2. 모델과 토크나이저 설정
        model, tokenizer = setup_model_and_tokenizer()
        
        # 3. 트레이너 설정
        trainer = setup_trainer(model, tokenizer, dataset)
        
        # 4. 학습 실행
        if not train_model(trainer):
            log_print("❌ 5단계 실패: 학습 실행")
            return False
        
        # 5. 모델 저장
        output_dir = save_model_and_adapters(model, tokenizer)
        
        # 6. 메모리 정리
        cleanup_memory()
        
        # 전체 완료 시간 계산
        total_elapsed = time.time() - overall_start_time
        hours = total_elapsed // 3600
        minutes = (total_elapsed % 3600) // 60
        
        log_print("\n" + "=" * 60)
        log_print("🎉 2단계 QLoRA 파인튜닝 완전 완료!")
        log_print("=" * 60)
        log_print(f"⏱️ 총 소요시간: {int(hours)}시간 {int(minutes)}분")
        log_print(f"📁 LoRA 어댑터 위치: {output_dir}")
        log_print(f"📊 어댑터 크기: 약 3GB")
        log_print("=" * 60)
        log_print("📋 다음 단계 안내:")
        log_print("   ✅ step3 실행: python step3_validate_results.py")
        log_print("   🔍 학습 효과 검증 및 추론 속도 측정")
        log_print("=" * 60)
        
        return True
        
    except Exception as e:
        elapsed = time.time() - overall_start_time
        log_print(f"\n❌ 오류 발생 (실행시간: {elapsed:.1f}초)")
        log_print(f"🔍 오류 내용: {e}")
        log_print("=" * 60)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
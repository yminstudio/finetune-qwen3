#!/usr/bin/env python3
"""
🔄 4단계: LoRA 어댑터 병합 (1시간)

목적:
- 4-bit Base + 16-bit LoRA → 16-bit 통합 모델 생성
- 추론 최적화를 위한 단일 모델 구조 확보
- 호환성 문제 해결
"""

import os
import sys
import time
import json
import torch
import gc
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
import psutil

# 설정 파일 import
from config import *

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'step4_merge_adapters.log')
os.makedirs(LOG_DIR, exist_ok=True)
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        print(*args, **kwargs, file=f)

def check_system_requirements():
    """시스템 요구사항 확인"""
    log_print("💾 시스템 요구사항 확인 중...")
    
    # 현재 모델의 메모리 요구사항 가져오기
    memory_req = get_memory_requirements()
    required_ram = memory_req["merge_ram"]
    required_disk = memory_req["disk_space"]
    
    # RAM 확인
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    log_print(f"   사용 가능한 RAM: {available_gb:.1f}GB")
    log_print(f"   필요한 RAM: {required_ram}GB ({MODEL_BASE_NAME} 모델)")
    
    # 모델별 메모리 요구사항 체크
    if available_gb < required_ram:
        log_print("   ⚠️ RAM 부족 경고:")
        log_print(f"   병합 과정에서 {required_ram}GB+ 메모리가 필요합니다.")
        log_print(f"   현재 사용 가능: {available_gb:.1f}GB")
        log_print("   ⚠️ 메모리 부족으로 인해 병합이 실패할 수 있습니다.")
        log_print("   💡 다른 프로세스를 종료하거나 더 작은 모델을 사용하세요.")
        # 자동으로 계속 진행 (경고만 표시)
    
    # 디스크 공간 확인 (모델별 요구사항)
    disk = psutil.disk_usage('/')
    free_gb = disk.free / (1024**3)
    
    log_print(f"   사용 가능한 디스크: {free_gb:.1f}GB")
    log_print(f"   필요한 디스크: {required_disk}GB ({MODEL_BASE_NAME} 모델)")
    
    if free_gb < required_disk:
        log_print("   ❌ 디스크 공간 부족:")
        log_print(f"   병합된 모델 저장에 {required_disk}GB+ 필요")
        log_print(f"   현재 사용 가능: {free_gb:.1f}GB")
        return False
    
    log_print("   ✅ 시스템 요구사항 확인 완료")
    return True

def load_base_model_16bit():
    """베이스 모델을 16-bit로 로드"""
    log_print(f"🤖 베이스 모델 16-bit 로드 중: {MODEL_NAME}")
    
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    start_time = time.time()
    
    try:
        # 16-bit로 베이스 모델 로드 (양자화 없음)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        log_print(f"   ✅ 베이스 모델 로드 완료 ({load_time:.1f}초)")
        
        # 모델 크기 정보
        total_params = sum(p.numel() for p in model.parameters())
        memory_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        
        log_print(f"   모델 파라미터: {total_params:,}")
        log_print(f"   메모리 사용량: {memory_gb:.1f}GB")
        
        return model, tokenizer
        
    except Exception as e:
        log_print(f"❌ 베이스 모델 로드 실패: {e}")
        return None, None

def load_lora_adapters():
    """LoRA 어댑터 로드"""
    log_print("🔧 LoRA 어댑터 로드 중...")
    
    lora_adapter_path = OUTPUT_DIR / "qwen3_lora_adapters"
    
    if not lora_adapter_path.exists():
        log_print("❌ LoRA 어댑터를 찾을 수 없습니다.")
        log_print("   먼저 2단계를 실행하세요: python3 step2_train_qlora.py")
        return None
    
    # 어댑터 파일 확인
    adapter_files = list(lora_adapter_path.glob("adapter_*.safetensors"))
    if not adapter_files:
        adapter_files = list(lora_adapter_path.glob("adapter_*.bin"))
    
    if not adapter_files:
        log_print("❌ 어댑터 파일을 찾을 수 없습니다.")
        return None
    
    adapter_file = adapter_files[0]
    adapter_size_gb = adapter_file.stat().st_size / (1024**3)
    
    log_print(f"   어댑터 파일: {adapter_file.name}")
    log_print(f"   어댑터 크기: {adapter_size_gb:.2f}GB")
    
    return lora_adapter_path

def merge_lora_with_base(model, tokenizer, lora_adapter_path):
    """LoRA 어댑터를 베이스 모델에 병합"""
    log_print("🔄 LoRA 어댑터 병합 중...")
    
    start_time = time.time()
    
    try:
        from peft import PeftModel
        
        # PEFT 모델로 어댑터 로드
        log_print("   PEFT 모델 생성 중...")
        peft_model = PeftModel.from_pretrained(
            model,
            str(lora_adapter_path),
            torch_dtype=torch.bfloat16
        )
        
        log_print("   어댑터 병합 실행 중...")
        # 병합 실행 - 이 과정에서 많은 메모리 사용
        merged_model = peft_model.merge_and_unload()
        
        merge_time = time.time() - start_time
        log_print(f"   ✅ 병합 완료 ({merge_time:.1f}초)")
        
        # 메모리 정리
        del peft_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return merged_model
        
    except Exception as e:
        log_print(f"❌ 병합 실패: {e}")
        
        # 대안 방법 시도 (Unsloth 사용)
        log_print("   대안 방법 시도 중...")
        try:
            # Unsloth로 다시 로드 후 병합
            model_4bit, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )
            
            # LoRA 어댑터 추가
            model_4bit = FastLanguageModel.get_peft_model(
                model_4bit,
                **QLORA_CONFIG
            )
            
            # 어댑터 가중치 로드
            model_4bit.load_state_dict(
                torch.load(lora_adapter_path / "adapter_model.bin"), 
                strict=False
            )
            
            # 16-bit로 병합
            merged_model = model_4bit.merge_and_unload()
            
            log_print("   ✅ 대안 방법으로 병합 완료")
            return merged_model
            
        except Exception as e2:
            log_print(f"❌ 대안 방법도 실패: {e2}")
            return None

def verify_merged_model(merged_model, tokenizer):
    """병합된 모델 검증"""
    log_print("🧪 병합된 모델 검증 중...")
    
    # 간단한 텍스트 생성 테스트
    test_prompts = [
        "안녕하세요! 저는",
        "파이썬에서 리스트 컴프리헨션이란",
        "머신러닝의 기본 개념은"
    ]
    
    for i, prompt in enumerate(test_prompts):
        log_print(f"\n   테스트 {i+1}/3: {prompt}")
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)
            
            with torch.no_grad():
                outputs = merged_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            log_print(f"   응답: {response[len(prompt):].strip()[:100]}...")
            
        except Exception as e:
            log_print(f"   ❌ 테스트 실패: {e}")
            return False
    
    log_print("   ✅ 병합된 모델 검증 완료")
    return True

def save_merged_model(merged_model, tokenizer):
    """병합된 모델 저장"""
    log_print("💾 병합된 모델 저장 중...")
    
    # 저장 경로 설정
    merged_model_dir = OUTPUT_DIR / "qwen3_finetune_merged"
    merged_model_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    try:
        # 모델 저장
        log_print("   모델 가중치 저장 중...")
        merged_model.save_pretrained(
            str(merged_model_dir),
            safe_serialization=True,  # safetensors 형식 사용
            max_shard_size="5GB"      # 파일 크기 제한
        )
        
        # 토크나이저 저장
        log_print("   토크나이저 저장 중...")
        tokenizer.save_pretrained(str(merged_model_dir))
        
        save_time = time.time() - start_time
        log_print(f"   ✅ 저장 완료 ({save_time:.1f}초)")
        
        # 저장된 모델 크기 확인
        total_size = sum(f.stat().st_size for f in merged_model_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        
        log_print(f"   저장 위치: {merged_model_dir}")
        log_print(f"   모델 크기: {size_gb:.1f}GB")
        
        # 메타데이터 저장
        metadata = {
            "model_name": MODEL_NAME,
            "merged_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "original_lora_config": QLORA_CONFIG,
            "merged_model_size_gb": size_gb,
            "model_type": "merged_16bit",
            "description": "LoRA 어댑터가 베이스 모델에 병합된 16-bit 모델"
        }
        
        metadata_file = merged_model_dir / "merge_info.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        log_print(f"   메타데이터 저장: {metadata_file}")
        
        return merged_model_dir
        
    except Exception as e:
        log_print(f"❌ 저장 실패: {e}")
        return None

def cleanup_intermediate_files():
    """중간 파일 정리"""
    log_print("🧹 중간 파일 정리 중...")
    
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 시스템 메모리 정리
    gc.collect()
    
    log_print("   메모리 정리 완료")

def main():
    """메인 실행 함수"""
    log_print("🔄 4단계: LoRA 어댑터 병합 시작")
    log_print("=" * 50)
    
    try:
        # 1. 시스템 요구사항 확인
        if not check_system_requirements():
            log_print("❌ 시스템 요구사항 부족")
            return False
        
        # 2. LoRA 어댑터 확인
        lora_adapter_path = load_lora_adapters()
        if lora_adapter_path is None:
            return False
        
        # 3. 베이스 모델 16-bit 로드
        model, tokenizer = load_base_model_16bit()
        if model is None:
            return False
        
        # 4. LoRA 어댑터 병합
        merged_model = merge_lora_with_base(model, tokenizer, lora_adapter_path)
        if merged_model is None:
            return False
        
        # 5. 병합된 모델 검증
        if not verify_merged_model(merged_model, tokenizer):
            return False
        
        # 6. 병합된 모델 저장
        merged_model_dir = save_merged_model(merged_model, tokenizer)
        if merged_model_dir is None:
            return False
        
        # 7. 정리
        cleanup_intermediate_files()
        
        log_print("=" * 50)
        log_print("✅ 4단계 완료!")
        log_print("📋 결과:")
        log_print(f"   병합된 모델: {merged_model_dir}")
        log_print(f"   파인튜닝 효과: LoRA 어댑터가 베이스 모델에 완전히 병합됨")
        log_print(f"   모델 형식: 표준 16-bit transformers 모델")
        log_print("\n📋 다음 단계 안내:")
        log_print("   - 5단계 실행: python3 step5_convert_to_gguf.py")
        log_print("   - 병합된 모델을 GGUF로 변환하여 Ollama에서 사용")
        
        return True
        
    except Exception as e:
        log_print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
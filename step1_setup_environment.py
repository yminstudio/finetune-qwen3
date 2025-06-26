#!/usr/bin/env python3
"""
🎯 1단계: 모델 준비 및 환경 설정 (30분)

목적:
- 표준 transformers 환경 구축  
- Qwen3-8B 모델 다운로드 및 캐시
- GPU 메모리 및 환경 최적화
"""

import os
import sys
import time
import torch
import psutil
import GPUtil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
from config import PROJECT_ROOT

# 설정 파일 import
from config import *

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'step1_setup_environment.log')
os.makedirs(LOG_DIR, exist_ok=True)
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        print(*args, **kwargs, file=f)

def setup_cuda_environment():
    """CUDA 환경 최적화"""
    log_print("🔧 CUDA 환경 설정 중...")
    
    # GPU 메모리 할당 전략 설정
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # CUDA 가시성 설정 (필요시)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        log_print(f"✅ CUDA 사용 가능 - GPU 개수: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_props = torch.cuda.get_device_properties(i)
            memory_gb = gpu_props.total_memory / (1024**3)
            log_print(f"   GPU {i}: {gpu_props.name} ({memory_gb:.1f}GB)")
        
        # 현재 GPU 메모리 상태
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            memory_percent = gpu.memoryUtil * 100
            log_print(f"   GPU {gpu.id}: 메모리 사용률 {memory_percent:.1f}% ({gpu.memoryUsed}MB/{gpu.memoryTotal}MB)")
    else:
        log_print("❌ CUDA를 사용할 수 없습니다!")
        return False
    
    return True

def check_system_resources():
    """시스템 리소스 확인"""
    log_print("💾 시스템 리소스 확인 중...")
    
    # RAM 확인
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    log_print(f"   RAM: {memory.used/(1024**3):.1f}GB / {memory.total/(1024**3):.1f}GB ({memory.percent:.1f}%)")
    log_print(f"   Swap: {swap.used/(1024**3):.1f}GB / {swap.total/(1024**3):.1f}GB ({swap.percent:.1f}%)")
    
    # 디스크 공간 확인
    disk = psutil.disk_usage('/')
    log_print(f"   디스크: {disk.used/(1024**3):.1f}GB / {disk.total/(1024**3):.1f}GB ({(disk.used/disk.total)*100:.1f}%)")
    
    # 최소 요구사항 체크 (8B 모델 기준)
    warnings = []
    if memory.total < 16 * (1024**3):  # 16GB
        warnings.append(f"⚠️ RAM이 부족할 수 있습니다. 권장: 24GB+, 최소: 16GB, 현재: {memory.total/(1024**3):.1f}GB")
    
    if swap.total < 16 * (1024**3):  # 16GB
        warnings.append(f"⚠️ Swap 메모리가 부족할 수 있습니다. 권장: 16GB+, 현재: {swap.total/(1024**3):.1f}GB")
    
    if disk.free < 40 * (1024**3):  # 40GB
        warnings.append(f"⚠️ 디스크 공간이 부족할 수 있습니다. 최소: 40GB, 현재 여유: {disk.free/(1024**3):.1f}GB")
    
    for warning in warnings:
        log_print(warning)
    
    return len(warnings) == 0

def download_and_cache_model():
    """모델 다운로드 및 캐시"""
    log_print(f"🔄 모델 다운로드 시작: {MODEL_NAME}")
    start_time = time.time()
    
    try:
        # BitsAndBytesConfig 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        
        log_print("   토크나이저 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        log_print("   4-bit 양자화 모델 로드 중...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 기본 테스트
        log_print("   모델 테스트 중...")
        messages = [
            {"role": "user", "content": "안녕하세요! 간단한 인사말을 해주세요."}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 빠른 테스트를 위해 thinking 모드 비활성화
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        log_print(f"   테스트 출력: {response[:100]}...")
        
        # 메모리 정리
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        elapsed_time = time.time() - start_time
        log_print(f"✅ 모델 다운로드 완료 ({elapsed_time:.1f}초)")
        
        # 캐시 위치 정보
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Qwen--{MODEL_BASE_NAME}"
        if cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            log_print(f"   캐시 위치: {cache_dir}")
            log_print(f"   캐시 크기: {cache_size/(1024**3):.1f}GB")
        
        return True
        
    except Exception as e:
        log_print(f"❌ 모델 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_dependencies():
    """의존성 패키지 확인"""
    log_print("📦 의존성 패키지 확인 중...")
    
    required_packages = [
        ("torch", "2.0.0"),
        ("transformers", "4.51.0"), 
        ("peft", "0.12.0"),
        ("bitsandbytes", "0.43.0"),
        ("accelerate", "0.30.0")
    ]
    
    missing_packages = []
    
    for package_name, min_version in required_packages:
        try:
            pkg = __import__(package_name)
            version = getattr(pkg, '__version__', 'unknown')
            log_print(f"   ✅ {package_name}: {version}")
        except ImportError:
            log_print(f"   ❌ {package_name}: 설치되지 않음")
            missing_packages.append(package_name)
    
    if missing_packages:
        log_print(f"⚠️ 누락된 패키지: {', '.join(missing_packages)}")
        log_print("   다음 명령으로 설치하세요: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """메인 실행 함수"""
    log_print("🚀 1단계: 모델 준비 및 환경 설정 시작")
    log_print("=" * 50)
    
    # 1. 의존성 확인
    if not verify_dependencies():
        log_print("❌ 의존성 패키지 설치 후 다시 실행하세요.")
        return False
    
    # 2. CUDA 환경 설정
    if not setup_cuda_environment():
        log_print("❌ CUDA 환경 설정 실패")
        return False
    
    # 3. 시스템 리소스 확인
    check_system_resources()
    
    # 4. 모델 다운로드 및 테스트
    if not download_and_cache_model():
        log_print("❌ 모델 준비 실패")
        return False
    
    log_print("=" * 50)
    log_print("✅ 1단계 완료!")
    log_print("📋 다음 단계 안내:")
    log_print("   - 2단계 실행: python step2_train_qlora.py")
    log_print(f"   - 학습 데이터를 {PROJECT_ROOT / 'data'} 폴더에 준비해주세요")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
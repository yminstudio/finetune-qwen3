#!/usr/bin/env python3
"""
⚡ 5단계: GGUF 변환 및 Ollama 등록 (1-2시간)

목적:
- 16-bit 모델을 추론 최적화된 GGUF 포맷으로 변환
- Ollama 엔진에서 고속 추론 가능하도록 준비
- 메모리 효율성과 속도 최적화
"""

import os
import sys
import time
import json
import subprocess
import shutil
from pathlib import Path

# 설정 파일 import
from config import *
from config import LLAMA_CPP_DIR
from config import MODELS_DIR
from config import PROJECT_ROOT

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'step5_convert_to_gguf.log')
os.makedirs(LOG_DIR, exist_ok=True)
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        print(*args, **kwargs, file=f)

def check_prerequisites():
    """필수 도구 확인"""
    log_print("🔍 필수 도구 확인 중...")
    
    prerequisites = {
        "git": "git --version",
        "python3": "python3 --version",
        "cmake": "cmake --version",
        "make": "make --version"
    }
    
    missing_tools = []
    
    for tool, check_cmd in prerequisites.items():
        try:
            result = subprocess.run(check_cmd.split(), 
                                  capture_output=True, text=True, check=True)
            version_info = result.stdout.strip().split('\n')[0]  # 첫 번째 줄만
            log_print(f"   ✅ {tool}: {version_info.split()[-1] if version_info else 'OK'}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            log_print(f"   ❌ {tool}: 설치되지 않음")
            missing_tools.append(tool)
    
    if missing_tools:
        log_print(f"\n⚠️ 누락된 도구들을 설치하세요:")
        if "cmake" in missing_tools:
            log_print("   sudo apt-get install cmake")
        if "make" in missing_tools:
            log_print("   sudo apt-get install build-essential")
        if "git" in missing_tools:
            log_print("   sudo apt-get install git")
        return False
    
    return True

def setup_llama_cpp():
    """llama.cpp 환경 구축"""
    log_print("🛠️ llama.cpp 환경 구축 중...")
    
    llama_cpp_dir = LLAMA_CPP_DIR
    
    if llama_cpp_dir.exists():
        log_print("   llama.cpp 디렉토리가 이미 존재합니다.")
        # 빌드 파일들이 있는지 확인
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        quantize_binary = llama_cpp_dir / "llama-quantize"
        if convert_script.exists() and quantize_binary.exists():
            log_print("   ✅ 이미 빌드되어 있습니다.")
            return llama_cpp_dir
        else:
            log_print("   빌드 파일이 없어서 다시 빌드합니다.")
    if not llama_cpp_dir.exists():
        log_print("   llama.cpp 저장소 클론 중...")
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/ggerganov/llama.cpp.git",
                str(llama_cpp_dir)
            ], check=True, cwd=PROJECT_ROOT)
            log_print("   ✅ 클론 완료")
        except subprocess.CalledProcessError as e:
            log_print(f"   ❌ 클론 실패: {e}")
            return None
    # CMake를 사용한 빌드
    build_dir = llama_cpp_dir / "build"
    build_dir.mkdir(exist_ok=True)
    log_print("   CMake 설정 중...")
    try:
        # CUDA 지원 CMake 설정
        cmake_cmd = [
            "cmake", "..", 
            "-DGGML_CUDA=ON",
            "-DCMAKE_BUILD_TYPE=Release"
        ]
        result = subprocess.run(cmake_cmd, check=True, cwd=build_dir, 
                              capture_output=True, text=True)
        log_print("   ✅ CMake 설정 완료 (CUDA 지원)")
    except subprocess.CalledProcessError as e:
        log_print(f"   ⚠️ CUDA CMake 실패, CPU 버전으로 시도: {e}")
        # CUDA 없이 CPU 버전으로 대체
        try:
            cmake_cmd = [
                "cmake", "..", 
                "-DCMAKE_BUILD_TYPE=Release"
            ]
            subprocess.run(cmake_cmd, check=True, cwd=build_dir,
                          capture_output=True, text=True)
            log_print("   ✅ CMake 설정 완료 (CPU 버전)")
        except subprocess.CalledProcessError as e2:
            log_print(f"   ❌ CMake 설정 실패: {e2}")
            return None
    log_print("   빌드 중...")
    try:
        # make를 사용한 빌드
        subprocess.run([
            "make", "-j", str(os.cpu_count() or 4)
        ], check=True, cwd=build_dir)
        log_print("   ✅ 빌드 완료")
        # 빌드된 바이너리를 상위 디렉토리로 복사
        quantize_src = build_dir / "bin" / "llama-quantize"
        quantize_dst = llama_cpp_dir / "llama-quantize"
        if quantize_src.exists():
            shutil.copy2(quantize_src, quantize_dst)
            log_print("   ✅ llama-quantize 복사 완료")
        else:
            # 다른 위치 확인
            alt_paths = [
                build_dir / "llama-quantize",
                build_dir / "quantize"
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    shutil.copy2(alt_path, quantize_dst)
                    log_print(f"   ✅ llama-quantize 복사 완료 ({alt_path})")
                    break
            else:
                log_print("   ⚠️ llama-quantize 바이너리를 찾을 수 없습니다.")
    except subprocess.CalledProcessError as e:
        log_print(f"   ❌ 빌드 실패: {e}")
        return None
    # 변환 스크립트 실행 권한 확인
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        log_print(f"   ❌ 변환 스크립트를 찾을 수 없습니다: {convert_script}")
        return None
    return llama_cpp_dir

def convert_hf_to_gguf(merged_model_dir, llama_cpp_dir):
    """HuggingFace 모델을 GGUF로 변환"""
    log_print("🔄 HuggingFace → GGUF 변환 중...")
    
    if not merged_model_dir.exists():
        log_print("❌ 병합된 모델을 찾을 수 없습니다.")
        log_print("   먼저 4단계를 실행하세요: python3 step4_merge_adapters.py")
        return None
    
    # 출력 파일명 설정
    gguf_output_dir = OUTPUT_DIR / "gguf_models"
    gguf_output_dir.mkdir(exist_ok=True)
    
    gguf_filename = f"{MODEL_BASE_NAME}-korean-f16.gguf"
    gguf_output_path = gguf_output_dir / gguf_filename
    
    # 변환 실행
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    
    log_print(f"   입력: {merged_model_dir}")
    log_print(f"   출력: {gguf_output_path}")
    
    start_time = time.time()
    
    try:
        cmd = [
            "python3", str(convert_script),
            str(merged_model_dir),
            "--outfile", str(gguf_output_path),
            "--outtype", "f16"  # 16-bit 부동소수점
        ]
        
        log_print(f"   실행 명령: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        conversion_time = time.time() - start_time
        log_print(f"   ✅ 변환 완료 ({conversion_time:.1f}초)")
        
        # 파일 크기 확인
        if gguf_output_path.exists():
            size_gb = gguf_output_path.stat().st_size / (1024**3)
            log_print(f"   GGUF 파일 크기: {size_gb:.1f}GB")
        
        return gguf_output_path
        
    except subprocess.CalledProcessError as e:
        log_print(f"   ❌ 변환 실패: {e}")
        if e.stderr:
            log_print(f"   오류 메시지: {e.stderr}")
        return None

def quantize_gguf_model(gguf_path, llama_cpp_dir):
    """GGUF 모델 양자화"""
    log_print("🗜️ GGUF 모델 양자화 중...")
    
    # 양자화된 모델 출력 경로
    quantized_filename = GGUF_CONFIG["output_filename"]
    quantized_path = gguf_path.parent / quantized_filename
    
    # quantize 바이너리 확인
    quantize_binary = llama_cpp_dir / "llama-quantize"
    if not quantize_binary.exists():
        quantize_binary = llama_cpp_dir / "quantize"  # 이전 버전 호환성
    
    if not quantize_binary.exists():
        log_print("   ❌ quantize 바이너리를 찾을 수 없습니다.")
        return None
    
    quant_type = GGUF_CONFIG["quantization_type"].upper()
    log_print(f"   양자화 타입: {quant_type}")
    log_print(f"   입력: {gguf_path}")
    log_print(f"   출력: {quantized_path}")
    
    start_time = time.time()
    
    try:
        cmd = [
            str(quantize_binary),
            str(gguf_path),
            str(quantized_path),
            quant_type
        ]
        
        log_print(f"   실행 명령: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        quantization_time = time.time() - start_time
        log_print(f"   ✅ 양자화 완료 ({quantization_time:.1f}초)")
        
        # 압축률 계산
        if quantized_path.exists():
            original_size = gguf_path.stat().st_size
            quantized_size = quantized_path.stat().st_size
            compression_ratio = quantized_size / original_size
            
            log_print(f"   원본 크기: {original_size/(1024**3):.1f}GB")
            log_print(f"   양자화 크기: {quantized_size/(1024**3):.1f}GB")
            log_print(f"   압축률: {compression_ratio:.2f} ({(1-compression_ratio)*100:.1f}% 절약)")
        
        return quantized_path
        
    except subprocess.CalledProcessError as e:
        log_print(f"   ❌ 양자화 실패: {e}")
        if e.stderr:
            log_print(f"   오류 메시지: {e.stderr}")
        return None

def check_ollama_installation():
    """Ollama 설치 확인"""
    log_print("🔍 Ollama 설치 확인 중...")
    
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        log_print(f"   ✅ Ollama 설치됨: {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        log_print("   ❌ Ollama가 설치되지 않았습니다.")
        log_print("   설치 방법: curl -fsSL https://ollama.com/install.sh | sh")
        return False

def create_ollama_modelfile(quantized_gguf_path):
    """Ollama Modelfile 생성"""
    log_print("📝 Ollama Modelfile 생성 중...")
    
    modelfile_content = f"""FROM {quantized_gguf_path.name}

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER temperature {OLLAMA_CONFIG['temperature']}
PARAMETER top_p {OLLAMA_CONFIG['top_p']}
PARAMETER repeat_penalty {OLLAMA_CONFIG['repeat_penalty']}
PARAMETER num_ctx {OLLAMA_CONFIG['num_ctx']}
PARAMETER num_keep {OLLAMA_CONFIG['num_keep']}

SYSTEM \"\"\"당신은 한국어 기술 전문가 AI입니다. 파이썬, 머신러닝, 웹개발, Docker, Git 등의 기술적 질문에 정확하고 실용적인 답변을 제공합니다.\"\"\"
"""
    
    modelfile_path = quantized_gguf_path.parent / f"{MODEL_BASE_NAME.lower()}-finetune-modelfile"
    
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    log_print(f"   ✅ Modelfile 생성: {modelfile_path}")
    return modelfile_path

def register_ollama_model(modelfile_path):
    """Ollama에 모델 등록"""
    log_print("📋 Ollama에 모델 등록 중...")
    
    model_name = OLLAMA_CONFIG['model_name']
    
    try:
        cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
        log_print(f"   실행 명령: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        log_print(f"   ✅ 모델 등록 완료: {model_name}")
        return model_name
        
    except subprocess.CalledProcessError as e:
        log_print(f"   ❌ 모델 등록 실패: {e}")
        if e.stderr:
            log_print(f"   오류 메시지: {e.stderr}")
        return None

def test_ollama_model(model_name):
    """등록된 Ollama 모델 테스트"""
    log_print("🧪 Ollama 모델 테스트 중...")
    
    test_prompts = [
        "안녕하세요! 간단한 자기소개를 해주세요.",
        "파이썬 리스트와 튜플의 차이점을 설명해주세요.",
        "머신러닝에서 과적합이란 무엇인가요?"
    ]
    
    for i, prompt in enumerate(test_prompts):
        log_print(f"\n   테스트 {i+1}/3: {prompt[:30]}...")
        
        try:
            start_time = time.time()
            
            cmd = ["ollama", "run", model_name, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=60, check=True)
            
            response_time = time.time() - start_time
            response = result.stdout.strip()
            
            log_print(f"   응답 시간: {response_time:.1f}초")
            log_print(f"   응답: {response[:150]}...")
            
        except subprocess.TimeoutExpired:
            log_print("   ⚠️ 응답 시간 초과")
        except subprocess.CalledProcessError as e:
            log_print(f"   ❌ 테스트 실패: {e}")
            return False
    
    log_print("   ✅ 모델 테스트 완료")
    return True

def cleanup_temp_files():
    """임시 파일 정리"""
    log_print("🧹 임시 파일 정리 중...")
    
    # F16 GGUF 파일 삭제 (양자화된 버전만 유지)
    gguf_dir = OUTPUT_DIR / "gguf_models"
    if gguf_dir.exists():
        for f16_file in gguf_dir.glob("*-f16.gguf"):
            try:
                f16_file.unlink()
                log_print(f"   삭제: {f16_file.name}")
            except OSError:
                pass
    
    log_print("   임시 파일 정리 완료")

def main():
    """메인 실행 함수"""
    log_print("⚡ 5단계: GGUF 변환 및 Ollama 등록 시작")
    log_print("=" * 50)
    
    try:
        # 1. 필수 도구 확인
        if not check_prerequisites():
            return False
        
        # 2. 병합된 모델 확인
        merged_model_dir = OUTPUT_DIR / "qwen3_finetune_merged"
        if not merged_model_dir.exists():
            log_print("❌ 병합된 모델을 찾을 수 없습니다.")
            log_print("   먼저 4단계를 실행하세요: python3 step4_merge_adapters.py")
            return False
        
        # 3. llama.cpp 환경 구축
        llama_cpp_dir = setup_llama_cpp()
        if llama_cpp_dir is None:
            return False
        
        # 4. HuggingFace → GGUF 변환
        gguf_path = convert_hf_to_gguf(merged_model_dir, llama_cpp_dir)
        if gguf_path is None:
            return False
        
        # 5. GGUF 모델 양자화
        quantized_path = quantize_gguf_model(gguf_path, llama_cpp_dir)
        if quantized_path is None:
            return False
        
        # 6. Ollama 설치 확인
        if not check_ollama_installation():
            log_print("   Ollama 설치 후 다시 실행하세요.")
            return False
        
        # 7. Ollama Modelfile 생성
        modelfile_path = create_ollama_modelfile(quantized_path)
        
        # 8. Ollama에 모델 등록
        model_name = register_ollama_model(modelfile_path)
        if model_name is None:
            return False
        
        # 9. 모델 테스트
        if not test_ollama_model(model_name):
            log_print("   ⚠️ 모델 테스트에 문제가 있지만 등록은 완료되었습니다.")
        
        # 10. 임시 파일 정리
        cleanup_temp_files()

        # GGUF 파일을 models 디렉토리로 이동
        try:
            MODELS_DIR.mkdir(exist_ok=True)
            final_gguf_path = MODELS_DIR / quantized_path.name
            shutil.move(str(quantized_path), str(final_gguf_path))
            log_print(f"   ✅ GGUF 파일을 models 디렉토리로 이동: {final_gguf_path}")
        except Exception as e:
            log_print(f"   ⚠️ GGUF 파일 이동 실패: {e}")

        log_print("=" * 50)
        log_print("✅ 5단계 완료!")
        log_print("📋 결과:")
        log_print(f"   GGUF 모델: {final_gguf_path if 'final_gguf_path' in locals() else quantized_path}")
        log_print(f"   Ollama 모델명: {model_name}")
        log_print(f"   양자화: {GGUF_CONFIG['quantization_type'].upper()}")
        log_print("\n📋 사용 방법:")
        log_print(f"   ollama run {model_name}")
        log_print(f"   또는 API: http://localhost:11434/api/generate")
        log_print("\n📋 다음 단계 안내:")
        log_print("   - 6단계 실행: python3 step6_optimize_deployment.py")
        log_print("   - API 서버 최적화 및 배포 준비")
        return True
        
    except Exception as e:
        log_print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
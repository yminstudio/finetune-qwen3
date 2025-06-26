#!/usr/bin/env python3
"""
🔍 Step 0: 시스템 요구사항 체크

파이프라인 실행 전 필수 요구사항을 자동으로 확인합니다:
- GPU 메모리 45GB+ 확인
- 시스템 RAM 64GB+ 확인  
- 디스크 여유공간 150GB+ 확인
- CUDA 설치 확인
- Python 환경 확인
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_banner():
    """체크 시작 배너"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║  🔍 시스템 요구사항 체크 (Step 0)                                   ║
║                                                                  ║
║  QLora 파인튜닝 파이프라인 실행을 위한 필수 조건들을 확인합니다      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_python_version():
    """Python 버전 확인"""
    print("🐍 Python 버전 확인 중...")
    
    version = sys.version_info
    python_version = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"   현재 Python 버전: {python_version}")
    
    if version >= (3, 8):
        print("   ✅ Python 3.8+ 요구사항 만족")
        return True
    else:
        print("   ❌ Python 3.8 이상이 필요합니다")
        print("   💡 해결방법: Python 3.8+ 설치")
        return False

def check_system_info():
    """시스템 기본 정보 출력"""
    print("\n💻 시스템 정보:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   아키텍처: {platform.machine()}")
    print(f"   프로세서: {platform.processor()}")

def check_ram():
    """시스템 RAM 확인"""
    print("\n💾 시스템 RAM 확인 중...")
    
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        used_gb = memory.used / (1024**3)
        
        print(f"   총 RAM: {total_gb:.1f}GB")
        print(f"   사용 중: {used_gb:.1f}GB")
        print(f"   사용 가능: {available_gb:.1f}GB")
        
        if total_gb >= 64:
            print("   ✅ RAM 64GB+ 요구사항 만족")
            return True
        elif total_gb >= 48:
            print("   ⚠️ RAM 48GB-64GB: 최소 요구사항 만족 (권장: 64GB+)")
            return True
        else:
            print("   ❌ RAM 부족: 최소 48GB 필요, 권장 64GB+")
            print("   💡 해결방법: 시스템 RAM 업그레이드")
            return False
            
    except ImportError:
        print("   ❌ psutil 패키지가 설치되지 않았습니다")
        print("   💡 해결방법: pip install psutil")
        return False

def check_disk_space():
    """디스크 여유공간 확인"""
    print("\n💽 디스크 여유공간 확인 중...")
    
    try:
        import psutil
        
        # 현재 디렉토리의 디스크 사용량 확인
        current_path = Path.cwd()
        disk_usage = psutil.disk_usage(current_path)
        
        total_gb = disk_usage.total / (1024**3)
        used_gb = disk_usage.used / (1024**3)
        free_gb = disk_usage.free / (1024**3)
        
        print(f"   디스크 경로: {current_path}")
        print(f"   총 용량: {total_gb:.1f}GB")
        print(f"   사용 중: {used_gb:.1f}GB")
        print(f"   여유 공간: {free_gb:.1f}GB")
        
        if free_gb >= 150:
            print("   ✅ 디스크 150GB+ 요구사항 만족")
            return True
        elif free_gb >= 100:
            print("   ⚠️ 디스크 100GB-150GB: 최소 요구사항 만족")
            return True
        else:
            print("   ❌ 디스크 공간 부족: 최소 150GB 필요")
            print("   💡 해결방법: 디스크 정리 또는 추가 저장공간 확보")
            return False
            
    except ImportError:
        print("   ❌ psutil 패키지가 설치되지 않았습니다")
        return False

def check_cuda():
    """CUDA 설치 확인"""
    print("\n⚡ CUDA 설치 확인 중...")
    
    # nvidia-smi 명령어 확인
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print("   ✅ nvidia-smi 명령어 사용 가능")
        
        # CUDA 버전 정보 추출
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if 'CUDA Version:' in line:
                cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                print(f"   CUDA 드라이버 버전: {cuda_version}")
                break
                
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ nvidia-smi 명령어를 찾을 수 없습니다")
        print("   💡 해결방법: NVIDIA GPU 드라이버 설치")
        return False
    
    # nvcc 명령어 확인
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
        nvcc_output = result.stdout
        for line in nvcc_output.split('\n'):
            if 'release' in line:
                cuda_toolkit_version = line.split('release')[1].split(',')[0].strip()
                print(f"   CUDA Toolkit 버전: {cuda_toolkit_version}")
                break
        print("   ✅ CUDA Toolkit 설치 확인")
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ⚠️ nvcc 명령어를 찾을 수 없습니다 (CUDA Toolkit 미설치)")
        print("   💡 PyTorch로 CUDA 지원 확인 중...")
    
    return True

def check_gpu():
    """GPU 메모리 확인"""
    print("\n🎮 GPU 확인 중...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("   ❌ CUDA GPU를 사용할 수 없습니다")
            print("   💡 해결방법: CUDA 호환 GPU 및 PyTorch CUDA 버전 설치")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"   사용 가능한 GPU 수: {gpu_count}개")
        
        total_memory_gb = 0
        gpu_pass = False
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            print(f"   GPU {i}: {gpu_name}")
            print(f"   VRAM: {gpu_memory_gb:.1f}GB")
            
            total_memory_gb += gpu_memory_gb
            
            if gpu_memory_gb >= 45:
                gpu_pass = True
        
        print(f"   총 VRAM: {total_memory_gb:.1f}GB")
        
        if gpu_pass:
            print("   ✅ GPU 메모리 45GB+ 요구사항 만족")
            return True
        elif total_memory_gb >= 40:
            print("   ⚠️ GPU 메모리 40GB-45GB: 거의 만족 (일부 제한 가능)")
            return True
        else:
            print("   ❌ GPU 메모리 부족: 45GB+ VRAM 필요")
            print("   💡 권장 GPU: RTX A6000 (48GB), H100 (80GB), A100 (40GB/80GB)")
            return False
            
    except ImportError:
        print("   ❌ PyTorch가 설치되지 않았습니다")
        print("   💡 해결방법: pip install torch")
        return False
    except Exception as e:
        print(f"   ❌ GPU 확인 중 오류 발생: {e}")
        return False

def check_gpu_detailed():
    """상세 GPU 정보 확인"""
    print("\n🔍 상세 GPU 정보:")
    
    try:
        import GPUtil
        
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("   GPUtil로 GPU 정보를 가져올 수 없습니다")
            return
        
        for gpu in gpus:
            print(f"   GPU {gpu.id}: {gpu.name}")
            print(f"   메모리: {gpu.memoryTotal:.0f}MB (사용중: {gpu.memoryUsed:.0f}MB)")
            print(f"   온도: {gpu.temperature}°C")
            print(f"   사용률: {gpu.load*100:.1f}%")
            
    except ImportError:
        print("   GPUtil 패키지가 설치되지 않았습니다")
    except Exception as e:
        print(f"   상세 GPU 정보 확인 실패: {e}")

def check_essential_packages():
    """필수 패키지 설치 여부 확인"""
    print("\n📦 필수 패키지 확인 중...")
    
    essential_packages = [
        'torch',
        'transformers', 
        'accelerate',
        'bitsandbytes',
        'peft',
        'datasets',
        'psutil',
        'GPUtil'
    ]
    
    missing_packages = []
    
    for package in essential_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}: 설치됨")
        except ImportError:
            print(f"   ❌ {package}: 설치되지 않음")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   💡 누락된 패키지: {', '.join(missing_packages)}")
        print("   해결방법: pip install -r requirements.txt")
        return False
    else:
        print("   ✅ 모든 필수 패키지 설치 확인")
        return True

def check_network():
    """네트워크 연결 확인"""
    print("\n🌐 네트워크 연결 확인 중...")
    
    try:
        import requests
        
        # Hugging Face Hub 연결 확인
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            print("   ✅ Hugging Face Hub 연결 가능")
        else:
            print("   ⚠️ Hugging Face Hub 연결 확인 필요")
            
        # GitHub 연결 확인 (llama.cpp 다운로드용)
        response = requests.get("https://github.com", timeout=10)
        if response.status_code == 200:
            print("   ✅ GitHub 연결 가능")
        else:
            print("   ⚠️ GitHub 연결 확인 필요")
            
        return True
        
    except ImportError:
        print("   ❌ requests 패키지가 설치되지 않았습니다")
        return False
    except Exception as e:
        print(f"   ⚠️ 네트워크 연결 확인 실패: {e}")
        return True  # 네트워크는 필수가 아니므로 True 반환

def estimate_time_and_space():
    """예상 소요시간 및 공간 계산"""
    print("\n⏱️ 예상 소요시간 및 공간:")
    
    print("   📊 단계별 예상 시간:")
    print("   1단계 (환경 설정): 30분")
    print("   2단계 (QLora 학습): 2-3시간")
    print("   3단계 (결과 검증): 30분")
    print("   4단계 (모델 병합): 1시간")
    print("   5단계 (GGUF 변환): 1-2시간")
    print("   6단계 (서비스 배포): 1시간")
    print("   💡 총 소요시간: 6-8시간")
    
    print("\n   💾 예상 저장공간:")
    print("   원본 모델 캐시: ~30GB")
    print("   LoRA 어댑터: ~3GB")
    print("   병합된 모델: ~60GB")
    print("   GGUF 모델: ~20GB")
    print("   임시 파일들: ~10GB")
    print("   💡 총 필요공간: ~120-150GB")

def provide_solutions():
    """문제 해결 방법 제시"""
    print("\n🛠️ 일반적인 문제 해결 방법:")
    
    print("\n   GPU 메모리 부족:")
    print("   - gradient_accumulation_steps 늘리기")
    print("   - per_device_train_batch_size 줄이기")
    print("   - max_seq_length 줄이기")
    
    print("\n   시스템 RAM 부족:")
    print("   - 브라우저 및 불필요한 프로그램 종료")
    print("   - swap 메모리 활성화")
    print("   - max_steps 줄여서 단계별 실행")
    
    print("\n   디스크 공간 부족:")
    print("   - 불필요한 파일 삭제")
    print("   - 외장 드라이브 활용")
    print("   - Docker 캐시 정리")
    
    print("\n   CUDA 문제:")
    print("   - NVIDIA 드라이버 재설치")
    print("   - PyTorch CUDA 버전 확인")
    print("   - CUDA Toolkit 설치")

def main():
    """메인 실행 함수"""
    print_banner()
    
    # 시스템 정보 출력
    check_system_info()
    
    # 각 요구사항 체크
    checks = []
    
    checks.append(("Python 버전", check_python_version()))
    checks.append(("시스템 RAM", check_ram()))
    checks.append(("디스크 공간", check_disk_space()))
    checks.append(("CUDA 설치", check_cuda()))
    checks.append(("GPU 메모리", check_gpu()))
    checks.append(("필수 패키지", check_essential_packages()))
    checks.append(("네트워크 연결", check_network()))
    
    # 상세 GPU 정보
    check_gpu_detailed()
    
    # 예상 시간 및 공간
    estimate_time_and_space()
    
    # 결과 요약
    print("\n" + "="*70)
    print("📋 요구사항 체크 결과:")
    print("="*70)
    
    passed = 0
    total = len(checks)
    
    for check_name, result in checks:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"   {check_name:<15}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 전체 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 요구사항을 만족합니다!")
        print("✅ 파이프라인 실행 준비 완료")
        print("\n🚀 다음 단계:")
        print("   python3 run_pipeline.py")
        return True
    elif passed >= total - 2:
        print("⚠️ 대부분의 요구사항을 만족합니다")
        print("💡 일부 제한이 있을 수 있지만 실행 가능합니다")
        
        proceed = input("\n계속 진행하시겠습니까? (y/N): ").lower()
        if proceed == 'y':
            print("✅ 파이프라인 실행을 진행합니다")
            return True
        else:
            provide_solutions()
            return False
    else:
        print("❌ 필수 요구사항이 부족합니다")
        print("💡 아래 해결방법을 참고하여 환경을 준비해주세요")
        provide_solutions()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
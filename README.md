# 🚀 Qwen3 Korean QLora Fine-tuning Pipeline

**Qwen3 모델을 한국어 기술 Q&A로 파인튜닝하고 프로덕션 서비스로 배포하는 완전 자동화 파이프라인**

## ✨ 주요 특징

- 🤖 **Qwen3-4B 모델** 한국어 파인튜닝
- ⚡ **QLora 기법** 사용으로 메모리 효율적 학습
- 🔄 **완전 자동화** 6단계 파이프라인
- 🚀 **프로덕션 배포** Ollama + FastAPI
- 📊 **성능 최적화** GGUF 양자화 적용
- 🎯 **37.7 토큰/초** 고속 추론

## 📋 파이프라인 단계

| 단계 | 설명 | 소요시간 | 출력물 |
|------|------|----------|--------|
| 0️⃣ | 시스템 요구사항 체크 | 5분 | 환경 검증 |
| 1️⃣ | 모델 준비 및 환경 설정 | 30분 | 베이스 모델 캐시 |
| 2️⃣ | QLora 파인튜닝 실행 | 6분* | LoRA 어댑터 |
| 3️⃣ | 학습 결과 검증 | 16분 | 성능 리포트 |
| 4️⃣ | LoRA 어댑터 병합 | 4분 | 병합된 모델 |
| 5️⃣ | GGUF 변환 및 Ollama 등록 | 7분 | 양자화 모델 |
| 6️⃣ | 서비스 최적화 및 배포 | 1분 | API 서버 |

> *GPU RTX A6000 기준, 5개 샘플 데이터

## 🛠️ 시스템 요구사항

### 최소 사양
- **GPU**: NVIDIA RTX A6000 (48GB VRAM) 또는 동급
- **RAM**: 62GB
- **디스크**: 150GB 여유 공간
- **OS**: Ubuntu 22.04 LTS
- **CUDA**: 12.8+

### 필수 패키지
- Python 3.10+
- PyTorch 2.6.0+
- transformers 4.52.3+
- CUDA Toolkit

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 저장소 클론
git clone <your-repo-url>
cd qwen3-korean-finetune

# 의존성 설치
pip install -r requirements.txt
```

### 2. 전체 파이프라인 실행
```bash
# 원클릭 실행 (약 1시간)
python3 run_pipeline.py
```

### 3. 서비스 시작
```bash
# Ollama 서버
ollama run qwen3-4b-finetune

# API 서버 (별도 터미널)
python3 api_server.py

# 브라우저에서 확인
# http://localhost:8000/docs
```

## 📁 프로젝트 구조

```
qwen3-korean-finetune/
├── 📄 run_pipeline.py          # 메인 파이프라인
├── ⚙️ config.py                # 설정 파일
├── 📋 requirements.txt         # 의존성
├── 🌐 api_server.py            # FastAPI 서버
├── 📊 dashboard.html           # 모니터링 대시보드
├── 🔧 step0_check_requirements.py  # 시스템 체크
├── 🔧 step1_setup_environment.py   # 환경 설정
├── 🔧 step2_train_qlora.py         # 파인튜닝
├── 🔧 step3_validate_results.py    # 결과 검증
├── 🔧 step4_merge_adapters.py      # 모델 병합
├── 🔧 step5_convert_to_gguf.py     # GGUF 변환
├── 🔧 step6_optimize_deployment.py # 배포 최적화
├── 📁 data/                    # 학습 데이터
└── 📁 models/                  # 출력 모델
```

## 🎯 성능 결과

### 학습 성능
- **파인튜닝 효과**: 97.3% 유사도
- **학습 시간**: 6분 (100 스텝)
- **메모리 사용**: 2.8GB GPU

### 추론 성능
- **짧은 응답**: 30.7 토큰/초
- **중간 응답**: 38.2 토큰/초  
- **긴 응답**: 44.1 토큰/초
- **평균**: 37.7 토큰/초

### 모델 크기
- **원본 모델**: 7.5GB
- **양자화 모델**: 2.3GB (69% 절약)

## 🔧 커스터마이징

### 모델 변경
```python
# config.py
MODEL_BASE_NAME = "Qwen3-8B"  # 4B → 8B로 변경
```

### 학습 데이터 추가
```bash
# data/ 폴더에 JSON 파일 추가
# 형식: [{"messages": [{"role": "user", "content": "질문"}, {"role": "assistant", "content": "답변"}]}]
```

### 파라미터 조정
```python
# config.py
TRAINING_CONFIG = {
    "max_steps": 200,           # 학습 스텝 증가
    "learning_rate": 1e-4,      # 학습률 조정
    "per_device_train_batch_size": 2  # 배치 크기 증가
}
```

## 📊 API 사용법

### FastAPI 엔드포인트

```bash
# 건강 상태 확인
curl http://localhost:8000/health

# 채팅 API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "파이썬 리스트 컴프리헨션 설명해줘",
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

### Ollama 직접 사용
```bash
# 대화형 모드
ollama run qwen3-4b-finetune

# API 모드
curl http://localhost:11434/api/generate \
  -d '{
    "model": "qwen3-4b-finetune",
    "prompt": "Docker 최적화 팁은?",
    "stream": false
  }'
```

## 🔍 트러블슈팅

### 메모리 부족
```bash
# GPU 메모리 확인
nvidia-smi

# 배치 크기 줄이기
# config.py에서 per_device_train_batch_size = 1
```

### 모델 로딩 실패
```bash
# 캐시 정리
rm -rf cache/ unsloth_compiled_cache/

# 다시 실행
python3 step1_setup_environment.py
```

## 📄 라이선스

MIT License
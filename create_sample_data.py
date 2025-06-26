#!/usr/bin/env python3
"""
📊 한국어 기술 QA 샘플 데이터 생성기

5개 도메인의 기술 QA 데이터를 ChatML 형식으로 생성
- 파이썬, 머신러닝, 웹개발, Docker, Git
"""

import json
import os
from pathlib import Path
from config import PROJECT_ROOT

def create_sample_data():
    """샘플 학습 데이터 생성"""
    
    # 5개 도메인의 기술 QA 데이터
    sample_data = [
        # 파이썬 도메인
        {
            "messages": [
                {"role": "system", "content": "당신은 파이썬 전문가입니다. 명확하고 실용적인 답변을 제공하세요."},
                {"role": "user", "content": "파이썬에서 리스트 컴프리헨션과 일반 for 루프의 성능 차이를 설명해주세요."},
                {"role": "assistant", "content": "리스트 컴프리헨션이 일반 for 루프보다 약 2-3배 빠릅니다.\n\n**리스트 컴프리헨션:**\n```python\nsquares = [x**2 for x in range(1000)]\n```\n\n**일반 for 루프:**\n```python\nsquares = []\nfor x in range(1000):\n    squares.append(x**2)\n```\n\n**성능 차이 이유:**\n1. C 레벨에서 최적화된 내장 함수 사용\n2. 함수 호출 오버헤드 감소\n3. 메모리 할당 최적화\n\n단, 복잡한 로직에는 가독성을 위해 일반 루프를 사용하는 것이 좋습니다."}
            ]
        },
        
        # 머신러닝 도메인
        {
            "messages": [
                {"role": "system", "content": "당신은 머신러닝 전문가입니다. 실무에 도움되는 답변을 제공하세요."},
                {"role": "user", "content": "과적합(Overfitting)을 방지하는 주요 기법들을 알려주세요."},
                {"role": "assistant", "content": "과적합 방지를 위한 주요 기법들입니다:\n\n**1. 정규화 (Regularization)**\n- L1 정규화: 특성 선택 효과\n- L2 정규화: 가중치 크기 제한\n\n**2. 드롭아웃 (Dropout)**\n```python\nmodel.add(Dropout(0.5))\n```\n\n**3. 조기 종료 (Early Stopping)**\n- 검증 손실이 증가하기 시작하면 학습 중단\n\n**4. 데이터 증강 (Data Augmentation)**\n- 이미지: 회전, 크기 조정, 밝기 변경\n- 텍스트: 동의어 치환, 백번역\n\n**5. 교차 검증 (Cross-validation)**\n- K-fold CV로 모델 일반화 성능 평가\n\n적절한 기법 조합으로 과적합을 효과적으로 방지할 수 있습니다."}
            ]
        },
        
        # 웹개발 도메인
        {
            "messages": [
                {"role": "system", "content": "당신은 웹개발 전문가입니다. 실용적이고 보안을 고려한 답변을 제공하세요."},
                {"role": "user", "content": "RESTful API 설계 시 고려해야 할 핵심 원칙들을 설명해주세요."},
                {"role": "assistant", "content": "RESTful API 설계의 핵심 원칙들입니다:\n\n**1. HTTP 메서드 적절한 사용**\n- GET: 조회\n- POST: 생성\n- PUT: 전체 수정\n- PATCH: 부분 수정\n- DELETE: 삭제\n\n**2. 명확한 URL 구조**\n```\n/api/v1/users          # 사용자 목록\n/api/v1/users/{id}     # 특정 사용자\n/api/v1/users/{id}/orders  # 사용자의 주문들\n```\n\n**3. 상태 코드 활용**\n- 200: 성공\n- 201: 생성 완료\n- 400: 잘못된 요청\n- 401: 인증 필요\n- 404: 리소스 없음\n- 500: 서버 오류\n\n**4. 일관된 응답 형식**\n```json\n{\n  \"status\": \"success\",\n  \"data\": {...},\n  \"message\": \"처리 완료\"\n}\n```\n\n**5. 버전 관리와 보안**\n- API 버전 명시\n- 인증/인가 구현\n- 입력 검증 필수"}
            ]
        },
        
        # Docker 도메인
        {
            "messages": [
                {"role": "system", "content": "당신은 Docker 전문가입니다. 실무에서 바로 활용할 수 있는 답변을 제공하세요."},
                {"role": "user", "content": "Docker 컨테이너 최적화를 위한 Dockerfile 작성 팁을 알려주세요."},
                {"role": "assistant", "content": "Docker 컨테이너 최적화 팁들입니다:\n\n**1. 멀티스테이지 빌드 활용**\n```dockerfile\n# 빌드 스테이지\nFROM node:18 AS builder\nWORKDIR /app\nCOPY package*.json ./\nRUN npm install\nCOPY . .\nRUN npm run build\n\n# 실행 스테이지\nFROM nginx:alpine\nCOPY --from=builder /app/dist /usr/share/nginx/html\n```\n\n**2. 레이어 캐싱 최적화**\n- 자주 변경되지 않는 명령어를 먼저 배치\n- 종속성 설치와 소스 코드 복사 분리\n\n**3. 불필요한 패키지 제거**\n```dockerfile\nRUN apt-get update && apt-get install -y \\\n    package1 package2 \\\n    && rm -rf /var/lib/apt/lists/*\n```\n\n**4. 보안 강화**\n- 루트 사용자 대신 전용 사용자 생성\n- 최소 권한 원칙 적용\n\n**5. 이미지 크기 최소화**\n- alpine 베이스 이미지 사용\n- .dockerignore 파일 활용\n\n이러한 최적화로 빌드 시간과 이미지 크기를 크게 줄일 수 있습니다."}
            ]
        },
        
        # Git 도메인
        {
            "messages": [
                {"role": "system", "content": "당신은 Git 전문가입니다. 협업에 도움되는 실용적인 답변을 제공하세요."},
                {"role": "user", "content": "Git 브랜치 전략과 효과적인 커밋 메시지 작성법을 알려주세요."},
                {"role": "assistant", "content": "효과적인 Git 사용법입니다:\n\n**Git 브랜치 전략 (Git Flow)**\n```\nmain     : 프로덕션 배포용\ndevelop  : 개발 통합 브랜치\nfeature/ : 기능 개발용\nrelease/ : 배포 준비용\nhotfix/  : 긴급 수정용\n```\n\n**브랜치 생성 및 관리**\n```bash\n# 기능 브랜치 생성\ngit checkout -b feature/user-authentication\n\n# 작업 완료 후 develop에 병합\ngit checkout develop\ngit merge feature/user-authentication\n```\n\n**커밋 메시지 규칙**\n```\n타입(범위): 제목\n\n본문 내용\n\n푸터\n```\n\n**커밋 타입**\n- feat: 새 기능\n- fix: 버그 수정\n- docs: 문서 수정\n- style: 코드 포맷팅\n- refactor: 리팩토링\n- test: 테스트 추가\n\n**좋은 커밋 메시지 예시**\n```\nfeat(auth): 사용자 로그인 기능 구현\n\n- JWT 토큰 기반 인증 시스템 추가\n- 로그인 폼 유효성 검사 구현\n- 세션 관리 로직 개발\n\nCloses #123\n```"}
            ]
        }
    ]
    
    return sample_data

def save_data_as_json(data, filename="korean_tech_qa.json"):
    """데이터를 JSON 파일로 저장"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    filepath = data_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 학습 데이터 저장 완료: {filepath}")
    print(f"   샘플 수: {len(data)}개")
    
    # 데이터 미리보기
    print("\n📋 데이터 미리보기:")
    for i, sample in enumerate(data[:2]):
        print(f"--- 샘플 {i+1} ---")
        print(f"질문: {sample['messages'][1]['content'][:50]}...")
        print(f"답변: {sample['messages'][2]['content'][:100]}...")
        print()

def create_alpaca_format(data):
    """Alpaca 형식으로도 저장"""
    alpaca_data = []
    
    for sample in data:
        alpaca_sample = {
            "input": sample['messages'][1]['content'],
            "output": sample['messages'][2]['content'],
            "instruction": sample['messages'][0]['content']
        }
        alpaca_data.append(alpaca_sample)
    
    data_dir = Path("data")
    filepath = data_dir / "korean_tech_qa_alpaca.json"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Alpaca 형식 데이터 저장: {filepath}")

def main():
    """메인 실행 함수"""
    print("📊 한국어 기술 QA 샘플 데이터 생성 중...")
    print("=" * 50)
    
    # 샘플 데이터 생성
    data = create_sample_data()
    
    # ChatML 형식으로 저장
    save_data_as_json(data)
    
    # Alpaca 형식으로도 저장
    create_alpaca_format(data)
    
    print("=" * 50)
    print("✅ 데이터 생성 완료!")
    print("📁 생성된 파일:")
    print(f"   - {PROJECT_ROOT / 'data' / 'korean_tech_qa.json'} (ChatML 형식)")
    print(f"   - {PROJECT_ROOT / 'data' / 'korean_tech_qa_alpaca.json'} (Alpaca 형식)")

if __name__ == "__main__":
    main() 
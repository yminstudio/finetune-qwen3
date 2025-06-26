#!/usr/bin/env python3
"""
🚀 6단계: 서비스 최적화 및 배포 (1시간)

목적:
- 프로덕션 환경에 적합한 성능 달성
- API 서버 구축 및 모니터링 설정
- 확장성 및 안정성 확보
"""

import os
import sys
import time
import json
import subprocess
import threading
import requests
from pathlib import Path
import psutil
import GPUtil
from config import PROJECT_ROOT

# 설정 파일 import
from config import *
from config import OLLAMA_CONFIG, GGUF_CONFIG, LLAMA_CPP_DIR, OUTPUT_DIR

def check_ollama_service():
    """Ollama 서비스 상태 확인"""
    print("🔍 Ollama 서비스 상태 확인 중...")
    
    try:
        # Ollama 프로세스 확인
        ollama_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'ollama' in proc.info['name'].lower():
                ollama_processes.append(proc.info)
        
        if ollama_processes:
            print(f"   ✅ Ollama 프로세스 실행 중: {len(ollama_processes)}개")
        else:
            print("   ⚠️ Ollama 프로세스가 실행되지 않음")
            return False
        
        # API 서버 응답 확인
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"   ✅ API 서버 응답 OK: {len(models)}개 모델")
                return True
            else:
                print(f"   ❌ API 서버 오류: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"   ❌ API 서버 연결 실패: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ 서비스 확인 실패: {e}")
        return False

def start_ollama_service():
    """Ollama 서비스 시작"""
    print("🚀 Ollama 서비스 시작 중...")
    
    try:
        # Ollama 서버 시작 (백그라운드)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # 서비스 시작 대기
        print("   서비스 시작 대기 중...")
        for i in range(30):  # 30초 대기
            time.sleep(1)
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    print("   ✅ Ollama 서비스 시작 완료")
                    return True
            except:
                continue
        
        print("   ❌ Ollama 서비스 시작 실패")
        return False
        
    except Exception as e:
        print(f"   ❌ 서비스 시작 오류: {e}")
        return False

def optimize_ollama_performance():
    """Ollama 성능 최적화"""
    print("⚡ Ollama 성능 최적화 중...")
    
    # 환경 변수 설정
    optimization_env = {
        "OLLAMA_NUM_PARALLEL": "2",          # 병렬 요청 수
        "OLLAMA_MAX_LOADED_MODELS": "1",     # 메모리에 로드할 최대 모델 수
        "OLLAMA_FLASH_ATTENTION": "1",       # Flash Attention 활성화
        "OLLAMA_GPU_OVERHEAD": "0.1",        # GPU 오버헤드 설정
    }
    
    # 현재 환경에 적용
    for key, value in optimization_env.items():
        os.environ[key] = value
        print(f"   {key}={value}")
    
    # 설정 파일 생성
    ollama_config = {
        "experimental": {
            "numa": True
        },
        "gpu": {
            "enabled": True,
            "memory_fraction": 0.9
        },
        "server": {
            "host": "0.0.0.0",
            "port": 11434,
            "timeout": "5m"
        }
    }
    
    config_dir = Path.home() / ".ollama"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(ollama_config, f, indent=2)
    
    print(f"   ✅ 설정 파일 생성: {config_file}")

def benchmark_model_performance():
    """모델 성능 벤치마크"""
    print("📊 모델 성능 벤치마크 중...")
    
    model_name = OLLAMA_CONFIG['model_name']
    
    # 다양한 길이의 테스트 프롬프트
    test_cases = [
        {
            "name": "짧은 응답",
            "prompt": "안녕하세요!",
            "expected_tokens": 20
        },
        {
            "name": "중간 응답", 
            "prompt": "파이썬 리스트 컴프리헨션의 장점을 설명해주세요.",
            "expected_tokens": 150
        },
        {
            "name": "긴 응답",
            "prompt": "머신러닝 프로젝트의 전체 워크플로우를 단계별로 자세히 설명해주세요.",
            "expected_tokens": 300
        }
    ]
    
    benchmark_results = []
    
    for test_case in test_cases:
        print(f"\n   🧪 {test_case['name']} 테스트...")
        
        # 여러 번 실행하여 평균 구하기
        times = []
        token_counts = []
        
        for i in range(3):
            start_time = time.time()
            
            try:
                # API 호출
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_name,
                        "prompt": test_case["prompt"],
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": test_case["expected_tokens"]
                        }
                    },
                    timeout=60
                )
                
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get('response', '')
                    
                    # 토큰 수 추정 (간단한 방법)
                    estimated_tokens = len(generated_text.split()) * 1.3
                    
                    times.append(elapsed_time)
                    token_counts.append(estimated_tokens)
                    
                    print(f"      시도 {i+1}: {elapsed_time:.1f}초, ~{estimated_tokens:.0f} 토큰")
                
            except Exception as e:
                print(f"      시도 {i+1} 실패: {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            avg_tokens = sum(token_counts) / len(token_counts)
            tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0
            
            result = {
                "test_name": test_case["name"],
                "avg_time": avg_time,
                "avg_tokens": avg_tokens,
                "tokens_per_sec": tokens_per_sec
            }
            
            benchmark_results.append(result)
            
            print(f"   평균: {avg_time:.1f}초, {avg_tokens:.0f} 토큰, {tokens_per_sec:.1f} 토큰/초")
    
    return benchmark_results

def stress_test_concurrent_requests():
    """동시 요청 스트레스 테스트"""
    print("🔥 동시 요청 스트레스 테스트 중...")
    
    model_name = OLLAMA_CONFIG['model_name']
    test_prompt = "Python에서 리스트와 딕셔너리의 차이점은?"
    
    def make_request(request_id):
        """단일 요청 실행"""
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {"num_predict": 100}
                },
                timeout=120
            )
            elapsed_time = time.time() - start_time
            
            return {
                "request_id": request_id,
                "success": response.status_code == 200,
                "time": elapsed_time,
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "time": 0,
                "error": str(e)
            }
    
    # 동시 요청 수 테스트
    concurrent_levels = [1, 2, 3]
    stress_results = []
    
    for concurrent_count in concurrent_levels:
        print(f"\n   동시 요청 {concurrent_count}개 테스트...")
        
        # 스레드로 동시 요청 실행
        threads = []
        results = []
        
        start_time = time.time()
        
        for i in range(concurrent_count):
            thread = threading.Thread(
                target=lambda i=i: results.append(make_request(i))
            )
            threads.append(thread)
            thread.start()
        
        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # 결과 분석
        successful_requests = [r for r in results if r.get('success', False)]
        failed_requests = [r for r in results if not r.get('success', False)]
        
        if successful_requests:
            avg_response_time = sum(r['time'] for r in successful_requests) / len(successful_requests)
            success_rate = len(successful_requests) / len(results) * 100
        else:
            avg_response_time = 0
            success_rate = 0
        
        stress_result = {
            "concurrent_requests": concurrent_count,
            "total_time": total_time,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "successful_count": len(successful_requests),
            "failed_count": len(failed_requests)
        }
        
        stress_results.append(stress_result)
        
        print(f"      총 시간: {total_time:.1f}초")
        print(f"      성공률: {success_rate:.1f}%")
        print(f"      평균 응답시간: {avg_response_time:.1f}초")
        
        # 다음 테스트 전 잠시 대기
        time.sleep(5)
    
    return stress_results

def create_api_wrapper():
    """FastAPI 래퍼 생성"""
    print("🌐 API 래퍼 생성 중...")
    
    api_wrapper_code = '''
#!/usr/bin/env python3
"""
FastAPI 래퍼 - Ollama 모델을 위한 RESTful API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import time
from typing import Optional

app = FastAPI(title="Qwen3 Korean API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class ChatResponse(BaseModel):
    response: str
    processing_time: float
    model: str

@app.get("/")
async def root():
    return {"message": "Qwen3 Korean API Server", "status": "running"}

@app.get("/health")
async def health_check():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "ollama": "running"}
        else:
            raise HTTPException(status_code=503, detail="Ollama service unavailable")
    except:
        raise HTTPException(status_code=503, detail="Ollama service unavailable")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3-4b-finetune",
                "prompt": request.message,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            processing_time = time.time() - start_time
            
            return ChatResponse(
                response=result.get('response', ''),
                processing_time=processing_time,
                model="qwen3-4b-finetune"
            )
        else:
            raise HTTPException(status_code=500, detail="Model generation failed")
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    api_file = PROJECT_ROOT / "api_server.py"
    with open(api_file, 'w', encoding='utf-8') as f:
        f.write(api_wrapper_code)
    
    print(f"   ✅ API 래퍼 생성: {api_file}")
    print("   사용법: python3 api_server.py")
    
    return api_file

def create_monitoring_dashboard():
    """모니터링 대시보드 HTML 생성"""
    print("📊 모니터링 대시보드 생성 중...")
    
    dashboard_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Qwen3 Korean Model Monitoring</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 10px 0; }
        .status-ok { color: green; }
        .status-error { color: red; }
        .metric { display: inline-block; margin: 10px; padding: 10px; border-radius: 4px; background: #f5f5f5; }
        button { padding: 10px 20px; margin: 5px; cursor: pointer; }
        #chat-container { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }
        .chat-message { margin: 10px 0; }
        .user-message { text-align: right; color: blue; }
        .bot-message { text-align: left; color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Qwen3 Korean Model Dashboard</h1>
        
        <div class="card">
            <h2>서비스 상태</h2>
            <div id="service-status">Loading...</div>
            <button onclick="checkStatus()">상태 새로고침</button>
        </div>
        
        <div class="card">
            <h2>성능 메트릭</h2>
            <div id="performance-metrics">
                <div class="metric">응답 시간: <span id="response-time">-</span></div>
                <div class="metric">처리량: <span id="throughput">-</span></div>
                <div class="metric">메모리 사용량: <span id="memory-usage">-</span></div>
            </div>
        </div>
        
        <div class="card">
            <h2>테스트 채팅</h2>
            <div id="chat-container"></div>
            <input type="text" id="chat-input" placeholder="메시지를 입력하세요..." style="width: 70%;">
            <button onclick="sendMessage()">전송</button>
        </div>
    </div>

    <script>
        async function checkStatus() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                document.getElementById('service-status').innerHTML = 
                    '<span class="status-ok">✅ 서비스 정상</span>';
            } catch (error) {
                document.getElementById('service-status').innerHTML = 
                    '<span class="status-error">❌ 서비스 오류</span>';
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;
            
            const chatContainer = document.getElementById('chat-container');
            
            // 사용자 메시지 표시
            chatContainer.innerHTML += `<div class="chat-message user-message">👤 ${message}</div>`;
            input.value = '';
            
            // 봇 응답 로딩
            chatContainer.innerHTML += `<div class="chat-message bot-message">🤖 생각 중...</div>`;
            
            try {
                const startTime = Date.now();
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });
                
                const data = await response.json();
                const endTime = Date.now();
                
                // 로딩 메시지 제거 후 실제 응답 표시
                const messages = chatContainer.querySelectorAll('.chat-message');
                messages[messages.length - 1].innerHTML = 
                    `🤖 ${data.response} <small>(${(endTime - startTime)}ms)</small>`;
                
                // 성능 메트릭 업데이트
                document.getElementById('response-time').textContent = `${data.processing_time.toFixed(1)}초`;
                
            } catch (error) {
                const messages = chatContainer.querySelectorAll('.chat-message');
                messages[messages.length - 1].innerHTML = '🤖 ❌ 오류 발생';
            }
            
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Enter 키로 메시지 전송
        document.getElementById('chat-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // 페이지 로드 시 상태 확인
        checkStatus();
        setInterval(checkStatus, 30000); // 30초마다 상태 확인
    </script>
</body>
</html>
'''
    
    dashboard_file = PROJECT_ROOT / "dashboard.html"
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    print(f"   ✅ 대시보드 생성: {dashboard_file}")
    print("   브라우저에서 파일을 열어 모니터링 가능")
    
    return dashboard_file

def save_deployment_report(benchmark_results, stress_results):
    """배포 보고서 저장"""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": OLLAMA_CONFIG['model_name'],
        "performance_benchmark": benchmark_results,
        "stress_test": stress_results,
        "system_info": {
            "cpu_count": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_info": [gpu.name for gpu in GPUtil.getGPUs()] if GPUtil.getGPUs() else []
        },
        "deployment_config": {
            "ollama_config": OLLAMA_CONFIG,
            "api_config": API_CONFIG
        }
    }
    
    report_file = OUTPUT_DIR / "deployment_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"💾 배포 보고서 저장: {report_file}")

def register_model_to_ollama():
    """Ollama에 gguf 모델 등록 (Modelfile 자동 생성)"""
    print("📝 Ollama에 모델 등록 중...")
    # 실제 양자화된 GGUF 파일 경로로 수정
    gguf_path = PROJECT_ROOT / "models" / f"{MODEL_BASE_NAME.lower()}-finetune-q4_k_m.gguf"
    modelfile_path = gguf_path.parent / "modelfile"
    
    if not gguf_path.exists():
        print(f"❌ GGUF 파일이 존재하지 않습니다: {gguf_path}")
        return False
    
    # Modelfile 생성
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(f"FROM {gguf_path}\n")
    
    print(f"   ✅ Modelfile 생성: {modelfile_path}")
    
    try:
        # ollama create 명령어 실행 (Modelfile 사용)
        import subprocess
        model_name = f"{MODEL_BASE_NAME.lower()}-finetune"
        result = subprocess.run([
            "ollama", "create", model_name, "-f", str(modelfile_path)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ 모델 등록 성공: {model_name}")
            return True
        else:
            print(f"   ❌ 모델 등록 실패: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ 모델 등록 중 오류: {e}")
        return False

def test_ollama_inference():
    """Ollama에 질문을 보내고 응답을 받는 테스트"""
    print("🤖 Ollama 질문-응답 테스트 중...")
    import requests
    test_questions = [
        "파이썬에서 리스트와 딕셔너리의 차이점은?",
        "머신러닝에서 과적합을 방지하는 방법은?",
        "Dockerfile 최적화 팁을 알려줘.",
        "Git 브랜치 전략을 설명해줘."
    ]
    results = []
    for q in test_questions:
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": f"{MODEL_BASE_NAME.lower()}-finetune",
                    "prompt": q,
                    "stream": False,
                    "options": {"num_predict": 128}
                },
                timeout=60
            )
            if response.status_code == 200:
                answer = response.json().get('response', '')
                print(f"Q: {q}\nA: {answer[:200]}\n{'-'*40}")
                results.append({"question": q, "answer": answer})
            else:
                print(f"❌ 응답 실패: {response.status_code}")
                results.append({"question": q, "error": f"HTTP {response.status_code}"})
        except Exception as e:
            print(f"❌ 오류: {e}")
            results.append({"question": q, "error": str(e)})
    # 결과 저장
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    with open(logs_dir / "ollama_inference_test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 결과 저장: {logs_dir / 'ollama_inference_test.json'}")
    return results

def main():
    """메인 실행 함수"""
    print("🚀 6단계: 서비스 최적화 및 배포 시작")
    print("=" * 50)
    
    try:
        # 1. Ollama 서비스 확인/시작
        if not check_ollama_service():
            if not start_ollama_service():
                print("❌ Ollama 서비스 시작 실패")
                return False
        # 1-1. Ollama에 모델 등록
        if not register_model_to_ollama():
            print("❌ Ollama 모델 등록 실패")
            return False
        # 1-2. Ollama 질문-응답 테스트
        test_ollama_inference()
        
        # 2. 성능 최적화
        optimize_ollama_performance()
        
        # 3. 모델 성능 벤치마크
        benchmark_results = benchmark_model_performance()
        
        # 4. 스트레스 테스트
        stress_results = stress_test_concurrent_requests()
        
        # 5. API 래퍼 생성
        api_file = create_api_wrapper()
        
        # 6. 모니터링 대시보드 생성
        dashboard_file = create_monitoring_dashboard()
        
        # 7. 배포 보고서 저장
        save_deployment_report(benchmark_results, stress_results)
        
        print("=" * 50)
        print("✅ 6단계 완료!")
        print("🎉 전체 파이프라인 완료!")
        
        print("\n📋 최종 결과:")
        if benchmark_results:
            avg_speed = sum(r['tokens_per_sec'] for r in benchmark_results) / len(benchmark_results)
            print(f"   평균 생성 속도: {avg_speed:.1f} 토큰/초")
        
        if stress_results:
            valid_concurrent = [r['concurrent_requests'] for r in stress_results if r['success_rate'] > 80]
            if valid_concurrent:
                max_concurrent = max(valid_concurrent)
                print(f"   권장 동시 사용자: {max_concurrent}명")
            else:
                print("   권장 동시 사용자: (Ollama는 동시 요청을 공식 지원하지 않음)")
        
        print("\n🚀 서비스 시작 방법:")
        print(f"   1. Ollama: ollama run {OLLAMA_CONFIG['model_name']}")
        print(f"   2. API 서버: python3 {api_file}")
        print(f"   3. 모니터링: 브라우저에서 {dashboard_file} 열기")
        
        print("\n📊 API 엔드포인트:")
        print("   - GET  http://localhost:8000/health")
        print("   - POST http://localhost:8000/chat")
        print("   - Ollama http://localhost:11434/api/generate")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
#!/usr/bin/env python3
"""
🧪 3단계: 학습 결과 검증 (30분)

목적:
- 파인튜닝 효과 정량적/정성적 검증
- 학습 데이터와의 일치도 확인
- 과적합 여부 판단
"""

import os
import sys
import time
import json
import torch
import gc
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import psutil
import GPUtil

# 설정 파일 import
from config import *

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'step3_validate_results.log')
os.makedirs(LOG_DIR, exist_ok=True)
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        print(*args, **kwargs, file=f)

def load_finetuned_model():
    """파인튜닝된 모델 로드"""
    log_print("🤖 파인튜닝된 모델 로드 중...")
    
    lora_adapter_path = OUTPUT_DIR / "qwen3_lora_adapters"
    
    if not lora_adapter_path.exists():
        log_print("❌ LoRA 어댑터를 찾을 수 없습니다.")
        log_print("   먼저 2단계를 실행하세요: python3 step2_train_qlora.py")
        return None, None
    
    try:
        # BitsAndBytesConfig 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
    
        # 베이스 모델 로드
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
        # LoRA 어댑터 로드
        model = PeftModel.from_pretrained(model, str(lora_adapter_path))
        log_print(f"   ✅ LoRA 어댑터 로드 완료: {lora_adapter_path}")
    
        # 추론 모드로 설정
        model.eval()
        return model, tokenizer
    except Exception as e:
        log_print(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_test_data():
    """테스트 데이터 로드"""
    log_print("📊 테스트 데이터 로드 중...")
    
    # 절대 경로로 변경
    data_path = PROJECT_ROOT / "data" / "korean_tech_qa.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = []
    for item in data:
        messages = item['messages']
        system_prompt = messages[0]['content']
        user_question = messages[1]['content']
        expected_answer = messages[2]['content']
        
        test_cases.append({
            "system": system_prompt,
            "question": user_question,
            "expected": expected_answer
        })
    
    log_print(f"   로드된 테스트 케이스: {len(test_cases)}개")
    return test_cases

def generate_response(model, tokenizer, system_prompt, question, max_new_tokens=512):
    """응답 생성"""
    # Qwen3 chat template 사용
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 토크나이징
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    start_time = time.time()
    
    # 응답 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    
    # 응답 디코딩 (입력 프롬프트 제외)
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 토큰 수 계산
    num_tokens = len(generated_tokens)
    tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0
    
    return response.strip(), generation_time, tokens_per_second

def calculate_similarity(text1, text2):
    """텍스트 유사도 계산 (간단한 키워드 매칭)"""
    # 간단한 키워드 기반 유사도 (실제로는 더 정교한 방법 사용 가능)
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    jaccard_similarity = len(intersection) / len(union) if union else 0
    return jaccard_similarity

def test_training_data_accuracy(model, tokenizer, test_cases):
    """학습 데이터 정확도 테스트"""
    log_print("🎯 학습 데이터 정확도 테스트 중...")
    log_print("-" * 50)
    
    total_cases = len(test_cases)
    similarities = []
    response_times = []
    tokens_per_sec_list = []
    
    for i, case in enumerate(test_cases):
        log_print(f"\n📝 테스트 케이스 {i+1}/{total_cases}")
        log_print(f"질문: {case['question'][:50]}...")
        
        # 응답 생성
        response, gen_time, tokens_per_sec = generate_response(
            model, tokenizer, case['system'], case['question']
        )
        
        # 유사도 계산
        similarity = calculate_similarity(response, case['expected'])
        similarities.append(similarity)
        response_times.append(gen_time)
        tokens_per_sec_list.append(tokens_per_sec)
        
        log_print(f"생성 시간: {gen_time:.1f}초 ({tokens_per_sec:.2f} 토큰/초)")
        log_print(f"유사도: {similarity:.3f}")
        log_print(f"생성 응답: {response[:100]}...")
        log_print(f"기대 응답: {case['expected'][:100]}...")
    
    # 전체 통계
    avg_similarity = sum(similarities) / len(similarities)
    avg_response_time = sum(response_times) / len(response_times)
    avg_tokens_per_sec = sum(tokens_per_sec_list) / len(tokens_per_sec_list)
    
    log_print("\n" + "=" * 50)
    log_print("📊 학습 데이터 정확도 결과:")
    log_print(f"   평균 유사도: {avg_similarity:.3f} ({avg_similarity*100:.1f}%)")
    log_print(f"   평균 응답 시간: {avg_response_time:.1f}초")
    log_print(f"   평균 생성 속도: {avg_tokens_per_sec:.2f} 토큰/초")
    
    return {
        "avg_similarity": avg_similarity,
        "avg_response_time": avg_response_time,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "individual_similarities": similarities
    }

def test_generalization(model, tokenizer):
    """일반화 성능 테스트"""
    log_print("\n🔍 일반화 성능 테스트 중...")
    log_print("-" * 50)
    
    # 학습하지 않은 질문들
    unseen_questions = [
        {
            "system": "당신은 파이썬 전문가입니다. 명확하고 실용적인 답변을 제공하세요.",
            "question": "파이썬에서 제너레이터와 일반 함수의 차이점을 설명해주세요."
        },
        {
            "system": "당신은 머신러닝 전문가입니다. 실무에 도움되는 답변을 제공하세요.",
            "question": "딥러닝에서 배치 정규화(Batch Normalization)의 역할을 설명해주세요."
        },
        {
            "system": "당신은 웹개발 전문가입니다. 실용적이고 보안을 고려한 답변을 제공하세요.",
            "question": "웹 애플리케이션에서 CORS(Cross-Origin Resource Sharing) 문제를 해결하는 방법을 알려주세요."
        }
    ]
    
    generalization_results = []
    
    for i, case in enumerate(unseen_questions):
        log_print(f"\n🔍 일반화 테스트 {i+1}/{len(unseen_questions)}")
        log_print(f"질문: {case['question']}")
        
        response, gen_time, tokens_per_sec = generate_response(
            model, tokenizer, case['system'], case['question']
        )
        
        log_print(f"생성 시간: {gen_time:.1f}초 ({tokens_per_sec:.2f} 토큰/초)")
        log_print(f"응답: {response[:200]}...")
        
        # 한국어 품질 평가 (간단한 휴리스틱)
        korean_quality = evaluate_korean_quality(response)
        log_print(f"한국어 품질: {korean_quality}/5")
        
        generalization_results.append({
            "question": case['question'],
            "response": response,
            "generation_time": gen_time,
            "tokens_per_sec": tokens_per_sec,
            "korean_quality": korean_quality
        })
    
    return generalization_results

def evaluate_korean_quality(text):
    """한국어 품질 평가 (간단한 휴리스틱)"""
    score = 5  # 기본 점수
    
    # 한글 비율 확인
    korean_chars = sum(1 for char in text if '가' <= char <= '힣')
    total_chars = len([char for char in text if char.isalpha()])
    
    if total_chars > 0:
        korean_ratio = korean_chars / total_chars
        if korean_ratio < 0.3:
            score -= 2
        elif korean_ratio < 0.5:
            score -= 1
    
    # 기본적인 구조 확인
    if "**" in text or "```" in text:  # 마크다운 포맷 사용
        score += 1
    
    if len(text) < 50:  # 너무 짧은 답변
        score -= 1
    
    return max(1, min(5, score))

def benchmark_inference_speed(model, tokenizer):
    """추론 속도 벤치마크"""
    log_print("\n⚡ 추론 속도 벤치마크 중...")
    log_print("-" * 50)
    
    test_prompt = "당신은 AI 어시스턴트입니다."
    test_question = "인공지능에 대해 간단히 설명해주세요."
    
    # 여러 번 측정하여 평균 구하기
    times = []
    token_counts = []
    
    for i in range(5):
        log_print(f"   테스트 {i+1}/5 진행 중...")
        response, gen_time, tokens_per_sec = generate_response(
            model, tokenizer, test_prompt, test_question, max_new_tokens=100
        )
        times.append(gen_time)
        token_counts.append(len(tokenizer.encode(response)))
    
    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_speed = avg_tokens / avg_time if avg_time > 0 else 0
    
    log_print(f"   평균 생성 시간: {avg_time:.2f}초")
    log_print(f"   평균 토큰 수: {avg_tokens:.1f}")
    log_print(f"   평균 속도: {avg_speed:.2f} 토큰/초")
    
    return avg_speed

def save_validation_results(accuracy_results, generalization_results, benchmark_speed):
    """검증 결과 저장"""
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": accuracy_results,
        "generalization": generalization_results,
        "benchmark_speed": benchmark_speed,
        "model_config": {
            "model_name": MODEL_NAME,
            "lora_config": QLORA_CONFIG
        }
    }
    
    results_file = OUTPUT_DIR / "validation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    log_print(f"\n💾 검증 결과 저장: {results_file}")

def main():
    """메인 실행 함수"""
    log_print("🧪 3단계: 학습 결과 검증 시작")
    log_print("=" * 50)
    
    try:
        # 1. 파인튜닝된 모델 로드
        model, tokenizer = load_finetuned_model()
        if model is None:
            return False
        
        # 2. 테스트 데이터 로드
        test_cases = load_test_data()
        
        # 3. 학습 데이터 정확도 테스트
        accuracy_results = test_training_data_accuracy(model, tokenizer, test_cases)
        
        # 4. 일반화 성능 테스트
        generalization_results = test_generalization(model, tokenizer)
        
        # 5. 추론 속도 벤치마크
        benchmark_speed = benchmark_inference_speed(model, tokenizer)
        
        # 6. 결과 저장
        save_validation_results(accuracy_results, generalization_results, benchmark_speed)
        
        # 7. 최종 평가
        log_print("\n" + "=" * 50)
        log_print("📋 최종 검증 결과:")
        similarity_score = accuracy_results['avg_similarity'] * 100
        
        if similarity_score >= 90:
            log_print("   ✅ 학습 효과: 우수 (90%+)")
        elif similarity_score >= 70:
            log_print("   ⚠️ 학습 효과: 보통 (70-90%)")
        else:
            log_print("   ❌ 학습 효과: 부족 (<70%)")
        
        if benchmark_speed >= 10:
            log_print("   ✅ 추론 속도: 실용적 (10+ 토큰/초)")
        elif benchmark_speed >= 1:
            log_print("   ⚠️ 추론 속도: 느림 (1-10 토큰/초)")  
        else:
            log_print("   ❌ 추론 속도: 매우 느림 (<1 토큰/초)")
        
        log_print("\n📋 다음 단계 안내:")
        log_print("   - 4단계 실행: python3 step4_merge_adapters.py")
        
        return True
        
    except Exception as e:
        log_print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
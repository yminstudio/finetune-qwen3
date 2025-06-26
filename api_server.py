
#!/usr/bin/env python3
"""
FastAPI 래퍼 - Ollama 모델을 위한 RESTful API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import time
from typing import Optional
from config import MODEL_BASE_NAME

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
                "model": f"{MODEL_BASE_NAME.lower()}-finetune",
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
                model=f"{MODEL_BASE_NAME.lower()}-finetune"
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

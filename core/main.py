import os
import requests
import uvicorn
import modal
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Define Modal App
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")
app = modal.App(name="core-api", image=image)

# Define FastAPI App
fastapi_app = FastAPI()

# CORS Middleware for API Access
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@fastapi_app.get("/")
def home():
    return {"message": "Core services on Modal -- active!"}

# Define Request Model
class TextRequest(BaseModel):
    text: str

# External URL for Text Analysis API (Replace with your Modal endpoint)
TEXT_ANALYSIS_URL = "https://surbhitcodes--text-analysis-api-fastapi.modal.run"

@fastapi_app.post("/analyze-complexity")
def analyze_complexity_stream_relay(req: TextRequest):
    """Forwards request to text_analysis API"""
    headers = {"Content-Type": "application/json"}
    
    try:
        resp = requests.post(
            f"{TEXT_ANALYSIS_URL}/extract-analyze-complexity-stream",
            json={"text": req.text},
            headers=headers,
            stream=True
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to text_analysis API: {str(e)}"}

    def stream_llm_response():
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                yield line + "\n"

    return StreamingResponse(stream_llm_response(), media_type="application/json")

# Deploy FastAPI App to Modal
@app.function()
@modal.asgi_app()
def fastapi():
    return fastapi_app
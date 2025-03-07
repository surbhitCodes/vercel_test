
import os, requests, uvicorn, sys
from fastapi import (FastAPI, Depends)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import (BaseModel)
from dotenv import load_dotenv

# core api
app = FastAPI()

# for access config
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

@app.get("/")
def home():
    return {"message": "Core services on Fly.io -- active!"}

class TextRequest(BaseModel):
    text: str
    
# analyze-complexity --> text_analysis
@app.post("/analyze-complexity")
def analyze_complexity_stream_relay(req: TextRequest):
    headers = {
        "Content-Type": "application/json",
    }
    resp = requests.post(f"https://text-analysis-api.fly.dev/extract-analyze-complexity-stream", json={"text": req.text}, headers=headers, stream=True)
    # return resp.json()

    if resp.status_code != 200:
        return {"error": f"Error code {resp.status_code}, ", "details": resp.text}

    def stream_llm_response():
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                yield line + "\n"

    return StreamingResponse(stream_llm_response(), media_type="application/json")


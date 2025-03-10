import modal
from pydantic import BaseModel
import os, requests, json, sys, re, uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

class PassageRequest(BaseModel):
    text: str


app=modal.App(name="knowledge-graph-api", image=image)

# API setup
fastapi_app = FastAPI()
fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
@fastapi_app.get("/")
def index():
    return {
        "message":f"Knowledge Graph",
        "endpoints":{"Extract knowledge graph":"/extract-knowledge-graph"},
        "status":"live"
    }

API_KEY= os.getenv("API_KEY") # add api or env variable
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_MODEL = "gpt-4o"

@fastapi_app.post("/extract-knowledge-graph")
def get_knowledge_graph(req: PassageRequest):
    """
    Generate knowledge graphs using LLMs and stream the response.
    """
    prompt = f"""
    Create a knowledge graph for the given text by identifying entities, determining relationships, and classifying them.
    - Extract key entities (e.g., people, organizations, places).
    - Identify and classify the relationships between them.
    - Organize data hierarchically where applicable.

    The text: "{req.text}"
    """
    
    return StreamingResponse(stream_llm_chat_completion(prompt), media_type="application/json")

def stream_llm_chat_completion(prompt):
    """
    Stream LLM response structured as:
    - "start-tool"
    - "graph-delta"
    - "finish"
    """

    yield json.dumps({"type": "start-tool", "content": "knowledgeGraph"}) + "\n"

    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-type": "application/json"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": "You are an expert in knowledge extraction."},
                     {"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.3,
        "stream": True 
    }
    
    response = requests.post(url, headers=headers, json=payload, stream=True)

    # request fails, send an error message
    
    if response.status_code != 200:
        yield json.dumps({
            "type": "error",
            "status_code": response.status_code,
            "content": response.text
        }) + "\n"
        return  # end


    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8").strip()
            if decoded_line.startswith("data: "):  
                decoded_line = decoded_line[6:]  # Remove "data: " prefix
            
            if decoded_line == "[DONE]":
                break 
            
            try:
                data = json.loads(decoded_line)
                if "choices" in data and data["choices"]:
                    graph_content = data["choices"][0]["delta"].get("content", "")
                    yield json.dumps({"type": "graph-delta", "content": graph_content}) + "\n"
            except json.JSONDecodeError:
                continue  # ignore errors


    yield json.dumps({"type": "finish", "content": ""}) + "\n"


@app.function()
@modal.asgi_app()
def fastapi():
    return fastapi_app

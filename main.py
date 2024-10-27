import logging
import os
import random
import json
from urllib.parse import quote
from datetime import datetime, timezone
import time
import secrets
import string
import uuid
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp_sse_client import client as sse_client
from aiohttp import ClientTimeout

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, Response

import demjson3

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = FastAPI()

# Constants and Configuration
BASE_URL = "https://aichatonlineorg.erweima.ai/aichatonline"
APP_SECRET = os.getenv("APP_SECRET", "1")  # Replace with a secure default
ALLOWED_MODELS = [
    {"id": "gpt-4o", "name": "chat-gpt4"},
    {"id": "gpt-4o-mini", "name": "chat-gpt4m"},
    {"id": "o1-preview", "name": "chat-o1-preview"},
    {"id": "o1-mini", "name": "chat-o1-mini"},
    {"id": "gemini-1.5-flash", "name": "chat-gemini-flash"},
    {"id": "gemini-1.5-flash-exp-0827", "name": "chat-gemini-flash-exp"},
    {"id": "gemini-1.5-pro", "name": "chat-gemini-pro"},
    {"id": "gemini-1.5-pro-002", "name": "chat-gemini-pro-002"},
    {"id": "claude-3.5-sonnet", "name": "claude-sonnet"},
    {"id": "claude-3-haiku", "name": "claude-haiku"},
    {"id": "claude-3-opus", "name": "claude-opus"},
    {"id": "llama-3.1-70b", "name": "llama-3-70b"},
    {"id": "llama-3.1-8b", "name": "llama-3-8b"},
    {"id": "mistral-large2", "name": "mistral-large"},
]

# Configure CORS (Restrict origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Scheme
security = HTTPBearer()

# Pydantic Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False

# Utility Functions
def generate_random_string(length=21):
    chars = string.ascii_letters + string.digits + "_-"
    return ''.join(secrets.choice(chars) for _ in range(length))

def parse_js_object(data: str) -> Dict[str, Any]:
    try:
        return demjson3.decode(data)
    except demjson3.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return {}

def generate_cookie() -> str:
    current_timestamp = int(time.time())
    pk_id = f"{uuid.uuid4().hex[:16]}.{current_timestamp}"
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    
    initial_info = {
        "referrer": "direct",
        "date": current_time,
        "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "initialURL": "https://app.giz.ai/assistant?mode=chat",
        "browserLanguage": "zh-CN",
        "downlink": round(random.uniform(1.0, 2.0), 2),
        "pageTitle": "GizAI"
    }
    
    initial_info_json = json.dumps(initial_info, separators=(',', ':'))
    initial_info_encoded = quote(initial_info_json, safe='')
    cookie = f"_pk_id.1.2e21={pk_id}; _pk_ses.1.2e21=1; initialInfo={initial_info_encoded}"
    return cookie

def create_chat_completion_data(content: str, model: str, finish_reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": None,
    }

# Authentication Dependency
def verify_app_secret(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != APP_SECRET:
        logger.warning("Invalid APP_SECRET attempted.")
        raise HTTPException(status_code=403, detail="Invalid APP_SECRET")
    return credentials.credentials

# API Endpoints
@app.options("/hf/v1/chat/completions")
async def chat_completions_options():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )

@app.get("/hf/v1/models")
async def list_models():
    return {"object": "list", "data": ALLOWED_MODELS}

@app.post("/hf/v1/chat/completions")
async def chat_completions(
    request: ChatRequest, app_secret: str = Depends(verify_app_secret)
):
    logger.info(f"Received chat completion request for model: {request.model}")

    if request.model not in [model['id'] for model in ALLOWED_MODELS]:
        logger.error(f"Model {request.model} is not allowed.")
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} is not allowed. Allowed models are: {', '.join(model['id'] for model in ALLOWED_MODELS)}",
        )
    
    # Retrieve model name
    model_name = next((model["name"] for model in ALLOWED_MODELS if model["id"] == request.model), 'chat-gpt4')
    
    # Generate instanceId and full_cookie
    instance_id = generate_random_string()
    os.environ["instanceId"] = instance_id
    full_cookie = generate_cookie()
    os.environ["full_cookie"] = full_cookie

    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'Cookie': full_cookie,
        'Origin': 'https://app.giz.ai',
        'Pragma': 'no-cache',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0',
        'sec-ch-ua': '"Microsoft Edge";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }

    async def send_prompt_async(messages, model, headers):
        url = 'https://app.giz.ai/api/data/users/inferenceServer.infer'
        prompt = messages[-1].content

        input_messages = []

        if len(messages) > 1:
            input_messages.extend([
                {
                    "content": msg.content,
                    "type": "human" if msg.role == "user" else "ai" if msg.role == "assistant" else msg.role,
                    "unsaved": True
                } for msg in messages[:-1]
            ])

        # Add the last human message with internal instructions
        input_messages.append({
            "type": "human",
            "content": f'''{prompt}\n
                INTERNAL INSTRUCTIONS:\n
                * Use tools judiciously to fulfill the user's request effectively.\n
                * Each tool usage must be on a new line, starting with "$tools."\n
                * Always provide a clear response message along with any tool usage.\n
                * Format: $tools.toolName(parameters)\n
                $tools.generateImage("<A prompt that fits the response and describes the image style, historical, place setting and characters well for single image. Don't reuse prompts.>", "<Optional: UsingFace | SameShape>")\n
                $tools.generateVideo("<Prompt for generating a 2s video clip that fits the response in English>", "<Optional: InPainting | UsingFace>", "<Optional: A word of segment to replace with InPainting>", <Degree of segment change for InPainting>)\n
                $tools.generateAudio("<Prompt for generating a 10-second sound effect that fits the response in English. Use at the end of the response.>")
                $tools.addChoice(`<Provide a selectable response option for the user without number.>`)\n
                Use the tool only when explicitly requested by the user.''',
            "unsaved": True
        })

        if model == "chat-o1-mini":
            input_messages = [{
                "type": "human",
                "content": f"{prompt}",
                "unsaved": True
            }]

        data = {
            "model": "chat",
            "baseModel": model,
            "input": {
                "messages": input_messages,
                "prompt": f"{prompt}",
                "mode": "chat"
            },
            "subscribeId": generate_random_string(),
            "instanceId": instance_id
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    result = await response.text()
                    return result
        except aiohttp.ClientError as e:
            logger.error(f"Send prompt error: {e}")
            raise HTTPException(status_code=500, detail="Failed to send prompt to GizAI.")

    async def generate():
        url = f'https://app.giz.ai/api/notification/{instance_id}'
        timeout = ClientTimeout(total=120)
        async with sse_client.EventSource(url, headers=headers, timeout=timeout) as event_source:
            # Send the prompt
            response = await send_prompt_async(request.messages, model_name, headers)
            if not response:
                logger.error("No response from send_prompt_async")
                return

            # Process incoming events
            async for event in event_source:
                if event.data:
                    try:
                        data = parse_js_object(event.data)
                        if "message" in data and "output" in data["message"]:
                            output = data["message"]["output"]
                            yield f"data: {json.dumps(create_chat_completion_data(output, request.model))}\n\n"
                        elif "status" in data["message"] and data["message"]["status"] == "completed":
                            yield f"data: {json.dumps(create_chat_completion_data('', request.model, 'stop'))}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                        elif "url" in data:
                            markdown_url = f"\n ![]({data['url']}) \n"
                            yield f"data: {json.dumps(create_chat_completion_data(markdown_url, request.model))}\n\n"
                    except Exception as e:
                        logger.error(f"Failed to parse or process message data. Error: {e}")
                        logger.error(f"Problematic data: {event.data}")

    if request.stream:
        logger.info("Streaming response initiated.")
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        logger.info("Non-streaming response initiated.")
        full_response = ""
        async for chunk in generate():
            if chunk.startswith("data: ") and not chunk[6:].startswith("[DONE]"):
                try:
                    data = json.loads(chunk[6:])
                    if data["choices"][0]["message"].get("content"):
                        full_response += data["choices"][0]["message"]["content"]
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    continue

        return create_chat_completion_data(full_response, request.model, 'stop')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

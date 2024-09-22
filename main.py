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
from aiohttp_sse_client import client as sse_client
import aiohttp
from aiohttp import ClientTimeout
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, Response
import demjson3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()
BASE_URL = "https://aichatonlineorg.erweima.ai/aichatonline"
APP_SECRET = os.getenv("APP_SECRET","666")
ALLOWED_MODELS = [
    {"id": "gpt-4o", "name": "chat-gpt4"},
    {"id": "gpt-4o-mini", "name": "chat-gpt4m"},
    {"id": "o1-preview", "name": "chat-o1-preview"},
    {"id": "o1-mini", "name": "chat-o1-mini"},
    {"id": "gemini-1.5-flash", "name": "chat-gemini-flash"},
    {"id": "gemini-1.5-flash-exp-0827", "name": "chat-gemini-flash-exp"},
    {"id": "gemini-1.5-pro", "name": "chat-gemini-pro"},
    {"id": "gemini-1.5-pro-exp-0827", "name": "chat-gemini-pro-exp"},
    {"id": "claude-3.5-sonnet", "name": "claude-sonnet"},
    {"id": "claude-3-haiku", "name": "claude-haiku"},
    {"id": "claude-3-opus", "name": "claude-opus"},
    {"id": "llama-3.1-70b", "name": "llama-3-70b"},
    {"id": "llama-3.1-8b", "name": "llama-3-8b"},
    {"id": "mistral-large2", "name": "mistral-large"},
]
# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，您可以根据需要限制特定源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)
security = HTTPBearer()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False


def simulate_data(content, model):
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "role": "assistant"},
                "finish_reason": None,
            }
        ],
        "usage": None,
    }


def stop_data(content, model):
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }

    
def create_chat_completion_data(content: str, model: str, finish_reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "role": "assistant"},
                "finish_reason": finish_reason,
            }
        ],
        "usage": None,
    }


def verify_app_secret(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != APP_SECRET:
        raise HTTPException(status_code=403, detail="Invalid APP_SECRET")
    return credentials.credentials


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


def replace_escaped_newlines(input_string: str) -> str:
    return input_string.replace("\\n", "\n")


@app.get("/hf/v1/models")
async def list_models():
    return {"object": "list", "data": ALLOWED_MODELS}

def rV(e=21):
    chars = string.ascii_lowercase + string.digits + string.ascii_uppercase + "_-"
    return ''.join(secrets.choice(chars) for _ in range(e))

def parse_js_object(data):
    return demjson3.decode(data)


def generate_cookie():
    # 生成当前时间戳（以秒为单位）
    current_timestamp = int(time.time())

    # 生成随机的 pk_id
    pk_id = f"{uuid.uuid4().hex[:16]}.{current_timestamp}"

    # 生成当前UTC时间，格式化为正确的字符串
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # 生成 initialInfo 字典
    initial_info = {
        "referrer": "direct",
        "date": current_time,
        "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "initialURL": "https://app.giz.ai/assistant?mode=chat",
        "browserLanguage": "zh-CN",
        "downlink": round(random.uniform(1.0, 2.0), 2),
        "pageTitle": "GizAI"
    }

    # 将 initialInfo 转换为 JSON 字符串并进行正确的 URL 编码
    initial_info_json = json.dumps(initial_info, separators=(',', ':'))
    initial_info_encoded = quote(initial_info_json, safe='')

    # 构建完整的 cookie 字符串
    cookie = f"_pk_id.1.2e21={pk_id}; _pk_ses.1.2e21=1; initialInfo={initial_info_encoded}"

    return cookie


async def send_prompt_async(messages, model):
    url = 'https://app.giz.ai/api/data/users/inferenceServer.infer'
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        #'Cookie': '_pk_id.1=; _pk_ses.1=1; initialInfo=%7B%22referrer%22%3A%22direct%22%2C%22userAgent%22%3A%22Mozilla%2F5.0%20(Windows%20NT%2010.0%3B%20Win64%3B%20x64)%20AppleWebKit%2F537.36%20(KHTML%2C%20like%20Gecko)%20Chrome%2F129.0.0.0%20Safari%2F537.36%20Edg%2F129.0.0.0%22%2C%22initialURL%22%3A%22https%3A%2F%2Fapp.giz.ai%2Fassistant%3Fmode%3Dchat%22%2C%22browserLanguage%22%3A%22zh-CN%22%2C%22downlink%22%3A1.3%2C%22pageTitle%22%3A%22GizAI%22%7D',
        'Cookie': os.environ['full_cookie'],
        'Origin': 'https://app.giz.ai',
        'Pragma': 'no-cache',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0',
        'sec-ch-ua': '"Microsoft Edge";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }
    prompt = messages[-1].content

    # 构建消息列表
    input_messages = [
        {
            "content": f"Respond in language of \"{prompt}\" unless requested.",
            "type": "system",
            "unsaved": True
        }
    ]

    # 只有当 messages[:-1] 不为空时，才添加历史消息
    if len(messages) > 1:
        input_messages.extend([
            {
                "content": msg.content,
                "type": "human" if msg.role == "user" else "ai" if msg.role == "assistant" else msg.role,
                "unsaved": True
            } for msg in messages[:-1]
        ])

    # 添加最后的人类消息
    input_messages.append({
        "type": "human",
        "content": f"{prompt}\n* Use tools only when needed. Each tool should be finished one per line starting with $tools.\n$tools.generateImage(\"<A prompt that fits the response and describes the image style, historical, place setting and characters well for single image. Don't reuse prompts.>\", \"<Optional: UsingFace | SameShape>\");\n$tools.generateVideo(\"<Prompt for generating a 2s video clip that fits the response in English>\", \"<Optional: InPainting | UsingFace>\", \"<Optional: A word of segment to replace with InPainting>\", <Degree of segment change for InPainting>);\n$tools.generateAudio(\"<Prompt for generating a 10-second sound effect that fits the response in English. Use at the end of the response.>\");\n$tools.addChoice(`<Provide a selectable response option for the user without number.>`);\nAlways provide a response message too.\nIMPORTANT: NEVER USE TOOLS WITHOUT *EXPLICIT* USER REQUEST.",
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
        "subscribeId": f"{rV()}",
        "instanceId": f"{os.getenv('instanceId')}"
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                result = await response.text()
                response.raise_for_status()
                # 如果需要，你可以在这里处理响应
                # result = await response.json()
                # print("Response:", result)
    except aiohttp.ClientError as e:
        logger.error(f"Send prompt error: {e}\n{result}")
        # 可以选择在这里重新抛出异常，或者进行其他错误处理


@app.post("/hf/v1/chat/completions")
async def chat_completions(
    request: ChatRequest, app_secret: str = Depends(verify_app_secret)
):
    logger.info(f"Received chat completion request for model: {request.model}")

    if request.model not in [model['id'] for model in ALLOWED_MODELS]:
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} is not allowed. Allowed models are: {', '.join(model['id'] for model in ALLOWED_MODELS)}",
        )
    model_name = 'chat-gpt4'
    for model in ALLOWED_MODELS:
        if request.model == model["id"]:
            model_name = model["name"]

    os.environ["instanceId"] = rV()
    os.environ["full_cookie"] = generate_cookie()
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        #'Cookie': '_pk_id.1=; _pk_ses.1=1; initialInfo=%7B%22referrer%22%3A%22direct%22%2C%22userAgent%22%3A%22Mozilla%2F5.0%20(Windows%20NT%2010.0%3B%20Win64%3B%20x64)%20AppleWebKit%2F537.36%20(KHTML%2C%20like%20Gecko)%20Chrome%2F129.0.0.0%20Safari%2F537.36%20Edg%2F129.0.0.0%22%2C%22initialURL%22%3A%22https%3A%2F%2Fapp.giz.ai%2Fassistant%3Fmode%3Dchat%22%2C%22browserLanguage%22%3A%22zh-CN%22%2C%22downlink%22%3A1.3%2C%22pageTitle%22%3A%22GizAI%22%7D',
        'Cookie': os.environ["full_cookie"],
        'Origin': 'https://app.giz.ai',
        'Pragma': 'no-cache',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0',
        'sec-ch-ua': '"Microsoft Edge";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }

    async def generate():
        url = f'https://app.giz.ai/api/notification/{os.getenv("instanceId")}'
        # 设置超时时间，例如60秒
        timeout = ClientTimeout(total=120)
        async with sse_client.EventSource(url, headers=headers,timeout=timeout) as event_source:
            # 发送 prompt
            await send_prompt_async(request.messages, model_name)
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
        logger.info("Streaming response")
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        logger.info("Non-streaming response")
        full_response = ""
        async for chunk in generate():
            if chunk.startswith("data: ") and not chunk[6:].startswith("[DONE]"):
                # print(chunk)
                data = json.loads(chunk[6:])
                if data["choices"][0]["delta"].get("content"):
                    full_response += data["choices"][0]["delta"]["content"]
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_response},
                    "finish_reason": "stop",
                }
            ],
            "usage": None,
        }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

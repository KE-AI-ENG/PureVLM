
import os
import argparse
import base64
import time
import uuid
from io import BytesIO
from typing import Optional, List, Union

import torch
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from purevlm.engine import InferEngine

# 全局模型变量
inf_engine_ = None
model_path = None

# 创建 FastAPI 应用
app = FastAPI(
    title="Qwen3VL API Server",
    description="兼容 OpenAI Chat Completions API 的视觉语言模型推理服务",
    version="1.0.0"
)

# ==================== Pydantic 模型定义 ====================

class ImageUrl(BaseModel):
    """图片 URL 模型"""
    url: str
    detail: Optional[str] = "auto"

class ContentPart(BaseModel):
    """消息内容部分"""
    type: str  # "text" 或 "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    """聊天消息"""
    role: str  # "system", "user", "assistant"
    content: Union[str, List[ContentPart]]

class ChatCompletionRequest(BaseModel):
    """Chat Completions 请求模型"""
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=128, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    """响应选项"""
    index: int
    message: Message
    finish_reason: str  # "stop", "length", "content_filter"

class Usage(BaseModel):
    """Token 使用情况"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    """Chat Completions 响应模型"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

# ==================== 辅助函数 ====================

def load_model(model_dir: str, path_online_quant, max_seq_len: int = 4096) -> InferEngine:
    """加载模型到全局变量"""
    global inf_engine_, model_path
    if inf_engine_ is None or model_path != model_dir:
        print(f"正在加载模型ckpt路径: {model_dir}")
        inf_engine_ = InferEngine(ckpt_path=model_dir, max_seq_len=max_seq_len, path_online_quant=path_online_quant)
        model_path = model_dir
        print("模型加载完成!")
    return inf_engine_

def download_image(url: str) -> Image.Image:
    """从 URL 下载图片"""
    try:
        # 支持 base64 data URL
        if url.startswith('data:image'):
            # 格式: data:image/png;base64,iVBORw0KG...
            header, base64_data = url.split(',', 1)
            image_data = base64.b64decode(base64_data)
            return Image.open(BytesIO(image_data)).convert('RGB')
        
        # 支持 HTTP/HTTPS URL
        elif url.startswith('http://') or url.startswith('https://'):
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        
        # 支持本地文件路径
        elif os.path.exists(url):
            return Image.open(url).convert('RGB')
        
        else:
            raise ValueError(f"不支持的图片 URL 格式: {url}")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法加载图片: {str(e)}")

def parse_messages(messages: List[Message]) -> tuple[str, Optional[Image.Image], str]:
    """
    解析消息列表，提取系统提示、用户提示和图片
    
    返回: (system_prompt, image, user_prompt)
    """
    system_prompt = ""
    user_prompt = ""
    image = None
    
    for message in messages:
        role = message.role
        content = message.content
        
        if role == "system":
            # 处理系统消息
            if isinstance(content, str):
                system_prompt = content
            elif isinstance(content, list):
                for part in content:
                    if part.type == "text" and part.text:
                        system_prompt += part.text
        
        elif role == "user":
            # 处理用户消息
            if isinstance(content, str):
                user_prompt = content
            elif isinstance(content, list):
                for part in content:
                    if part.type == "text" and part.text:
                        user_prompt += part.text
                    elif part.type == "image_url" and part.image_url:
                        # 只取第一张图片
                        if image is None:
                            image = download_image(part.image_url.url)
    
    return system_prompt, image, user_prompt

def build_prompt(system_prompt: str, user_prompt: str, has_image: bool=True) -> str:
    """构建完整的提示词"""
    # 如果没有系统提示，使用默认的
    if not system_prompt:
        if has_image:
            system_prompt = (
                "You are a smart home assistant."
            )
        else:
            system_prompt = (
                "你是一个智能家居助手，根据用户提供的图片以及用户正在寻找物品，"
                "请仔细分析图片中的所有物品和位置，如果发现用户要找的物品，请详细描述其位置和周围环境。"
                "如果没有发现，回答没找到，要回答没找到原因。回答格式：是否找到，以及周围描述，原因。"
            )
    
    if has_image:
        user_content = f"<|vision_start|><|image_pad|><|vision_end|>{user_prompt}"
    else:
        user_content = user_prompt
    
    prompt = (
        f'<|im_start|>system\n{system_prompt}<|im_end|>\n'
        f'<|im_start|>user\n{user_content}<|im_end|>\n'
        f'<|im_start|>assistant\n'
    )
    return prompt

def generate_text(
    prompt: str,
    image: Image.Image,
    max_tokens: int = 128,
    temperature: float = 0.7,
    do_sample: bool = True
) -> tuple[str, int, int]:
    """生成文本"""
    global inf_engine_
    
    if inf_engine_ is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    # 生成文本
    with torch.no_grad():
        generated_ids, p_len, d_len = inf_engine_.generate(
            prompts=prompt,
            images=image,
            generated_len=max_tokens,  # 生成长度
            temperature=temperature,
            do_sample=do_sample
        )
    
    output_text = inf_engine_.tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if output_text else "" , p_len, d_len

# def estimate_tokens(text: str) -> int:
#     """估算 token 数量（简单估算）"""
#     # 简单估算：中文按字符数，英文按单词数
#     chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
#     english_words = len([w for w in text.split() if w.isascii()])
#     return chinese_chars + english_words

# ==================== API 端点 ====================

@app.on_event("startup")
async def startup_event():
    """服务启动时的初始化"""
    print("=" * 50)
    print("Qwen3VL API Server 启动中...")
    print("兼容 OpenAI Chat Completions API")
    print("=" * 50)

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Qwen3VL API Server",
        "status": "running",
        "model_loaded": inf_engine_ is not None,
        "api_version": "v1",
        "compatible_with": "OpenAI Chat Completions API"
    }

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen",
                "permission": [],
                "root": "qwen3",
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat Completions API 端点
    
    兼容 OpenAI 的 Chat Completions API 格式
    """
    try:
        # 解析消息
        system_prompt, image, user_prompt = parse_messages(request.messages)
        
        # 检查是否有图片
        # if image is None:
        #     raise HTTPException(
        #         status_code=400, 
        #         detail="请求中必须包含至少一张图片"
        #     )
        
        # 检查是否有用户提示
        if not user_prompt:
            raise HTTPException(
                status_code=400, 
                detail="请求中必须包含用户提示文本"
            )
        
        # 构建完整提示词
        full_prompt = build_prompt(system_prompt, user_prompt, has_image=image is not None)
        
        # 生成文本
        generated_text, prefill_token_len, decode_token_len = generate_text(
            prompt=full_prompt,
            image=image,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0
        )
        
        # 估算 token 使用量
        # prompt_tokens = estimate_tokens(full_prompt)
        # completion_tokens = estimate_tokens(generated_text)
        
        # 构建响应
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=generated_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prefill_token_len,
                completion_tokens=decode_token_len,
                total_tokens=prefill_token_len + decode_token_len
            )
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": inf_engine_ is not None,
        "timestamp": int(time.time())
    }

# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        prog='vlm-api-server',
        description='Qwen3VL API Server - 兼容 OpenAI Chat Completions API',
        epilog='Copyright(r), 2025'
    )
    parser.add_argument(
        '-m', '--model-path', 
        type=str, 
        required=True, 
        help='模型路径'
    )
    parser.add_argument(
        '--host', 
        type=str, 
        default='0.0.0.0', 
        help='服务器地址'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8002, 
        help='服务器端口'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=1, 
        help='工作进程数'
    )
    parser.add_argument(
        "-q", '--using-online-quant', 
        type=str, 
        default="", 
        help="Path for online quantization json. Not '' indicates using online quantization for base model, eg. Qwen3-VL-8B-Instruct"
    )
    parser.add_argument(
        '--max-seq-len', 
        type=int, 
        default=4096, 
        help="max sequence length for inference, include prefill-prompt and decode-generated, default 4096"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    load_model(args.model_path, args.using_online_quant, max_seq_len=args.max_seq_len)
    
    # 启动服务器
    print(f"\n启动 API Server:")
    print(f"  - 地址: http://{args.host}:{args.port}")
    print(f"  - Chat Completions: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  - 文档: http://{args.host}:{args.port}/docs")
    print(f"  - 工作进程: {args.workers}")
    print("\n")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )

if __name__ == "__main__":
    main()
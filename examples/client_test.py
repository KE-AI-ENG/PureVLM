
import requests
import json
import argparse
from pathlib import Path
from typing import Optional

def get_image_content(image_input: str) -> dict:
    """
    根据输入类型返回图片内容
    - 如果是 URL（http/https 开头），直接使用
    - 如果是本地文件路径，直接传绝对路径（不转 base64）
    """
    if image_input.startswith(('http://', 'https://')):
        # 网络图片 URL
        return {
            "type": "image_url",
            "image_url": {"url": image_input}
        }
    else:
        # 本地文件路径
        img_path = Path(image_input).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {img_path}")
        
        # 检查文件扩展名
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        if img_path.suffix.lower() not in valid_extensions:
            raise ValueError(f"不支持的图片格式: {img_path.suffix}，支持的格式: {valid_extensions}")
        
        # 直接传绝对路径
        return {
            "type": "image_url",
            "image_url": {"url": str(img_path)}
        }

def test_chat_completions(
    server_url: str,
    user_message: str,
    image_input: str=None,  # 可以是 URL 或本地路径
    system_message: Optional[str] = None,
    max_tokens: int = 128,
    temperature: float = 0.7
):
    """测试 Chat Completions API"""
    
    url = f"{server_url}/v1/chat/completions"
    
    # 构建消息
    messages = []
    
    # 添加系统消息
    if system_message:
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ]
        })
    
    if image_input is None:
        # 如果没有提供图片，直接添加用户消息
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_message}]
        })
        image_content = None
    else:
        # 获取图片内容（自动处理 URL 或本地文件）
        try:
            image_content = get_image_content(image_input)
        except Exception as e:
            print(f"错误: {e}")
            return
    
        # 添加用户消息（包含文本和图片）
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                image_content
            ]
        })
    
    # 构建请求
    payload = {
        "model": "qwen3",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # 发送请求
    print("发送请求...")
    print(f"URL: {url}")
    if image_content is not None:
        print(f"图片来源: {'URL' if image_input.startswith(('http://', 'https://')) else '本地文件'}")
        print(f"图片路径: {image_input}")
    
    
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    # 打印响应
    print("=" * 50)
    print("响应:")
    print("=" * 50)
    
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 提取生成的文本
        if result.get("choices"):
            generated_text = result["choices"][0]["message"]["content"]
            print("\n" + "=" * 50)
            print("生成的文本:")
            print("=" * 50)
            print(generated_text)
            print("=" * 50)
    else:
        print(f"错误: {response.status_code}")
        print(response.text)

def main():
    parser = argparse.ArgumentParser(description='测试 OpenAI 兼容的 Chat Completions API')
    parser.add_argument(
        '--server',
        type=str,
        default='http://localhost:8002',
        help='服务器地址'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='图片路径（支持 URL 或本地文件路径，如：./image.jpg 或 http://example.com/image.jpg）'
    )
    parser.add_argument(
        '--message',
        type=str,
        default='帮我找下音箱',
        help='用户消息'
    )
    parser.add_argument(
        '--system',
        type=str,
        help='系统消息'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=128,
        help='最大生成 token 数'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='温度参数'
    )
    
    args = parser.parse_args()
    
    # 默认系统消息
    if not args.system:
        args.system = (
            "你是一个智能家居助手，根据用户提供的图片以及用户正在寻找物品，"
            "请仔细分析图片中的所有物品和位置，如果发现用户要找的物品，请详细描述其位置和周围环境。"
            "如果没有发现，回答没找到，要回答没找到原因。回答格式：是否找到，以及周围描述，原因。"
        )
    
    test_chat_completions(
        server_url=args.server,
        image_input=args.image,
        user_message=args.message,
        system_message=args.system,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main()
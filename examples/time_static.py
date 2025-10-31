
import requests
import json
from pathlib import Path
import argparse
import time
from datetime import datetime

class Timer:
    """计时器类"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.checkpoints = {}
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self.start_time
    
    def checkpoint(self, name: str):
        """记录检查点"""
        if self.start_time is None:
            raise ValueError("计时器未启动")
        self.checkpoints[name] = time.time() - self.start_time
    
    def stop(self):
        """停止计时"""
        if self.start_time is None:
            raise ValueError("计时器未启动")
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def get_elapsed(self):
        """获取已用时间"""
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 1:
            return f"{seconds * 1000:.2f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.2f}s"
    
    def print_summary(self):
        """打印时间统计摘要"""
        print("\n" + "=" * 50)
        print("⏱️  耗时统计")
        print("=" * 50)
        
        if self.checkpoints:
            print("\n检查点:")
            prev_time = 0
            for name, elapsed in self.checkpoints.items():
                duration = elapsed - prev_time
                print(f"  • {name:20s}: {self.format_time(duration):>12s} (累计: {self.format_time(elapsed)})")
                prev_time = elapsed
        
        total_time = self.get_elapsed()
        print(f"\n总耗时: {self.format_time(total_time)}")
        print("=" * 50)

def test_chat_completions(
    server_url: str,
    image_url: str,
    user_message: str,
    system_message: str = None,
    max_tokens: int = 128,
    temperature: float = 0.7,
    verbose: bool = True
):
    """测试 Chat Completions API"""
    
    # 创建计时器
    timer = Timer()
    timer.start()
    
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
    
    # 添加用户消息（包含文本和图片）
    if image_url.startswith(('http://', 'https://')):
        img_path = image_url
    else:
        # 本地文件路径
        img_path = Path(image_url).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {img_path}")
        
        # 检查文件扩展名
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        if img_path.suffix.lower() not in valid_extensions:
            raise ValueError(f"不支持的图片格式: {img_path.suffix}，支持的格式: {valid_extensions}")
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_message},
            {"type": "image_url", "image_url": {"url": str(img_path)}}
        ]
    })
    
    # 构建请求
    payload = {
        "model": "qwen3",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    timer.checkpoint("构建请求")
    
    # 发送请求
    if verbose:
        print("=" * 50)
        print(f"🚀 发送请求")
        print("=" * 50)
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"URL: {url}")
        print(f"图片: {image_url}")
        print(f"消息: {user_message}")
        print(f"最大 tokens: {max_tokens}")
        print(f"温度: {temperature}")
        print()
    
    try:
        # 记录请求开始时间
        request_start = time.time()
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300  # 5分钟超时
        )
        
        # 记录请求结束时间
        request_time = time.time() - request_start
        timer.checkpoint("发送请求并接收响应")
        
        # 打印响应
        if verbose:
            print("=" * 50)
            print("📥 响应")
            print("=" * 50)
            print(f"状态码: {response.status_code}")
            print(f"请求耗时: {timer.format_time(request_time)}")
            print()
        
        if response.status_code == 200:
            result = response.json()
            timer.checkpoint("解析 JSON 响应")
            
            if verbose:
                print("完整响应:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 提取生成的文本
            if result.get("choices"):
                generated_text = result["choices"][0]["message"]["content"]
                
                print("\n" + "=" * 50)
                print("✨ 生成的文本")
                print("=" * 50)
                print(generated_text)
                print("=" * 50)
                
                # 提取 token 使用情况
                if result.get("usage"):
                    usage = result["usage"]
                    print("\n" + "=" * 50)
                    print("📊 Token 使用统计")
                    print("=" * 50)
                    print(f"  • 提示 tokens:  {usage.get('prompt_tokens', 0):>6d}")
                    print(f"  • 生成 tokens:  {usage.get('completion_tokens', 0):>6d}")
                    print(f"  • 总计 tokens:  {usage.get('total_tokens', 0):>6d}")
                    
                    # 计算生成速度
                    completion_tokens = usage.get('completion_tokens', 0)
                    if completion_tokens > 0 and request_time > 0:
                        tokens_per_second = completion_tokens / request_time
                        print(f"  • 生成速度:     {tokens_per_second:>6.2f} tokens/s")
                    print("=" * 50)
            
            timer.checkpoint("处理响应")
            
            # 打印耗时统计
            timer.stop()
            timer.print_summary()
            
            return result
        else:
            print(f"❌ 错误: {response.status_code}")
            print(response.text)
            timer.stop()
            timer.print_summary()
            return None
    
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        timer.stop()
        timer.print_summary()
        return None
    
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        timer.stop()
        timer.print_summary()
        return None
    
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        timer.stop()
        timer.print_summary()
        return None

def batch_test(
    server_url: str,
    test_cases: list,
    verbose: bool = False
):
    """批量测试"""
    print("\n" + "=" * 50)
    print(f"🔄 批量测试 ({len(test_cases)} 个测试用例)")
    print("=" * 50)
    
    results = []
    total_timer = Timer()
    total_timer.start()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 50}")
        print(f"测试用例 {i}/{len(test_cases)}")
        print(f"{'=' * 50}")
        
        result = test_chat_completions(
            server_url=server_url,
            image_url=test_case['image_url'],
            user_message=test_case['message'],
            system_message=test_case.get('system'),
            max_tokens=test_case.get('max_tokens', 128),
            temperature=test_case.get('temperature', 0.7),
            verbose=verbose
        )
        
        results.append({
            'case': test_case,
            'result': result,
            'success': result is not None
        })
    
    total_timer.stop()
    
    # 打印批量测试摘要
    print("\n" + "=" * 50)
    print("📈 批量测试摘要")
    print("=" * 50)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"总测试数: {len(test_cases)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(test_cases) - success_count}")
    print(f"成功率: {success_count / len(test_cases) * 100:.1f}%")
    print(f"总耗时: {total_timer.format_time(total_timer.get_elapsed())}")
    print(f"平均耗时: {total_timer.format_time(total_timer.get_elapsed() / len(test_cases))}")
    print("=" * 50)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='测试 OpenAI 兼容的 Chat Completions API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个测试
  python test_openai_api.py --image-url "https://example.com/image.jpg" --message "描述这张图片"
  
  # 批量测试
  python test_openai_api.py --batch test_cases.json
  
  # 简洁模式
  python test_openai_api.py --image-url "https://example.com/image.jpg" --message "描述这张图片" --quiet
        """
    )
    parser.add_argument(
        '--server',
        type=str,
        default='http://localhost:8002',
        help='服务器地址'
    )
    parser.add_argument(
        '--image-url',
        type=str,
        help='图片 URL'
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
    parser.add_argument(
        '--batch',
        type=str,
        help='批量测试配置文件 (JSON)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='简洁模式，只显示关键信息'
    )
    
    args = parser.parse_args()
    
    # 默认系统消息
    if not args.system:
        args.system = (
            "你是一个智能家居助手，根据用户提供的图片以及用户正在寻找物品，"
            "请仔细分析图片中的所有物品和位置，如果发现用户要找的物品，请详细描述其位置和周围环境。"
            "如果没有发现，回答没找到，要回答没找到原因。回答格式：是否找到，以及周围描述，原因。"
        )
    
    # 批量测试模式
    if args.batch:
        try:
            with open(args.batch, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
            batch_test(
                server_url=args.server,
                test_cases=test_cases,
                verbose=not args.quiet
            )
        except Exception as e:
            print(f"❌ 读取批量测试配置失败: {e}")
            return
    
    # 单个测试模式
    elif args.image_url:
        test_chat_completions(
            server_url=args.server,
            image_url=args.image_url,
            user_message=args.message,
            system_message=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=not args.quiet
        )
    else:
        parser.print_help()
        print("\n❌ 错误: 必须指定 --image-url 或 --batch")

if __name__ == "__main__":
    main()
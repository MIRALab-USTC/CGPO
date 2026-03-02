import json
import os
from collections import defaultdict
import math

from tqdm import tqdm

def count_token_distribution(jsonl_file, tokenizer_path, max_length=32768):
    """
    统计JSONL文件中对话的token长度分布
    
    参数:
    - jsonl_file: JSONL文件路径，包含messages字段
    - tokenizer_path: tokenizer路径
    - max_length: 最大统计长度
    
    返回:
    - prompt_stats: prompt token长度统计
    - response_stats: response token长度统计  
    - total_stats: 完整对话token长度统计
    """
    
    # 导入tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        raise Exception(f"无法加载tokenizer: {e}")
    
    # 初始化统计字典
    prompt_stats = defaultdict(int)
    response_stats = defaultdict(int)
    total_stats = defaultdict(int)
    
    # 定义统计区间（2的幂）
    intervals = [2**i for i in range(10, 20)]  # 1024, 2048, 4096, ..., 524288
    
    def get_interval(length):
        """获取长度对应的统计区间"""
        for interval in intervals:
            if length <= interval:
                return interval
        return max_length  # 超过最大值的归到一个区间
    
    # 读取并处理JSONL文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in tqdm(enumerate(f, 1)):
            try:
                data = json.loads(line)
                
                # 提取messages
                messages = data.get('messages', [])
                if not messages:
                    continue
                
                # 分离用户和助手消息
                user_messages = []
                assistant_messages = []
                
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_messages.append(msg.get('content', ''))
                    elif msg.get('role') == 'assistant':
                        assistant_messages.append(msg.get('content', ''))
                
                # 合并所有用户消息作为prompt
                prompt_text = ' '.join(user_messages)
                
                # 合并所有助手消息作为response
                response_text = ' '.join(assistant_messages)
                
                # 计算token长度
                prompt_tokens = len(tokenizer.encode(prompt_text))
                response_tokens = len(tokenizer.encode(response_text))
                total_tokens = prompt_tokens + response_tokens
                
                # 统计到对应区间
                prompt_stats[get_interval(prompt_tokens)] += 1
                response_stats[get_interval(response_tokens)] += 1
                total_stats[get_interval(total_tokens)] += 1
                
            except json.JSONDecodeError:
                print(f"警告: 第{line_num}行JSON解析失败")
                continue
            except Exception as e:
                print(f"警告: 第{line_num}行处理出错: {e}")
                continue
    
    return prompt_stats, response_stats, total_stats

def print_statistics(stats, title):
    """打印统计结果"""
    print(f"\n{title}统计:")
    print("-" * 30)
    sorted_stats = sorted(stats.items())
    for interval, count in sorted_stats:
        if interval == 32768:  # 最大值特殊标记
            print(f"超过524288: {count}")
        else:
            print(f"<={interval}: {count}")

# 使用示例
if __name__ == "__main__":
    # 示例用法
    jsonl_path = "/home/xzliang/General-Reasoner/data/AM-DeepSeek-R1-Distilled-1.4M/am_0.9M.jsonl"  # 替换为你的JSONL文件路径
    tokenizer_path = "/home/shared/xzliang/Qwen2.5-7B-Instruct"  # 替换为你的tokenizer路径
    
    try:
        prompt_stats, response_stats, total_stats = count_token_distribution(
            jsonl_path, tokenizer_path
        )
        
        print_statistics(prompt_stats, "Prompt Token")
        print_statistics(response_stats, "Response Token")
        print_statistics(total_stats, "Total Token")
        
    except Exception as e:
        print(f"处理出错: {e}")
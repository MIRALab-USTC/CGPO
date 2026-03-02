import pandas as pd
import os
import copy
from ipdb import set_trace as stc
import wandb
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import shutil
from transformers import AutoTokenizer

def plot_histogram(values, filename, bin_width=0.1):
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    bins = [i * bin_width for i in range(int(1/bin_width)+1)]  # e.g., [0,0.1,0.2,...,1.0]
    plt.figure(figsize=(8,4.5))
    plt.hist(values, bins=bins, edgecolor='black')
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title(f"Histogram of values with bin width {bin_width}")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def get_top_subset(filename, top_n, save_filename, key="qwen2.5_7b_pass_rate"):
    df = pd.read_parquet(filename)
    
    topn = df.nlargest(top_n, key)
    remaining = df.drop(topn.index)
    random_remaining = remaining.sample(int(top_n / 4), random_state=42)
    
    subset = pd.concat([topn, random_remaining], ignore_index=True)
    subset.to_parquet(save_filename, index=False)
    print("Done.")
    
def get_bottom_subset(filename, low_n, save_filename, key="qwen3_30b_pass_rate"):
    df = pd.read_parquet(filename)
    
    # 获取最低分的样本
    bottom_n = df.nsmallest(low_n, key)
    
    # 合并并保存
    
    bottom_n.to_parquet(save_filename, index=False)
    print("Done.")

def get_pass_rate_at_ratio(filename, ratio=0.5):
    # 读取 parquet 文件
    data = pd.read_parquet(filename)

    # 确认有这个 key
    if "qwen2.5_7b_pass_rate" not in data.columns:
        raise KeyError("列 'qwen2.5_7b_pass_rate' 不存在")

    # 排序
    sorted_values = data["qwen2.5_7b_pass_rate"].sort_values(ascending=False).reset_index(drop=True)

    # 计算位置
    index = int(len(sorted_values) * ratio)
    if index >= len(sorted_values):
        index = len(sorted_values) - 1  # 防止 ratio=1.0 越界
    # stc()
    return sorted_values.iloc[index]
    return sorted_values.iloc[index]["qwen2.5_7b_pass_rate"]

def read_jsonl(filename):
    data = []
    with open(filename, "r") as file:
        for line in file.readlines():
            data.append(json.loads(line))
        
    return data

def save_jsonl(data, filename, mode="w"):
    with open(filename, mode, encoding='utf-8') as file:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            file.write(json_line + "\n")

def sample_parquet(input_path, output_path, sample_ratio=0.1, random_seed=42):
    """
    从一个 Parquet 文件中随机抽取指定比例的数据，并保存为新的 Parquet 文件。

    参数：
    - input_path (str): 原始 Parquet 文件路径
    - output_path (str): 保存抽样后数据的 Parquet 文件路径
    - sample_ratio (float): 抽样比例，范围为 0~1（如 0.1 表示抽取 10%）
    - random_seed (int): 随机种子，确保可重复性
    """
    if not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio 应该在 (0, 1] 范围内")

    # 读取原始 parquet 文件
    df = pd.read_parquet(input_path)

    # 抽样
    sampled_df = df.sample(frac=sample_ratio, random_state=random_seed)

    # 保存为新的 parquet 文件
    sampled_df.to_parquet(output_path, index=False)

    print(f"已从 {input_path} 中抽取 {len(sampled_df)} 条记录，保存至 {output_path}")



def filter_and_sample_parquet(input_path, output_path, k, seed=42):
    # 读 parquet 文件
    df = pd.read_parquet(input_path)

    # 选出 >= 0.25 的数据
    high_pass = df[df["qwen2.5_7b_pass_rate"] >= 0.25]

    # 选出 < 0.25 的数据
    low_pass = df[df["qwen2.5_7b_pass_rate"] < 0.25]

    # 从低分数据中随机抽样 k 个
    sampled_low = low_pass.sample(n=min(k, len(low_pass)), random_state=seed)

    # 合并
    result = pd.concat([high_pass, sampled_low], ignore_index=True)

    # 保存到新的 parquet 文件
    result.to_parquet(output_path, index=False)

    print(f"处理完成！共保存 {len(result)} 条数据到 {output_path}")

def added_markdown_codeblock(filename, backup=True):
    """
    修改 parquet 文件中 prompt[0]['content'] 的结尾字符串，并保存回文件。
    
    参数：
        filename: str, parquet 文件路径
        backup: bool, 是否在修改前备份原文件
    """
    # 备份原文件
    if backup:
        backup_path = filename + ".bak"
        shutil.copyfile(filename, backup_path)
        print(f"📦 已备份原文件到 {backup_path}")

    # 读取 parquet
    data = pd.read_parquet(filename, engine="pyarrow")  # pyarrow 或 fastparquet 均可

    # 遍历每行修改
    count = 0
    for i in range(len(data)):
        prompt = data.iloc[i]['prompt']  # ndarray of dict
        if isinstance(prompt, (list, np.ndarray)) and len(prompt) > 0:
            content = prompt[0].get('content', '')
            if isinstance(content, str) and content.endswith(
                "Now solve the problem and return the code."
            ):
                count += 1
                prompt[0]['content'] = content.replace(
                    "Now solve the problem and return the code.",
                    "Now solve the problem and return your code in a Python markdown code block."
                )

    # 保存回 parquet
    data.to_parquet(filename, index=False, engine="pyarrow")
    print(f"✅ 有{count}行被修改，已保存修改后的 parquet 文件: {filename}")

def get_rank_ratio(filename, value):
    # 读取 parquet 文件
    data = pd.read_parquet(filename)

    if "qwen2.5_7b_pass_rate" not in data.columns:
        raise KeyError("列 'qwen2.5_7b_pass_rate' 不存在")

    col = data["qwen2.5_7b_pass_rate"].dropna()

    # 从高到低排序
    sorted_values = col.sort_values(ascending=False).reset_index(drop=True)

    # 如果 value 在数据中，直接取排名
    if value in sorted_values.values:
        index = sorted_values[sorted_values == value].index[0]
        used_value = value
    else:
        # 如果 value 不在数据中，找最接近的值
        diffs = np.abs(sorted_values.values - value)
        closest_idx = np.argmin(diffs)
        index = closest_idx
        used_value = sorted_values.iloc[closest_idx]

    # 计算比例（排名 / 总数）
    ratio = (index + 1) / len(sorted_values)  # +1 因为 index 从 0 开始

    return ratio

def get_sorted_pass_rates(files, key="qwen2.5_7b_pass_rate"):
    all_values = []

    for f in files:
        df = pd.read_parquet(f)
        
        if key not in df.columns:
            raise KeyError(f"文件 {f} 中没有 {key} 列")

        # 去掉缺失值
        values = df[key].dropna().tolist()
        all_values.extend(values)

    # 从大到小排序
    sorted_values = sorted(all_values, reverse=True)
    return sorted_values

def keep_first_prompt_element(filename, backup=True):
    """
    处理 parquet 文件中的 prompt 字段，只保留每个 prompt 数组中的第一个元素。
    保持 prompt 为 numpy 数组格式。
    
    参数：
        filename: str, parquet 文件路径
        backup: bool, 是否在修改前备份原文件
    """
    import pandas as pd
    import shutil
    import numpy as np
    
    # 备份原文件
    if backup:
        backup_path = filename + ".bak"
        shutil.copyfile(filename, backup_path)
        print(f"📦 已备份原文件到 {backup_path}")
    
    # 读取 parquet 文件
    data = pd.read_parquet(filename, engine="pyarrow")
    
    # 处理每一行的 prompt 字段
    count = 0
    for i in range(len(data)):
        prompt = data.iloc[i]['prompt']
        # 检查 prompt 是否为 numpy 数组且至少有一个元素
        if isinstance(prompt, (list, tuple, np.ndarray)) and len(prompt) > 0:
            # 只保留第一个元素，并确保它仍然是 numpy 数组
            first_element = prompt[0]
            data.at[i, 'prompt'] = np.array([first_element])  # 转换为只包含一个元素的 numpy 数组
            count += 1
    
    # 保存回原文件
    data.to_parquet(filename, index=False, engine="pyarrow")
    print(f"✅ 处理完成，共处理了 {count} 行数据，已保存到 {filename}")


def filter_by_pass_rate(filename, save_filename, threshold=0.25):
    """
    筛选 parquet 文件中 qwen3_30b_pass_rate 小于或等于 threshold 的行，并保存到新文件中
    
    参数:
        filename (str): 输入的 parquet 文件路径
        save_filename (str): 输出的 parquet 文件路径
        threshold (float): qwen3_30b_pass_rate 的阈值，默认为 0.25
    """
    # 读取 parquet 文件
    df = pd.read_parquet(filename)
    
    # 检查是否存在 qwen3_30b_pass_rate 列
    if "qwen3_30b_pass_rate" not in df.columns:
        raise KeyError(f"列 'qwen3_30b_pass_rate' 在文件 {filename} 中不存在")
    
    # 筛选 qwen3_30b_pass_rate 小于或等于 threshold 的行
    filtered_df = df[df["qwen3_30b_pass_rate"] <= threshold]
    
    # 保存到新的 parquet 文件
    filtered_df.to_parquet(save_filename, index=False)
    
    print(f"原始文件 {filename} 中共有 {len(df)} 行数据")
    print(f"筛选出 {len(filtered_df)} 行 qwen3_30b_pass_rate <= {threshold} 的数据")
    print(f"结果已保存至 {save_filename}")
    
    # return filtered_df

def process_wildchat(filename, final_filename, data_source="creative__writing__WildChat", ability="creative_writing"):
    data = pd.read_parquet(filename)
    stc()
    all_processed_data = []
    for i in range(len(data)):
        entry = data.iloc[i]
        messages = entry["messages"].tolist()
        
        prompt = [messages[0]]
        
        metadata = entry["metadata"]
        question_id = metadata["question_id"]
        processed_data = {
            "data_source": data_source,
            "prompt": prompt,
            "ability": ability,
            "apply_chat_template": True,
            "reward_model": {
                "style": "model",
                "ground_truth": messages[-1]["content"],
            },
            "extra_info": {
                "question_id": question_id,
                "question": messages[0]["content"]
            },
        }
        all_processed_data.append(processed_data)
    
    df = pd.DataFrame(all_processed_data)
    df.to_parquet(final_filename, engine='pyarrow', index=False)
    print(f"Parquet文件已保存至 {final_filename}")
    
def process_litbench(filename, final_filename, data_source="creative__writing__LitBench", ability="creative_writing"):
    data = pd.read_parquet(filename)
    
    all_processed_data = []
    for i in range(len(data)):
        entry = data.iloc[i]
        
        prompt = entry["prompt"]
        processed_prompt = prompt
        if prompt[0] == "[":
            processed_prompt = prompt[4:].strip()
        prompt = [
            {
                "role": "user",
                "content": processed_prompt,
            },
        ]
        
        processed_data = {
            "data_source": data_source,
            "prompt": prompt,
            "ability": ability,
            "apply_chat_template": True,
            "reward_model": {
                "style": "model",
                "ground_truth": entry["chosen_story"],
            },
            "extra_info": {
                "index": i,
                "question": processed_prompt,
            },
        }
        all_processed_data.append(processed_data)
    
    df = pd.DataFrame(all_processed_data)
    df.to_parquet(final_filename, engine='pyarrow', index=False)
    print(f"Parquet文件已保存至 {final_filename}")


def process_am(filename, final_filename):
    data = read_jsonl(filename)
    final_list = []
    for entry in data:
        final_list.append(
            {
                "messages":[
                    {
                        "role": "user",
                        "content": entry["messages"][0]["content"],
                    },
                    {
                        "role": "assistant",
                        "content": entry["messages"][1]["content"],
                    },
                ]
            },
        )
        
    save_jsonl(final_list, final_filename)

tokenizer = AutoTokenizer.from_pretrained("/home/shared/xzliang/Qwen2.5-7B-Instruct", trust_remote_code=True)
data = read_jsonl("/home/xzliang/General-Reasoner/data/AM-DeepSeek-R1-Distilled-1.4M/am_0.9M.jsonl")

final_list = []
lengths = []

def get_thought_length(response, tokenizer):
    start_tag = "<think>"
    end_tag = "</think>"
    start_idx = response.find(start_tag)
    end_idx = response.find(end_tag)

    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        raise ValueError("未正确找到<think>或</think>标签")

    
    thought_content = response[start_idx + len(start_tag):end_idx]

    
    encoded = tokenizer.encode(thought_content)
    return len(encoded)

for entry in tqdm(data):
    temp = {
        "messages": [
            {
                "role": "user",
                "content": entry["messages"][0]["content"],
            },
            {
                "role": "assistant",
                "content": entry["messages"][1]["content"],
            },
        ]
    }
    response = entry["messages"][1]["content"]
    length = get_thought_length(response, tokenizer)
    lengths.append(length)
    final_list.append(temp)

save_jsonl(final_list, "/home/xzliang/General-Reasoner/data/AM-DeepSeek-R1-Distilled-1.4M/am_0.9M_processed.jsonl")

with open("/home/xzliang/General-Reasoner/data/AM-DeepSeek-R1-Distilled-1.4M/am_0.9M_lengths.json", "w", encoding="utf-8") as file:
    json.dump(lengths, file, ensure_ascii=False, indent=4)

stc()
# data = read_jsonl("/home/xzliang/General-Reasoner/vllm_inference/results/aime2024/qwen2.5-7b-instruct-am-deepseek-distill-0.9m-sft-checkpoint1600-TEMP0.6-TOPP0.95-SEED0-REP4-MAXTOKEN16384.jsonl")

# for item in data:
#     print(item["completion"])
#     input("Press Enter to continue...")

# stc()
# files = [
#     "/home/xzliang/General-Reasoner-Qwen3/data/guru-RL-92k/train/codegen__leetcode2k_1.3k.parquet",
#     "/home/xzliang/General-Reasoner-Qwen3/data/guru-RL-92k/train/codegen__livecodebench_440.parquet",
#     "/home/xzliang/General-Reasoner-Qwen3/data/guru-RL-92k/train/codegen__primeintellect_7.5k.parquet",
#     "/home/xzliang/General-Reasoner-Qwen3/data/guru-RL-92k/train/codegen__taco_8.8k.parquet",
# ]

# filter_by_pass_rate(
#     filename="/home/xzliang/General-Reasoner-Qwen3/data/guru-RL-92k/train/codegen__taco_8.8k.parquet",
#     save_filename="/home/xzliang/General-Reasoner-Qwen3/data/guru-RL-92k/train/codegen__taco_hard_8.8k.parquet",
#     threshold=0.5,
# )

# files = [
#     "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/codegen__leetcode2k_easy_0.9k.parquet",
#     "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/codegen__livecodebench_easy_145.parquet",
#     "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/codegen__primeintellect_easy_2k.parquet",
#     "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/codegen__taco_easy_1.5k.parquet",
# ]

# res = get_sorted_pass_rates(files)


stc()

# total_len = 0
# for file in files:
#     data = pd.read_parquet(file)
#     total_len += len(data)

# print(total_len)


# files = [
#     "/home/xzliang/General-Reasoner-Qwen3/data/guru-RL-92k/train/math__combined_54.4k.parquet",
# ]

# res = get_sorted_pass_rates(files)
# res = np.array(res)

# for i in range(1, 17):
#     print(f"{i*1/16}: {(res==i*1/16).sum()}")

# get_bottom_subset(
#     filename="/home/xzliang/General-Reasoner-Qwen3/data/guru-RL-92k/train/math__combined_54.4k.parquet",
#     low_n=6250,
#     save_filename="/home/xzliang/General-Reasoner-Qwen3/data/guru-RL-92k/train/math__combined_hard_6.25k.parquet",
#     key="qwen3_30b_pass_rate",
# )

stc()

# data = read_jsonl("/home/xzliang/General-Reasoner/question_and_response/qwen2.5-7b-instruct-guru-math-creative6k-easy_250902093233/creative.jsonl")

# sample_parquet(
#     input_path="/home/xzliang/General-Reasoner/data/guru-RL-92k/train_sanity/stem__web_3k.parquet",
#     output_path="/home/xzliang/General-Reasoner/data/guru-RL-92k/train_sanity/stem__web_0.3k.parquet",
#     sample_ratio=0.3/3,
#     random_seed=42,
# )

# process_am("/home/xzliang/General-Reasoner/data/AM-DeepSeek-R1-Distilled-1.4M/am_0.9M_sample_1k.jsonl", "/home/xzliang/General-Reasoner/data/AM-DeepSeek-R1-Distilled-1.4M/am_0.9M_sample_1k_processed.jsonl")

stc()

# sample_parquet(
#     input_path="/home/xzliang/General-Reasoner/data/guru-RL-92k/train/creative__wildchat_6.0k.parquet",
#     output_path="/home/xzliang/General-Reasoner/data/guru-RL-92k/train/creative__wildchat_2.0k.parquet",
#     sample_ratio=2/6,
#     random_seed=42,
# )

# data = read_jsonl("/home/xzliang/General-Reasoner/question_and_response/qwen2.5-7b-instruct-guru-math-codegen-stem-easy_250822083947/codegen.jsonl")

# process_wildchat("/home/xzliang/General-Reasoner/data/wildchat-creative-writing-3k-rft/ranking-definitive/train-00000-of-00001.parquet", "/home/xzliang/General-Reasoner/data/wildchat-creative-writing-3k-rft/processed.parquet")

# data = pd.read_parquet("/home/xzliang/General-Reasoner/data/LitBench-Train/data/train-00000-of-00001.parquet")

# process_litbench("/home/xzliang/General-Reasoner/data/LitBench-Train/data/train-00000-of-00001.parquet", "/home/xzliang/General-Reasoner/data/LitBench-Train/data/creative__litbench_43k.parquet")

# first_four_letters = set()
# for i in range(len(data)):
#     entry = data.iloc[i]
#     prompt = entry["prompt"]
#     if prompt[0] != "[":
#         stc()
    
#     first_four_letters.add(prompt[:4])

stc()

# data = pd.read_parquet("/home/xzliang/General-Reasoner/data/wildchat-creative-writing-3k-rft/ranking-definitive/train-00000-of-00001.parquet", "/home/xzliang/General-Reasoner/data/wildchat-creative-writing-3k-rft/processed.parquet",)

# count = {
#     1: 0,
#     2: 0,
#     3: 0,
#     4: 0,
#     -1: 0,
# }
# for i in range(len(data)):
#     if len(data.iloc[i]["messages"]) <= 4:
#         count[len(data.iloc[i]["messages"])] += 1
#     else:
#         count[-1] += 1
        

# data = read_jsonl("/home/xzliang/General-Reasoner/data/AM-DeepSeek-R1-Distilled-1.4M/am_0.5M.jsonl")

stc()

# data = read_jsonl("/home/xzliang/General-Reasoner/data/Creative_Writing-ShareGPT/Creative_Writing-ShareGPT.jsonl")
# keep_first_prompt_element("/home/xzliang/General-Reasoner/data/Creative_Writing-ShareGPT/output_processed_data_no_filter.parquet", backup=True)
# for entry in data:
#     conv = entry["conversations"]
#     if len(conv) > 2:
#         print(f"Human: {conv[0]['value']}\n\n")
#         print(f"GPT: {conv[1]['value']}\n\n")
#         print(f"Human: {conv[2]['value']}\n\n")
#         print(f"GPT: {conv[3]['value']}\n\n")
#         stc()

stc()

# filter_and_sample_parquet(
#     input_path="/home/xzliang/General-Reasoner/data/guru-RL-92k/train/codegen__taco_8.8k.parquet",
#     output_path="/home/xzliang/General-Reasoner/data/guru-RL-92k/train/codegen__taco_easy.parquet",
#     k=463,
# )
# stc()
# get_top_subset(
#     "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/math__combined_54.4k.parquet",
#     top_n=14000,
#     save_filename="/home/xzliang/General-Reasoner/data/guru-RL-92k/train/math__combined_easy14k.parquet",
# )

# filename = "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/math__combined_54.4k.parquet"
# value = 0.25
# res = get_rank_ratio(filename, value)
# print(f"{res.item()=}")
# stc()
# filenames = [
#     "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/codegen__leetcode2k_1.3k.parquet",
#     "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/codegen__livecodebench_440.parquet",
#     "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/codegen__primeintellect_7.5k.parquet",
#     "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/codegen__taco_8.8k.parquet",
#     "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/stem__web_3.6k.parquet",
# ]

# filenames_logic = [
#     # "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/logic__arcagi1_111.parquet",
#     # "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/logic__arcagi2_190.parquet",
#     # "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/logic__barc_1.6k.parquet",
#     # "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/logic__graph_logical_1.2k.parquet",
#     # "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/logic__ordering_puzzle_1.9k.parquet",
#     # "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/logic__zebra_puzzle_1.3k.parquet",
# ]

# for filename in tqdm(filenames_logic):

#     data = pd.read_parquet(filename)

#     for i in range(len(data)):
#         entry = data.iloc[i]
#         if "</answer>" in entry["prompt"][0]["content"] or "<answer>" in entry["prompt"][0]["content"]:
#             stc()
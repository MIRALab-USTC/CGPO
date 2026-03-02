from transformers import AutoTokenizer
from ipdb import set_trace as stc

import pandas as pd
import numpy as np

from omegaconf import ListConfig
import os
from typing import List, Union, Optional, Callable
import copy


from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key: str = 'prompt',
                 image_key: str = 'images',
                 max_prompt_length: int = 1024,
                 cache_dir: str = '~/.cache/verl/rlhf',
                 chat_template_func: Optional[Callable] = None,
                 return_raw_chat: bool = False,
                 truncation: str = 'error',
                 filter_overlong_prompts: bool = False,
                 num_workers: Optional[int] = None):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]
        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts
        if num_workers is None:
            self.num_workers = max(1, os.cpu_count() // 4)
        else:
            self.num_workers = min(num_workers, os.cpu_count())

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        
        # 参考 verl 0.3
        for parquet_file in self.parquet_files:
            try:
                dataframe = pd.read_parquet(parquet_file)
            except Exception:
                # NOTE: added by Reasoning360
                import polars as pl
                dataframe = pl.read_parquet(parquet_file).to_pandas()
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        # for parquet_file in self.parquet_files:
        #     # read parquet files and cache
        #     dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
        #     dataframes.append(dataframe)
        # from ipdb import set_trace as stc; stc()
        # self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        # Safely check if apply_chat_template exists in dataframe
        # NOTE: added by Reasoning360
        if "apply_chat_template" not in self.dataframe:
            print("Warning: apply_chat_template column not found in dataframe. Defaulting to True.")
            self.dataframe["apply_chat_template"] = [True] * len(self.dataframe)
        
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe[
                self.dataframe.apply(
                    lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True) if doc["apply_chat_template"] else tokenizer.encode(doc["raw_prompt"])) <= self.max_prompt_length,
                    axis=1,
                )
            ]

            print(f"filter dataset len: {len(self.dataframe)}")
        
        # if self.filter_overlong_prompts:
        #     tokenizer = self.tokenizer
        #     prompt_key = self.prompt_key
        #     self.dataframe = self.dataframe.filter(
        #         lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)
        #                        ) <= self.max_prompt_length,
        #         num_proc=self.num_workers,
        #         desc=f"Filtering prompts longer than {self.max_prompt_length} tokens")

        #     print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()

def sort_and_shuffle_dataset(train_dataset):
    """
    按照 qwen2.5_7b_pass_rate 从高到低排序，将 NaN 值样本随机插入
    
    Args:
        train_dataset: RLHFDataset 实例
        
    Returns:
        排序并随机插入 NaN 样本后的新 RLHFDataset 实例
    """
    
    # 直接使用 train_dataset.dataframe
    df = train_dataset.dataframe.copy()
    
    # 分离有有效 pass_rate 和 NaN pass_rate 的样本
    valid_mask = pd.notna(df['qwen2.5_7b_pass_rate'])
    valid_samples = df[valid_mask].copy()
    nan_samples = df[~valid_mask].copy()
    
    # 对有效样本按 qwen2.5_7b_pass_rate 从高到低排序
    valid_samples_sorted = valid_samples.sort_values(
        by='qwen2.5_7b_pass_rate', 
        ascending=False
    ).reset_index(drop=True)
    
    # 如果没有 NaN 样本，直接返回基于排序后数据的新数据集
    if len(nan_samples) == 0:
        # 创建新的 RLHFDataset 实例
        new_dataset = copy_dataset_structure(train_dataset)
        new_dataset.dataframe = valid_samples_sorted
        return new_dataset
    
    # 如果所有样本都是 NaN，直接返回原数据集的随机排列
    if len(valid_samples) == 0:
        new_dataset = copy_dataset_structure(train_dataset)
        new_dataset.dataframe = df.sample(frac=1).reset_index(drop=True)
        return new_dataset
    
    # 将 NaN 样本随机打乱
    nan_samples_shuffled = nan_samples.sample(frac=1).reset_index(drop=True)
    
    # 在排序后的有效样本中随机选择插入位置
    # 生成插入位置索引（包括开头和结尾）
    insertion_points = np.random.choice(
        len(valid_samples_sorted) + 1,  # +1 表示可以在末尾插入
        size=len(nan_samples_shuffled),
        replace=True  # 允许在相同位置插入多个样本
    )
    
    # 按插入位置排序，确保按顺序插入
    insertion_order = np.argsort(insertion_points)
    insertion_points = insertion_points[insertion_order]
    nan_samples_shuffled = nan_samples_shuffled.iloc[insertion_order].reset_index(drop=True)
    
    # 构建最终数据框
    result_dfs = []
    valid_idx = 0
    nan_idx = 0
    
    # 遍历每个可能的插入点
    for i in range(len(valid_samples_sorted) + len(nan_samples_shuffled)):
        # 插入所有应该在此位置插入的 NaN 样本
        while nan_idx < len(insertion_points) and insertion_points[nan_idx] == i:
            result_dfs.append(nan_samples_shuffled.iloc[nan_idx:nan_idx+1])
            nan_idx += 1
        
        # 如果还有有效样本未添加，则添加下一个有效样本
        if valid_idx < len(valid_samples_sorted):
            result_dfs.append(valid_samples_sorted.iloc[valid_idx:valid_idx+1])
            valid_idx += 1
    
    # 处理剩余的 NaN 样本（如果有的话）
    while nan_idx < len(nan_samples_shuffled):
        result_dfs.append(nan_samples_shuffled.iloc[nan_idx:nan_idx+1])
        nan_idx += 1
    
    # 合并所有数据框
    final_df = pd.concat(result_dfs, ignore_index=True)
    
    # 创建新的 RLHFDataset 实例
    new_dataset = copy_dataset_structure(train_dataset)
    new_dataset.dataframe = final_df
    
    return new_dataset

def copy_dataset_structure(original_dataset):
    """
    复制 RLHFDataset 的结构，但不复制实际数据
    
    Args:
        original_dataset: 原始 RLHFDataset 实例
        
    Returns:
        新的 RLHFDataset 实例（不包含实际数据）
    """
    # 创建新的数据集实例
    new_dataset = RLHFDataset.__new__(RLHFDataset)
    
    # 复制所有属性，除了 dataframe
    for key, value in original_dataset.__dict__.items():
        if key != 'dataframe':
            setattr(new_dataset, key, value)
    
    return new_dataset


def sort_by_prompt_length_per_category_then_interleave(train_dataset):
    """
    按照 data_source 分类，每类内部按 raw_prompt_ids 长度排序，最后将四个类别穿插混合
    
    Args:
        train_dataset: RLHFDataset 实例
        
    Returns:
        按照分类排序后的新 RLHFDataset 实例
    """
    # 强制生成所有样本的 raw_prompt_ids
    print("Generating raw_prompt_ids for all samples...")
    raw_prompt_ids_list = []
    
    # 通过访问每个样本确保 raw_prompt_ids 被生成并存储
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        raw_prompt_ids_list.append(sample['raw_prompt_ids'])
    
    # 更新 dataframe
    df = train_dataset.dataframe.copy()
    df['raw_prompt_ids'] = raw_prompt_ids_list
    
    # 计算每个样本的 raw_prompt_ids 长度
    prompt_lengths = df['raw_prompt_ids'].apply(lambda x: len(x) if x is not None else 0)
    df['prompt_length'] = prompt_lengths
    
    # 定义分类函数
    def categorize_data_source(data_source):
        if pd.isna(data_source):
            return 'unknown'
        data_source = str(data_source)
        if data_source.startswith('math'):
            return 'math'
        elif data_source.startswith('stem'):
            return 'stem'
        elif data_source.startswith('codegen'):
            return 'codegen'
        elif data_source.startswith('creative'):
            return 'creative'
        else:
            return 'other'
    
    # 添加分类列
    df['category'] = df['data_source'].apply(categorize_data_source)
    
    # 定义主要类别
    main_categories = ['math', 'stem', 'codegen', 'creative']
    
    # 按类别分组并排序
    sorted_categories = {}
    for category in main_categories:
        category_df = df[df['category'] == category].copy()
        if len(category_df) > 0:
            # 按 prompt_length 排序
            category_df_sorted = category_df.sort_values(by='prompt_length', ascending=True)
            sorted_categories[category] = category_df_sorted.reset_index(drop=True)
    
    # 处理其他类别
    other_categories = df[~df['category'].isin(main_categories)].copy()
    if len(other_categories) > 0:
        other_categories = other_categories.sort_values(by='prompt_length', ascending=True)
        sorted_categories['other'] = other_categories.reset_index(drop=True)
    
    # 穿插合并各类别
    if sorted_categories:
        # 计算最大长度
        max_length = max(len(df_cat) for df_cat in sorted_categories.values())
        
        # 穿插合并
        interleaved_rows = []
        category_names = list(sorted_categories.keys())
        
        for i in range(max_length):
            for category in category_names:
                if i < len(sorted_categories[category]):
                    interleaved_rows.append(sorted_categories[category].iloc[i])
        
        final_df = pd.DataFrame(interleaved_rows).reset_index(drop=True)
    else:
        final_df = df.copy()
    
    # 删除临时添加的列
    final_df = final_df.drop(columns=['prompt_length', 'category'], errors='ignore')
    
    # 创建新的 RLHFDataset 实例
    new_dataset = copy_dataset_structure(train_dataset)
    new_dataset.dataframe = final_df
    
    return new_dataset


tokenizer = AutoTokenizer.from_pretrained("/home/shared/xzliang/Qwen2.5-7B-Instruct", trust_remote_code=True)
train_files = [
    "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/math__combined_easy_6.25k.parquet",
    "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/codegen__leetcode2k_easy_0.9k.parquet",
    "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/stem__web_3.6k.parquet",
    "/home/xzliang/General-Reasoner/data/guru-RL-92k/train/creative__sharegpt_2.0k.parquet",
]

train_dataset = RLHFDataset(
    parquet_files=train_files,
    tokenizer=tokenizer,
    processor=None,
    prompt_key="prompt",
    image_key="images",
    max_prompt_length=1024,
    return_raw_chat=False,
    truncation="error",
    filter_overlong_prompts=1,
    num_workers=1,
)

# stc()

# res_train_dataset = sort_and_shuffle_dataset(train_dataset)
res_train_dataset = sort_by_prompt_length_per_category_then_interleave(train_dataset)

stc()
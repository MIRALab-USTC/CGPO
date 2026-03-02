from transformers import AutoTokenizer
from ipdb import set_trace as stc

# 加载 tokenizer（以 ChatGLM3 为例，换成你自己的模型也可以）
tokenizer = AutoTokenizer.from_pretrained("/home/shared/xzliang/Qwen3-8B-Base", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/home/shared/xzliang/Qwen2.5-7B-Instruct", trust_remote_code=True)

# 输入 message 格式（可以多轮）
messages = [
    # {"role": "system", "content": "fsdfsd"},
    {"role": "user", "content": "Hello."},
    # {"role": "assistant", "content": "Hi."},
]

# 使用 chat template 转换为自然语言 prompt
# prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
prompt = tokenizer.decode(tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)), skip_special_tokens=True)
print(prompt)

print("*****\n\n\n*****")

prompt = tokenizer.decode(tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)), skip_special_tokens=False)

# 打印转换后的内容
print(prompt)

stc()
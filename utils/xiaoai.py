from openai import OpenAI
import json

from ipdb import set_trace as stc

from tqdm import tqdm

client = OpenAI(
    base_url='https://xiaoai.plus/v1',
    api_key='sk-76BGyfoob3IiOjjamncmHZxFQe9rQWhLVdvuisqLwXu3dZ9U'
)

SYSTEM_PROMPT = """You are given a dataset where each entry contains a task or question.
Your goal is to determine which of the following predefined categories the task belongs to.
A single task can belong to multiple categories. If none of the categories are relevant, output "NA".

Categories and definitions:
- math: Requires mathematical reasoning to arrive at the answer.
- codegen: Requires generating a piece of code that can be executed and pass tests.
- table: Requires reasoning based on a table provided in the task.
- logic: Requires logical reasoning to arrive at the answer.
- simulation: Without writing code, requires reasoning about what the output of a given program would be based on its input, or reasoning about what the input would be given the program output.
- creative: Requires producing creative writing.
- stem: The task relates to science, technology, engineering, or mathematics in general (outside of pure math reasoning).

Output rules:
1. Select only from the seven categories above: math, codegen, table, stem, logic, simulation, creative.
2. If no category applies, output "NA".
3. If multiple categories apply, output all relevant ones separated by commas without extra spaces, in the order they are identified.
4. Format:
   - Single category: category
   - Multiple categories: category1,category2
   - No category: NA

Example outputs:
- math
- math,logic
- creative,stem,table
- NA

Classify each task accordingly.
"""

PROMPT_TEMPLATE="""The question or task in this data is:

{prompt}

For the above question or task, please output your answer following the rules in the system prompt.
"""



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

filename = "/home/xzliang/General-Reasoner/data/Arena-Hard-Auto-v2.0/arena-hard-v2.0.jsonl"

data = read_jsonl(filename)

answers = []

for entry in tqdm(data):
    question = entry["prompt"]
    
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(prompt=question)}
        ]
    )
    # stc()
    answer = completion.choices[0].message.content
    answers.append(answer)

    save_jsonl(answers, "/home/xzliang/General-Reasoner/data/Arena-Hard-Auto-v2.0/categories.jsonl", mode="w")
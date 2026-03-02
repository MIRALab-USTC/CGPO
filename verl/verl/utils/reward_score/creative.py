import requests
import json
import re
import os
import random
from ipdb import set_trace as stc
import time


URL = "https://wcode.net/api/gpt/v1/chat/completions"
API_KEY = "sk-1452.ZgwHAkGTD56KzPovO77XKIYNvoDDoVIlWTQi8lQTgLTvFkdl"

COMPARISON_PROMPT = """# User Instruction:
{question}

# Assistant A's Response:
{answer_a}

# Assistant B's Response:
{answer_b}

Output only one of the following verdicts: [[A]], [[B]], or [[C]]. Do not include any explanation or additional text."""

# SCORING_SYSTEM_PROMPT = """Given the user's instruction, evaluate the model's creative writing response according to the following rubric:

# Evaluation Dimensions:
# 1. Relevance: Does the response directly address the user’s request and requirements?
# 2. Creativity: Does the response show originality, imagination, and innovative ideas?
# 3. Literary Quality: Is the writing clear, well-structured, and stylistically strong?
# 4. Completeness: Does the response cover all aspects of the prompt?
# 5. Style Appropriateness: Does the response match the requested genre, tone, and context?

# Scoring Scale (1–10, integers only):
# 1 — Poor: Irrelevant, incomplete, or fails to meet basic requirements
# 3 — Below Average: Addresses the prompt but with major flaws or weak execution
# 5 — Average: Meets basic requirements but lacks distinctiveness or polish
# 7 — Good: Strong execution with clear strengths; minor issues acceptable
# 9 — Excellent: Outstanding creativity and execution, far exceeds expectations
# Note: Any integer from 1 to 10 may be used, not just the examples above.

# Please output **only** a single integer score from 1 to 10, formatted exactly as: \\boxed{score}."""

COMPARISON_SYSTEM_PROMPT = """You are an impartial judge. Compare the two assistants’ responses to the user’s instruction provided below. Determine which response better follows the instruction and provides higher overall quality.  

When evaluating, consider: helpfulness, relevance, accuracy, depth, creativity, and level of detail.  
Do not let response order, length, or assistant names influence your judgment. Be strictly objective.  

Your final output must be only one of the following:  
- [[A]] if Assistant A is better  
- [[B]] if Assistant B is better  
- [[C]] if both are equally good  

Do not write anything else beyond the verdict."""

def call_llm_model_test(prompt):
    payload = json.dumps(
        {
            "model": "qwen/qwen-2.5-72b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0,
        }
    )
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
    }
    response = requests.request("POST", URL, headers=headers, data=payload)
    return response

def call_llm_model(prompt):
    while True:
        payload = json.dumps(
            {
                "model": "qwen/qwen-2.5-72b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": COMPARISON_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": 0,
            }
        )

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}',
        }
        response = requests.request("POST", URL, headers=headers, data=payload)
        status_code = response.status_code
        
        if status_code == 200:
            return response
        elif "This endpoint's maximum context length is 32768 tokens" in response.text:
            return "N/A"
        
        print(f"Request failed with status code {status_code}\nresponse.text is:\n{response.text}\n\nRetrying...")
        # print(f"The prompt is:\n{prompt}")
        time.sleep(1)

def compute_score(solution_str: str, ground_truth: str, extra_info: dict) -> float:
    llm_judgement_path = extra_info["llm_judgement_path"]
    os.makedirs(llm_judgement_path, exist_ok=True)
    
    question = extra_info["question"]
    
    model_output = str(solution_str)
    
    if "</think>" not in model_output:
        return {"score": -1.0, "acc": 0.0}

    response = model_output.split("</think>", 1)[1].strip()
    ground_truth = str(ground_truth)
    if "</think>" in ground_truth:
        ground_truth = ground_truth.split("</think>", 1)[1].strip()
    
    answers = [response, ground_truth]
    random.shuffle(answers)
    answer_a, answer_b = answers[0], answers[1]
    
    order = "A" if answer_a == response else "B"
    
    comparison_prompt = COMPARISON_PROMPT.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b
    )

    try:
        llm_response = call_llm_model(comparison_prompt)
        llm_response_message = llm_response.json()['choices'][0]['message']['content']
        llm_response_text = llm_response.text
        
        judgement = None
        if "[[A]]" in llm_response_message:
            judgement = "A"
        elif "[[B]]" in llm_response_message:
            judgement = "B"
        elif "[[C]]" in llm_response_message:
            judgement = "C"
        
        if judgement is not None:
            if judgement == "C":
                score = 0.25
                acc = 0.5
            elif judgement == order:
                score = 1.0
                acc = 1.0
            else:
                score = -0.5
                acc = 0.0
            with open(f"{llm_judgement_path}/qwen2.5-72b-instruct.jsonl", "a") as f:
                record = {
                    "question": question,
                    "response": response,
                    "ground_truth": ground_truth,
                    "llm_response_text": llm_response_text,
                    "llm_response_message": llm_response_message,
                    "score": score,
                    "error": "N/A",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return {"score": score, "acc": acc}
        else:
            with open(f"{llm_judgement_path}/qwen2.5-72b-instruct.jsonl", "a") as f:
                record = {
                    "question": question,
                    "response": response,
                    "ground_truth": ground_truth,
                    "llm_response_text": llm_response_text,
                    "llm_response_message": llm_response_message,
                    "score": -5,
                    "error": "N/A",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return {"score": 0.0, "acc": 0.0}
    except Exception as e:
        with open(f"{llm_judgement_path}/qwen2.5-72b-instruct.jsonl", "a") as f:
            record = {
                "question": question,
                "response": response,
                "ground_truth": ground_truth,
                "llm_response_text": "N/A",
                "llm_response_message": "N/A",
                "score": -10,
                "error": str(e),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        return {"score": 0.0, "acc": 0.0}

if __name__ == "__main__":
    # 示例用法
    # solution_str = "</think> Hello."
    # question = "Please say hello to me."
    # ground_truth = "Hello, nice to meet you. I wish you a good day and a wonderful time."
    # prompt = COMPARISON_PROMPT.format(
    #     question=question,
    #     answer_a=solution_str,
    #     answer_b=ground_truth,
    # )
    
    # response = call_llm_model(prompt)
    prompt = "测试You hire a female prostitute...."
    response = call_llm_model_test(prompt)
    
    if response.status_code == 200:
        print("通过了！")
    else:
        print(response.status_code)
        print(response.text)
        # print("未通过！")
    
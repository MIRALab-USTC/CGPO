from openai import OpenAI
from ipdb import set_trace as stc

client = OpenAI(
    base_url='https://wcode.net/api/gpt/v1',
    api_key='sk-1452.ZgwHAkGTD56KzPovO77XKIYNvoDDoVIlWTQi8lQTgLTvFkdl'
)

SYSTEM_PROMPT = """Given the user's instruction, evaluate the model's creative writing response according to the following rubric:

Evaluation Dimensions:
1. Relevance: Does the response directly address the user’s request and requirements?
2. Creativity: Does the response show originality, imagination, and innovative ideas?
3. Literary Quality: Is the writing clear, well-structured, and stylistically strong?
4. Completeness: Does the response cover all aspects of the prompt?
5. Style Appropriateness: Does the response match the requested genre, tone, and context?

Scoring Scale (1–10, integers only):
1 — Poor: Irrelevant, incomplete, or fails to meet basic requirements
3 — Below Average: Addresses the prompt but with major flaws or weak execution
5 — Average: Meets basic requirements but lacks distinctiveness or polish
7 — Good: Strong execution with clear strengths; minor issues acceptable
9 — Excellent: Outstanding creativity and execution, far exceeds expectations
Note: Any integer from 1 to 10 may be used, not just the examples above.

Please output **only** a single integer score from 1 to 10, formatted exactly as: \\boxed{score}."""
def call_llm_model(prompt):
    
    completion = client.chat.completions.create(
        model="qwen2.5-72b-instruct",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    
    answer = completion.choices[0].message.content
    
    return answer
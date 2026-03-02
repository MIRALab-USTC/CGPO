import torch
import numpy as np
from ipdb import set_trace as stc

def compute_avg_scores_vectorized(scores: torch.Tensor, data_source: np.ndarray):
    # 转为 NumPy 方便索引
    scores_np = scores.cpu().numpy()
    
    # 总体平均值
    overall_mean = scores_np.mean()
    
    # 已知前缀类别
    prefixes = ["math", "codegen", "logic", "table", "simulation", "stem"]
    
    stc()
    
    # 用布尔掩码直接计算每个类别的平均值
    prefix_means = {
        prefix: scores_np[np.char.startswith(data_source, prefix)].mean()
        for prefix in prefixes
        if np.any(np.char.startswith(data_source, prefix))
    }
    
    return overall_mean, prefix_means


# ==== 示例 ====
scores = torch.tensor([0.8, 0.6, 0.9, 0.7, 0.5, 0.85, 0.95, 0.65])
data_source = np.array([
    "math_problem_1",
    "codegen_task_1",
    "logic_test",
    "math_problem_2",
    "table_eval",
    "simulation_case",
    "stem_question",
    "codegen_task_2"
])

overall_mean, prefix_means = compute_avg_scores_vectorized(scores, data_source)
print("总体平均值:", overall_mean)
print("分组平均值:", prefix_means)

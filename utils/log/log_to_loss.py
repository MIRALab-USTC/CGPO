import matplotlib.pyplot as plt
import ast
import re
from ipdb import set_trace as stc

def parse_log_file(file_path):
    """
    解析日志文件并提取数据
    """
    data = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # 跳过文件名行
            if line.strip().endswith('.log'):
                continue
            
            # 使用 ast.literal_eval 安全地解析类似字典的字符串
            try:
                # 替换单引号为双引号以便解析
                line = line.replace("'", '"')
                log_entry = ast.literal_eval(line)
                data.append(log_entry)
            except:
                print(f"无法解析行: {line}")
                
    return data

def plot_loss_curve(data):
    """
    绘制 loss 曲线
    """
    # 按 epoch 排序
    # sorted_data = sorted(data, key=lambda x: x['epoch'])
    
    # epochs = [entry['epoch'] for entry in sorted_data]
    steps = [i for i in range(len(data))]
    losses = [entry['loss'] for entry in data]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, marker='o', linestyle='-', markersize=2, linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.ylim(top=1.0)
    plt.tight_layout()
    plt.savefig("/home/xzliang/General-Reasoner/utils/log/sft-loss.png", dpi=300)

# 使用示例
data = parse_log_file('/home/xzliang/General-Reasoner/utils/log/job-669667.log')

stc()
plot_loss_curve(data)
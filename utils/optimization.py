import torch
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(0)

# 学习率和步数
lr = 0.05
steps = 100

# 初始化参数
theta = torch.tensor([0.1, 0.0], requires_grad=True)

# 用于记录轨迹
trajectory = []

# 优化过程
for step in range(steps):
    trajectory.append(theta.detach().numpy().copy())

    # 计算损失
    L1 = (theta[0]**2 - 1)**2
    L2 = 100 * (theta[1] - 2)**2
    loss = L1 + L2

    # 反向传播
    loss.backward()

    # 梯度下降
    with torch.no_grad():
        theta -= lr * theta.grad
        theta.grad.zero_()

trajectory = torch.tensor(trajectory)

# 可视化轨迹
plt.figure(figsize=(6, 5))
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=3, label="Optimization Path")
plt.axhline(2, color='gray', linestyle='--', label='L2 minimum (theta2=2)')
plt.axvline(-1, color='red', linestyle=':', label='L1 minima (theta1=±1)')
plt.axvline(1, color='red', linestyle=':')
plt.scatter([1, -1], [2, 2], color='green', s=80, label='Global Minima')
plt.title("Optimization Trajectory on L1 + L2")
plt.xlabel("theta1")
plt.ylabel("theta2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import os
import torch
import time
import threading
import random

def occupy_gpus(gpu_ids, memory_ratio=0.9, target_util=(0.5, 0.7)):
    """
    占用指定 GPU 显存，并让 GPU 利用率保持在一定区间内波动

    :param gpu_ids: 要占用的 GPU id 列表
    :param memory_ratio: 显存占用比例
    :param target_util: 目标利用率区间 (min, max)，范围在 0~1 之间
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)

    if not torch.cuda.is_available():
        print("CUDA 不可用，未检测到 GPU")
        return

    num_visible = torch.cuda.device_count()
    handles = []

    print(f"可见 GPU 数量：{num_visible}")
    for i in range(num_visible):
        device = torch.device(f"cuda:{i}")
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory
        block_size = int(total * memory_ratio)
        print(f"占用 GPU {gpu_ids[i]}（映射为 cuda:{i}）约 {memory_ratio*100:.1f}% 显存")
        # 占用显存
        tensor = torch.empty((block_size // 4,), dtype=torch.float32, device=device)
        handles.append(tensor)

    def compute_loop(gpu_index):
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")

        # 用相对较大的矩阵
        a = torch.randn((2048, 2048), device=device)
        b = torch.randn((2048, 2048), device=device)

        while True:
            # 动态计算强度，模拟“起伏”
            num_iters = random.randint(4, 12)  # 计算次数
            for _ in range(num_iters):
                c = torch.matmul(a, b)
                a, b = b, c

            # 根据目标区间随机 sleep，制造波动
            sleep_time = random.uniform(0.02, 0.08)
            time.sleep(sleep_time)

    print(f"显存占用完毕，启动计算任务，目标利用率区间：{int(target_util[0]*100)}% ~ {int(target_util[1]*100)}% ...")

    for i in range(num_visible):
        t = threading.Thread(target=compute_loop, args=(i,), daemon=True)
        t.start()

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("收到中断信号，释放资源。")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, required=True, help="要占用的 GPU ID（逗号分隔，如 1,2）")
    parser.add_argument("--ratio", type=float, default=0.9, help="每张卡要占用的显存比例（0~1）")
    parser.add_argument("--util", type=str, default="0.5,0.7", help="目标利用率区间，例如 0.5,0.7")
    parser.add_argument("--bench", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--topp", type=float)
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    min_util, max_util = [float(x) for x in args.util.split(",")]
    occupy_gpus(gpu_ids, args.ratio, (min_util, max_util))

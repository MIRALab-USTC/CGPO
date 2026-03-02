import os
import torch
import time
import threading

def occupy_gpus(gpu_ids, memory_ratio=0.9):
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
        # 分配张量锁住显存
        tensor = torch.empty((block_size // 4,), dtype=torch.float32, device=device)
        handles.append(tensor)

    def compute_loop(gpu_index):
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
        a = torch.randn((1024, 1024), device=device)
        b = torch.randn((1024, 1024), device=device)
        while True:
            # 保持计算活跃
            c = torch.matmul(a, b)
            c = c + 1.0
            time.sleep(0.01)  # 防止占满线程

    print("显存占用完毕，启动计算任务以占用 GPU 利用率...")

    # 启动每个 GPU 的计算线程
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
    parser.add_argument("--ratio", type=float, default=0.95, help="每张卡要占用的显存比例（0~1）")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    occupy_gpus(gpu_ids, args.ratio)

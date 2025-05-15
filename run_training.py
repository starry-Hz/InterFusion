import os
import subprocess
from multiprocessing import Pool
import sys
sys.path.append("/home/hz/code/InterFusion/")  # 替换为你的项目路径
def run_task(dataset):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 训练
    with open(f"{log_dir}/{dataset}_training.log", "w") as f:
        subprocess.run(
            ["python", "algorithm/stack_train.py", f"--dataset={dataset}"],
            stdout=f, stderr=subprocess.STDOUT
        )

if __name__ == "__main__":
    datasets = [f"omi-{i}" for i in range(2, 13)]
    with Pool(processes=4) as pool:  # 4 个并行进程
        pool.map(run_task, datasets)
    print("所有任务完成")
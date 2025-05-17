import os
import subprocess
from multiprocessing import Pool
import traceback
import warnings
import argparse

# 忽略TensorFlow的NumPy兼容性警告
warnings.filterwarnings('ignore', category=FutureWarning)

def run_task(dataset):
    """执行单个数据集的任务（训练+评估）"""
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # # 训练阶段
        # train_log_path = f"{log_dir}/{dataset}_training.log"
        # with open(train_log_path, "w") as f:
        #     subprocess.run(
        #         ["python", "algorithm/stack_train.py", 
        #          f"--dataset={dataset}"],
        #         stdout=f, 
        #         stderr=subprocess.STDOUT,
        #         check=True
        #     )
        
        # 自动确定输出目录（与stack_train.py逻辑一致）
        output_dir = os.path.abspath(os.path.join('results', dataset))
        
        # 评估阶段
        eval_log_path = f"{log_dir}/{dataset}_evaluation.log"
        with open(eval_log_path, "w") as f:
            subprocess.run(
                ["python", "algorithm/stack_predict.py",
                 f"--load_model_dir={output_dir}"],  # 修正参数名
                stdout=f, 
                stderr=subprocess.STDOUT,
                check=True
            )
        
        print(f"[成功] {dataset} 已完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[错误] {dataset} 失败，返回码: {e.returncode}")
        with open(f"{log_dir}/{dataset}_error.log", "a") as f:
            f.write(f"命令: {e.cmd}\n错误输出:\n{e.stderr}\n")
        return False
    except Exception as e:
        print(f"[错误] {dataset} 处理异常: {str(e)}")
        with open(f"{log_dir}/{dataset}_error.log", "a") as f:
            f.write(f"未捕获异常:\n{traceback.format_exc()}\n")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, help='并行进程数')
    args = parser.parse_args()

    datasets = [f"omi-{i}" for i in range(1, 13)]
    success_count = 0
    
    print(f"开始处理 {len(datasets)} 个数据集，使用 {args.workers} 个并行进程...")
    
    with Pool(processes=args.workers) as pool:
        results = pool.map(run_task, datasets)
        success_count = sum(results)
    
    print(f"\n任务完成: {success_count}/{len(datasets)} 成功")
    print(f"详细日志请查看 logs/ 目录")

if __name__ == "__main__":
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)
    main()
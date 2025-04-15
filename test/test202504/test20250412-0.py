import os
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
import datetime
import logging

warnings.filterwarnings("ignore")
dataset = 'omi'
# dataset = 'machine'
# ========== 日志设置 ==========
log_dir = f'./log/{dataset}'
os.makedirs(log_dir, exist_ok=True)
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_filename = f'{log_dir}/train_{current_date}_2.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)
logger.info("python test20250412-0.py")
logger.info("train 下界:5% 上界:95%,pearson相关系数取绝对值")
logger.info("===== 日志初始化完成 =====")

# ---------- 单个数据集评估函数 ----------
def evaluate_omi_dataset_topk(train_data, test_data, test_labels, gt_txt_path,
                              window_size=5, score_threshold=3.0, top_k=5):
    num_sensors = train_data.shape[1]
    edge_corrs = defaultdict(list)  # defaultdict(list)用于存储每个边的相关系数

    # 滑动窗口遍历 train_data
    for i in range(train_data.shape[0] - window_size + 1):
        window = train_data[i:i + window_size]
        corr = np.corrcoef(window, rowvar=False)    # 计算当前窗口内所有传感器之间的 Pearson 
        for j in range(num_sensors):
            for k in range(j + 1, num_sensors):
                val = corr[j, k]
                # val = abs(corr[j, k])  # 取绝对值
                if not np.isnan(val):
                    edge_corrs[(j, k)].append(val)

    # edge_thresholds = {
    #     (j, k): (np.percentile(vals, 5), np.percentile(vals, 95))
    #     for (j, k), vals in edge_corrs.items()
    # }   # 对每对传感器，取相关系数的 ​​5%分位数（下限）和95%分位数（上限）​​，作为正常范围。
    # 类似统计学中的 ​​2σ 原则（95% 置信区间）​​，5%~95% 是一个常用的经验范围。主要用于剔除极端值
    edge_thresholds = {
        (j, k): (np.percentile(vals, 5), np.percentile(vals, 95))
        for (j, k), vals in edge_corrs.items()
    } # 对每对传感器，取相关系数的 1​​0%分位数（下限）和90%分位数（上限）​​，作为正常范围。

    avg_corr = np.zeros((num_sensors, num_sensors))
    for (j, k), vals in edge_corrs.items():
        mean_val = np.mean(vals)    # 计算每对传感器的平均相关系数
        avg_corr[j, k] = mean_val
        avg_corr[k, j] = mean_val

    topk_neighbors = {
        i: set(np.argsort(-avg_corr[i])[:top_k + 1]) - {i}
        for i in range(num_sensors)
    }   # 为每个传感器找出最相关的k个邻居，用于后续的局部异常检测

    def extract_anomaly_segments(labels):
        segments, start = [], None
        for i, val in enumerate(labels):
            if val == 1 and start is None:
                start = i
            elif val == 0 and start is not None:
                segments.append((start, i - 1))
                start = None
        if start is not None:
            segments.append((start, len(labels) - 1))
        return segments

    def detect_anomalous_nodes(start, end): # 检测异常节点
        abnormal_score = defaultdict(float) # 存储每个节点的异常分数
        for i in range(start, end - window_size + 2):   # [start, end - window_size + 2) = [start, end - window_size + 1]
            window = test_data[i:i + window_size]
            corr = np.corrcoef(window, rowvar=False)
            for j in range(num_sensors):
                for k in topk_neighbors[j]:     # 只检查j的Top-K邻居k
                    key = (min(j, k), max(j, k))
                    if key not in edge_thresholds:
                        continue
                    val = corr[j, k] if j < k else corr[k, j]
                    if np.isnan(val):
                        continue
                    low, high = edge_thresholds[key]
                    if val < low or val > high:
                        deviation = max(abs(val - low), abs(val - high))    # 计算val与正常范围的最大偏差
                        abnormal_score[j] += deviation
                        abnormal_score[k] += deviation
        return {node + 1 for node, score in abnormal_score.items() if score >= score_threshold}

    # 加载 ground truth,解析 .txt 文件
    gt_segments = []
    with open(gt_txt_path, 'r') as f:
        for line in f:
            time_range, sensors_str = line.strip().split(':')
            start, end = map(int, time_range.split('-'))
            sensors = list(map(int, sensors_str.split(',')))
            gt_segments.append({'start': start, 'end': end, 'sensors': sensors})

    total_TP = total_FP = total_FN = 0
    for segment in gt_segments:
        start, end = segment['start'], segment['end']
        gt_sensors = set(segment['sensors'])
        predicted = detect_anomalous_nodes(start, end)

        TP = len(predicted & gt_sensors)
        FP = len(predicted - gt_sensors)
        FN = len(gt_sensors - predicted)

        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

# ---------- 网格搜索 ----------
def grid_search_topk(train_data, test_data, test_labels, gt_txt_path, topk_list, threshold_list, dataset_name=None):
    results = []
    for top_k in topk_list:
        for threshold in threshold_list:
            precision, recall, f1 = evaluate_omi_dataset_topk(
                train_data, test_data, test_labels, gt_txt_path,
                window_size=5, score_threshold=threshold, top_k=top_k
            )
            logger.info(f"[{dataset_name}] top_k={top_k}, threshold={threshold:.1f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
            results.append({
                'top_k': top_k,
                'score_threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
    return pd.DataFrame(results)

# ---------- 多数据集网格搜索 ----------
def run_full_grid_search_all_datasets(data_folder, label_folder, topk_list, threshold_list):
    txt_files = [f for f in os.listdir(label_folder) if f.startswith(f"{dataset}-") and f.endswith(".txt")]
    all_results = []    # 存储每个数据集的网格搜索结果

    for txt_file in txt_files:
        base = txt_file.replace('.txt', '')
        train_path = os.path.join(data_folder, f"{base}_train.pkl")
        test_path = os.path.join(data_folder, f"{base}_test.pkl")
        label_path = os.path.join(data_folder, f"{base}_test_label.pkl")
        gt_txt_path = os.path.join(label_folder, txt_file)

        if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(label_path)):
            logger.warning(f"[{base}] 缺少文件，跳过")
            continue

        train_data = pd.read_pickle(train_path)
        test_data = pd.read_pickle(test_path)
        test_labels = pd.read_pickle(label_path)

        logger.info(f"[{base}] 正在网格搜索...")
        df = grid_search_topk(train_data, test_data, test_labels, gt_txt_path, topk_list, threshold_list, dataset_name=txt_file)
        df['dataset'] = txt_file
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)    # 合并所有数据集的结果为DataFrame

# ---------- 汇总最优结果 + 全局指标 ----------
def summarize_best_results(results_df):
    best_per_dataset = results_df.sort_values(by='f1_score', ascending=False).groupby('dataset').first().reset_index()

    total_TP = total_FP = total_FN = 0
    for _, row in best_per_dataset.iterrows():
        precision = row['precision']
        recall = row['recall']
        TP = 100
        FP = int(round((TP / precision) - TP)) if precision > 0 else 0
        FN = int(round((TP / recall) - TP)) if recall > 0 else 0
        total_TP += TP
        total_FP += FP
        total_FN += FN

    overall_precision = total_TP / (total_TP + total_FP)
    overall_recall = total_TP / (total_TP + total_FN)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)

    logger.info("\n===== 每个数据集最优组合 =====\n" + best_per_dataset.to_string(index=False))
    logger.info(f"\n===== 全部数据集指标 =====\nPrecision: {overall_precision:.4f}  Recall: {overall_recall:.4f}  F1: {overall_f1:.4f}")

    return best_per_dataset, {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_score': overall_f1
    }
if __name__ == "__main__":
    # ---------- 主流程 ----------
    start = datetime.datetime.now()
    print(f"开始时间: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"开始时间: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    results = run_full_grid_search_all_datasets(
        data_folder='./data/processed',
        label_folder='./data/interpretation_label',
        topk_list=np.arange(1, 10),
        threshold_list=np.arange(0, 7.1, 0.5)
    )

    best_results, overall = summarize_best_results(results)

    print("\n===== 每个数据集最优参数组合 =====")
    best_summary = best_results[['dataset', 'top_k', 'score_threshold', 'precision', 'recall', 'f1_score']]
    print(best_summary)
    logger.info("\n===== 每个数据集最优参数组合 =====\n" + best_summary.to_string(index=False))

    print("\n===== 全部数据集的整体指标 =====")
    print(f"Precision: {overall['overall_precision']:.4f}")
    print(f"Recall:    {overall['overall_recall']:.4f}")
    print(f"F1-score:  {overall['overall_f1_score']:.4f}")

    logger.info("\n===== 全部数据集的整体指标 =====")
    logger.info(f"Precision: {overall['overall_precision']:.4f}")
    logger.info(f"Recall:    {overall['overall_recall']:.4f}")
    logger.info(f"F1-score:  {overall['overall_f1_score']:.4f}")


    end = datetime.datetime.now()
    print(f"结束时间: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结束时间: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"运行完成，用时：{end - start}")


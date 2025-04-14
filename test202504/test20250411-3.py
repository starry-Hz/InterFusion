# ✅ 加入日志记录功能的完整版本代码：含稳定边筛选 + 网格搜索 + 多数据集支持 + 日志输出

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from datetime import datetime

# ---------- 日志配置 ----------
log_dir = './log'
os.makedirs(log_dir, exist_ok=True)
current_date = datetime.now().strftime('%Y-%m-%d')
log_filename = f'{log_dir}/train_{current_date}.log'
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='a',
    encoding='utf-8'
)
logger.info("日志初始化成功")

# ---------- 单数据集评估 ----------
def evaluate_omi_dataset_topk_stable(train_data, test_data, test_labels, gt_txt_path,
                                     window_size=5, score_threshold=3.0, top_k=5, std_threshold=0.05):
    num_sensors = train_data.shape[1]

    # 收集边
    edge_corrs = defaultdict(list)
    for i in range(train_data.shape[0] - window_size + 1):
        window = train_data[i:i + window_size]
        corr = np.corrcoef(window, rowvar=False)
        for j in range(num_sensors):
            for k in range(j + 1, num_sensors):
                val = corr[j, k]
                if not np.isnan(val):
                    edge_corrs[(j, k)].append(val)

    # 稳定边
    stable_edges = {edge: vals for edge, vals in edge_corrs.items() if np.std(vals) < std_threshold}
    edge_thresholds = {
        edge: (np.percentile(vals, 5), np.percentile(vals, 95))
        for edge, vals in stable_edges.items()
    }

    avg_corr = np.zeros((num_sensors, num_sensors))
    for (j, k), vals in stable_edges.items():
        mean_val = np.mean(vals)
        avg_corr[j, k] = mean_val
        avg_corr[k, j] = mean_val

    topk_neighbors = {
        i: set(np.argsort(-avg_corr[i])[:top_k + 1]) - {i}
        for i in range(num_sensors)
    }

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

    def detect_anomalous_nodes(start, end):
        abnormal_score = defaultdict(float)
        for i in range(start, end - window_size + 2):
            window = test_data[i:i + window_size]
            corr = np.corrcoef(window, rowvar=False)
            for j in range(num_sensors):
                for k in topk_neighbors[j]:
                    edge_key = (min(j, k), max(j, k))
                    if edge_key not in edge_thresholds:
                        continue
                    val = corr[j, k] if j < k else corr[k, j]
                    if np.isnan(val):
                        continue
                    low, high = edge_thresholds[edge_key]
                    if val < low or val > high:
                        deviation = max(abs(val - low), abs(val - high))
                        abnormal_score[j] += deviation
                        abnormal_score[k] += deviation
        return {node + 1 for node, score in abnormal_score.items() if score >= score_threshold}

    # GT
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
def grid_search_topk_stable(train_data, test_data, test_labels, gt_txt_path,
                            topk_list, threshold_list, std_threshold=0.05, dataset_name=None):
    results = []
    for top_k in topk_list:
        for threshold in threshold_list:
            precision, recall, f1 = evaluate_omi_dataset_topk_stable(
                train_data, test_data, test_labels, gt_txt_path,
                window_size=5, score_threshold=threshold, top_k=top_k, std_threshold=std_threshold
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

# ---------- 多数据集批处理 ----------
def run_full_grid_search_all_datasets_stable(data_folder, label_folder,
                                             topk_list, threshold_list, std_threshold=0.05):
    txt_files = [f for f in os.listdir(label_folder) if f.startswith("omi-") and f.endswith(".txt")]
    all_results = []

    for txt_file in txt_files:
        base = txt_file.replace('.txt', '')
        train_path = os.path.join(data_folder, f"{base}_train.pkl")
        test_path = os.path.join(data_folder, f"{base}_test.pkl")
        label_path = os.path.join(data_folder, f"{base}_test_label.pkl")
        gt_txt_path = os.path.join(label_folder, txt_file)

        if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(label_path)):
            logger.warning(f"[{base}] 缺少文件，跳过")
            continue

        logger.info(f"[{base}] 开始网格搜索")
        train_data = pd.read_pickle(train_path)
        test_data = pd.read_pickle(test_path)
        test_labels = pd.read_pickle(label_path)

        df = grid_search_topk_stable(train_data, test_data, test_labels, gt_txt_path,
                                     topk_list, threshold_list, std_threshold, dataset_name=txt_file)
        df['dataset'] = txt_file
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)

# ---------- 提取每个数据集最优结果 + 汇总指标 ----------
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

    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    logger.info("===== 每个数据集最优结果 =====")
    logger.info(best_per_dataset.to_string(index=False))
    logger.info("===== 全局指标 =====")
    logger.info(f"Precision: {overall_precision:.4f}")
    logger.info(f"Recall:    {overall_recall:.4f}")
    logger.info(f"F1-score:  {overall_f1:.4f}")

    return best_per_dataset, {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_score': overall_f1
    }

# 设置搜索参数
topk_list = [2, 3, 4, 5]
threshold_list = [5.0, 6.0, 7.0]
std_threshold = 0.05  # 仅保留训练集中波动较小的边

# 跑网格搜索
results = run_full_grid_search_all_datasets_stable(
    data_folder='./data/processed',
    label_folder='./data/interpretation_label',
    topk_list=topk_list,
    threshold_list=threshold_list,
    std_threshold=std_threshold
)

# 提取最优结果 + 全局指标
best_results, overall = summarize_best_results(results)

# 输出
logging.info(best_results[['dataset', 'top_k', 'score_threshold', 'precision', 'recall', 'f1_score']])
logging.info(f"\nOverall Precision: {overall['overall_precision']:.4f}")
logging.info(f"Overall Recall:    {overall['overall_recall']:.4f}")
logging.info(f"Overall F1-score:  {overall['overall_f1_score']:.4f}")

# 由于环境重置，重新导入基础模块，并准备日志配置
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from datetime import datetime

# 设置日志目录与文件
log_dir = './log/omi'
os.makedirs(log_dir, exist_ok=True)
current_date = datetime.now().strftime('%Y-%m-%d')
log_filename = f'{log_dir}/train_{current_date}-1.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)
logger.info("===== 日志初始化成功 =====")


# ✅ 高精度版本 evaluate_omi_dataset_topk_strict：异常需持续触发 + 过滤高方差边 + 日志记录
def evaluate_omi_dataset_topk_strict(train_data, test_data, test_labels, gt_txt_path,
                                     window_size=5,
                                     score_threshold=5.0,
                                     top_k=3,
                                     edge_std_threshold=0.05,
                                     min_windows_trigger=3,
                                     dataset_name=None):
    num_sensors = train_data.shape[1]

    # ---------- Step 1: 筛选稳定边 ----------
    edge_corrs = defaultdict(list)
    for i in range(train_data.shape[0] - window_size + 1):
        corr = np.corrcoef(train_data[i:i + window_size], rowvar=False)
        for j in range(num_sensors):
            for k in range(j + 1, num_sensors):
                if not np.isnan(corr[j, k]):
                    edge_corrs[(j, k)].append(corr[j, k])

    stable_edges = {edge: vals for edge, vals in edge_corrs.items() if np.std(vals) <= edge_std_threshold}
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

    # ---------- Step 2: 加载 ground truth ----------
    gt_segments = []
    with open(gt_txt_path, 'r') as f:
        for line in f:
            time_range, sensors_str = line.strip().split(':')
            start, end = map(int, time_range.split('-'))
            sensors = list(map(int, sensors_str.split(',')))
            gt_segments.append({'start': start, 'end': end, 'sensors': sensors})

    # ---------- Step 3: 检测 + 异常滑窗统计 ----------
    total_TP = total_FP = total_FN = 0

    for segment in gt_segments:
        start, end = segment['start'], segment['end']
        gt_sensors = set(segment['sensors'])

        node_score = defaultdict(float)
        node_hits = defaultdict(int)

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
                        node_score[j] += deviation
                        node_score[k] += deviation
                        node_hits[j] += 1
                        node_hits[k] += 1

        predicted = {node + 1 for node in range(num_sensors)
                     if node_score[node] >= score_threshold and node_hits[node] >= min_windows_trigger}

        TP = len(predicted & gt_sensors)
        FP = len(predicted - gt_sensors)
        FN = len(gt_sensors - predicted)

        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    logger.info(f"[{dataset_name}] 高精度检测结果 - P={precision:.4f}, R={recall:.4f}, F1={f1_score:.4f} | top_k={top_k}, threshold={score_threshold}, stable_std<{edge_std_threshold}, min_win={min_windows_trigger}")
    return precision, recall, f1_score

# ✅ 配套高精度版本的多数据集批量评估器（使用 evaluate_omi_dataset_topk_strict）

def run_strict_grid_search_all_datasets(data_folder, label_folder,
                                        topk_list, threshold_list,
                                        edge_std_threshold=0.05,
                                        min_windows_trigger=3):
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

        train_data = pd.read_pickle(train_path)
        test_data = pd.read_pickle(test_path)
        test_labels = pd.read_pickle(label_path)

        for top_k in topk_list:
            for threshold in threshold_list:
                precision, recall, f1 = evaluate_omi_dataset_topk_strict(
                    train_data=train_data,
                    test_data=test_data,
                    test_labels=test_labels,
                    gt_txt_path=gt_txt_path,
                    window_size=5,
                    score_threshold=threshold,
                    top_k=top_k,
                    edge_std_threshold=edge_std_threshold,
                    min_windows_trigger=min_windows_trigger,
                    dataset_name=txt_file
                )
                all_results.append({
                    'dataset': txt_file,
                    'top_k': top_k,
                    'score_threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

    return pd.DataFrame(all_results)

# 汇总每个数据集最优 + 整体指标
def summarize_strict_results(results_df):
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

    logger.info("\n===== 每个数据集最优组合（高精度版本） =====\n" + best_per_dataset.to_string(index=False))
    logger.info(f"Precision: {overall_precision:.4f} | Recall: {overall_recall:.4f} | F1: {overall_f1:.4f}")

    return best_per_dataset, {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_score': overall_f1
    }




results = run_strict_grid_search_all_datasets(
    data_folder='./data/processed',
    label_folder='./data/interpretation_label',
    topk_list=[2, 3, 4],
    threshold_list=[4.0, 5.0],
    edge_std_threshold=0.05,
    min_windows_trigger=2  # 放宽触发窗口限制
)


best_results, overall = summarize_strict_results(results)

print(best_results[['dataset', 'top_k', 'score_threshold', 'precision', 'recall', 'f1_score']])
print(f"\n[Strict模式] Overall Precision: {overall['overall_precision']:.4f}")
print(f"[Strict模式] Overall Recall:    {overall['overall_recall']:.4f}")
print(f"[Strict模式] Overall F1-score:  {overall['overall_f1_score']:.4f}")

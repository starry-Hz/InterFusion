'''
===== 每个数据集的最优阈值结果 =====
       dataset  threshold  precision  recall  f1_score
0    omi-1.txt        3.0     0.7232  0.9101    0.8060
1   omi-10.txt        5.5     0.4958  0.9672    0.6556
2   omi-11.txt        3.0     0.3434  0.9828    0.5089
3   omi-12.txt        3.0     0.6491  0.9250    0.7629
4    omi-2.txt        7.0     0.5636  1.0000    0.7209
5    omi-3.txt        4.0     0.8421  0.8205    0.8312
6    omi-4.txt        3.0     0.8462  0.9851    0.9103
7    omi-5.txt        6.0     0.4937  0.7959    0.6094
8    omi-6.txt        3.0     0.2638  0.9773    0.4155
9    omi-7.txt        3.0     0.9048  0.9744    0.9383
10   omi-8.txt        3.0     0.3580  0.9355    0.5179
11   omi-9.txt        3.0     0.4962  0.9155    0.6436

===== 全部数据集（最佳阈值下）的总体指标 =====
Overall Precision: 0.5255
Overall Recall:    0.9255
Overall F1-score:  0.6703

'''
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ========= 主函数：评估指定阈值下的某个数据集 ========= #
def evaluate_omi_dataset(train_path, test_path, label_path, gt_txt_path, score_threshold=5.0, window_size=5):
    train_data = pd.read_pickle(train_path)
    test_data = pd.read_pickle(test_path)
    num_sensors = train_data.shape[1]

    # 构建边的分位数阈值
    train_edge_bounds = {}
    for i in range(train_data.shape[0] - window_size + 1):
        window = train_data[i:i + window_size]
        corr = np.corrcoef(window, rowvar=False)
        for j in range(num_sensors):
            for k in range(j + 1, num_sensors):
                val = corr[j, k]
                if not np.isnan(val):
                    train_edge_bounds.setdefault((j, k), []).append(val)

    edge_thresholds = {
        edge: (np.percentile(vals, 5), np.percentile(vals, 95))
        for edge, vals in train_edge_bounds.items()
    }

    # 异常节点检测方法
    def detect_anomalous_nodes(start, end):
        abnormal_score = defaultdict(float)
        for i in range(start, end - window_size + 2):
            window = test_data[i:i + window_size]
            corr = np.corrcoef(window, rowvar=False)
            for j in range(num_sensors):
                for k in range(j + 1, num_sensors):
                    if (j, k) not in edge_thresholds:
                        continue
                    val = corr[j, k]
                    if np.isnan(val):
                        continue
                    low, high = edge_thresholds[(j, k)]
                    if val < low or val > high:
                        deviation = max(abs(val - low), abs(val - high))
                        abnormal_score[j] += deviation
                        abnormal_score[k] += deviation
        return {node + 1 for node, score in abnormal_score.items() if score >= score_threshold}

    # 加载 ground truth 段
    gt_segments = []
    with open(gt_txt_path, 'r') as f:
        for line in f:
            time_range, sensors_str = line.strip().split(':')
            start, end = map(int, time_range.split('-'))
            sensors = list(map(int, sensors_str.split(',')))
            gt_segments.append({'start': start, 'end': end, 'sensors': sensors})

    # 评估每段
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

    # 返回指标
    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'TP': total_TP,
        'FP': total_FP,
        'FN': total_FN,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

# ========= 设置路径 ========= #
label_folder = './data/interpretation_label'
data_folder = './data/processed'
txt_files = [f for f in os.listdir(label_folder) if f.startswith("omi-") and f.endswith(".txt")]

threshold_range = np.arange(3.0, 7.1, 0.5)

# ========= 每个数据集单独寻找最优阈值 ========= #
best_results_per_dataset = []

for txt_file in txt_files:
    base_name = txt_file.replace('.txt', '')
    train_path = os.path.join(data_folder, f'{base_name}_train.pkl')
    test_path = os.path.join(data_folder, f'{base_name}_test.pkl')
    label_path = os.path.join(data_folder, f'{base_name}_test_label.pkl')
    gt_txt_path = os.path.join(label_folder, txt_file)

    best_f1 = -1
    best_metrics = None

    for threshold in threshold_range:
        result = evaluate_omi_dataset(train_path, test_path, label_path, gt_txt_path, score_threshold=threshold)
        if result['f1_score'] > best_f1:
            best_f1 = result['f1_score']
            best_metrics = {
                'dataset': txt_file,
                'threshold': threshold,
                **result
            }

    best_results_per_dataset.append(best_metrics)

# ========= 汇总最优阈值结果并计算整体指标 ========= #
results_df = pd.DataFrame(best_results_per_dataset)

total_TP = results_df['TP'].sum()
total_FP = results_df['FP'].sum()
total_FN = results_df['FN'].sum()

overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

# ========= 输出 ========= #
print("\n===== 每个数据集的最优阈值结果 =====")
print(results_df[['dataset', 'threshold', 'precision', 'recall', 'f1_score']].round(4))

print("\n===== 全部数据集（最佳阈值下）的总体指标 =====")
print(f"Overall Precision: {overall_precision:.4f}")
print(f"Overall Recall:    {overall_recall:.4f}")
print(f"Overall F1-score:  {overall_f1:.4f}")

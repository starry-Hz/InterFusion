'''
===== 所有数据集评估结果 =====
       dataset  precision  recall  f1_score
0    omi-1.txt     0.7103  0.8539    0.7755
1   omi-10.txt     0.4917  0.9672    0.6519
2   omi-11.txt     0.3434  0.9828    0.5089
3   omi-12.txt     0.6491  0.9250    0.7629
4    omi-2.txt     0.5082  1.0000    0.6739
5    omi-3.txt     0.8261  0.7308    0.7755
6    omi-4.txt     0.8442  0.9701    0.9028
7    omi-5.txt     0.4588  0.7959    0.5821
8    omi-6.txt     0.2614  0.9091    0.4061
9    omi-7.txt     0.9048  0.9744    0.9383
10   omi-8.txt     0.3500  0.9032    0.5045
11   omi-9.txt     0.4962  0.9155    0.6436

===== 全部数据集的总体指标 =====
Overall Precision: 0.5157
Overall Recall:    0.8997
Overall F1-score:  0.6556

===== 不同得分阈值下的整体指标评估 =====
   score_threshold  precision  recall  f1_score
0              3.0     0.5142  0.9347    0.6634
1              3.5     0.5144  0.9210    0.6601
2              4.0     0.5185  0.9179    0.6626
3              4.5     0.5172  0.9119    0.6601
4              5.0     0.5157  0.8997    0.6556
5              5.5     0.5207  0.8967    0.6588
6              6.0     0.5217  0.8951    0.6592
7              6.5     0.5209  0.8906    0.6573
8              7.0     0.5215  0.8860    0.6565

✅ 最佳得分阈值推荐：
score_threshold    3.0000
precision          0.5142
recall             0.9347
f1_score           0.6634
Name: 0, dtype: float64
'''
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ========= 1. 主评估函数 ========= #
def evaluate_omi_dataset(train_path, test_path, label_path, gt_txt_path, window_size=5, score_threshold=5.0):
    train_data = pd.read_pickle(train_path)
    test_data = pd.read_pickle(test_path)
    test_labels = pd.read_pickle(label_path)
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

    # 提取异常时间段
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

    # 检测异常节点
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

    # 读取 ground truth
    gt_segments = []
    with open(gt_txt_path, 'r') as f:
        for line in f:
            time_range, sensors_str = line.strip().split(':')
            start, end = map(int, time_range.split('-'))
            sensors = list(map(int, sensors_str.split(',')))
            gt_segments.append({'start': start, 'end': end, 'sensors': sensors})

    # 每段评估
    segment_results = []
    total_TP = total_FP = total_FN = 0
    for segment in gt_segments:
        start, end = segment['start'], segment['end']
        gt_sensors = set(segment['sensors'])
        predicted = detect_anomalous_nodes(start, end)

        TP = len(predicted & gt_sensors)
        FP = len(predicted - gt_sensors)
        FN = len(gt_sensors - predicted)
        TN = num_sensors - (TP + FP + FN)

        total_TP += TP
        total_FP += FP
        total_FN += FN

        segment_results.append({'start': start, 'end': end, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN})

    # 汇总指标
    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'dataset': os.path.basename(gt_txt_path),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'segment_results': pd.DataFrame(segment_results)
    }

# ========= 2. 路径和初始化 ========= #
label_folder = './data/interpretation_label'
data_folder = './data/processed'
txt_files = [f for f in os.listdir(label_folder) if f.startswith("omi-") and f.endswith(".txt")]

# ========= 3. 批量评估所有数据集 ========= #
results_summary = []
total_TP = total_FP = total_FN = 0

for txt_file in txt_files:
    base_name = txt_file.replace('.txt', '')
    train_path = os.path.join(data_folder, f'{base_name}_train.pkl')
    test_path = os.path.join(data_folder, f'{base_name}_test.pkl')
    label_path = os.path.join(data_folder, f'{base_name}_test_label.pkl')
    gt_txt_path = os.path.join(label_folder, txt_file)

    result = evaluate_omi_dataset(train_path, test_path, label_path, gt_txt_path, score_threshold=5.0)
    seg_df = result['segment_results']

    results_summary.append({
        'dataset': result['dataset'],
        'precision': result['precision'],
        'recall': result['recall'],
        'f1_score': result['f1_score']
    })

    total_TP += seg_df['TP'].sum()
    total_FP += seg_df['FP'].sum()
    total_FN += seg_df['FN'].sum()

summary_df = pd.DataFrame(results_summary)

# ========= 4. 总体指标输出 ========= #
overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

print("\n===== 所有数据集评估结果 =====")
print(summary_df.round(4))
print("\n===== 全部数据集的总体指标 =====")
print(f"Overall Precision: {overall_precision:.4f}")
print(f"Overall Recall:    {overall_recall:.4f}")
print(f"Overall F1-score:  {overall_f1:.4f}")

# ========= 5. 动态评估不同阈值下的整体表现 ========= #
def dynamic_threshold_evaluation(txt_files, data_folder, label_folder, threshold_range):
    threshold_metrics = []
    for threshold in threshold_range:
        total_TP = total_FP = total_FN = 0
        for txt_file in txt_files:
            base_name = txt_file.replace('.txt', '')
            train_path = os.path.join(data_folder, f'{base_name}_train.pkl')
            test_path = os.path.join(data_folder, f'{base_name}_test.pkl')
            label_path = os.path.join(data_folder, f'{base_name}_test_label.pkl')
            gt_txt_path = os.path.join(label_folder, txt_file)

            result = evaluate_omi_dataset(train_path, test_path, label_path, gt_txt_path, score_threshold=threshold)
            seg_df = result['segment_results']
            total_TP += seg_df['TP'].sum()
            total_FP += seg_df['FP'].sum()
            total_FN += seg_df['FN'].sum()

        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        threshold_metrics.append({
            'score_threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })

    return pd.DataFrame(threshold_metrics)

threshold_range = np.arange(3.0, 7.1, 0.5)
threshold_df = dynamic_threshold_evaluation(txt_files, data_folder, label_folder, threshold_range)

print("\n===== 不同得分阈值下的整体指标评估 =====")
print(threshold_df.round(4))

best_row = threshold_df.loc[threshold_df['f1_score'].idxmax()]
print("\n✅ 最佳得分阈值推荐：")
print(best_row.round(4))

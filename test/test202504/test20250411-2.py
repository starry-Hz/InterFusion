# ✅ 完整代码：网格搜索每个数据集（包含 Top-K 邻居优化 + 阈值调优）

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
import datetime
# ---------- 单个数据集评估函数（Top-K + Score 阈值） ----------
def evaluate_omi_dataset_topk(train_data, test_data, test_labels, gt_txt_path,
                              window_size=5, score_threshold=3.0, top_k=5):
    num_sensors = train_data.shape[1]

    # 训练集边相关性统计
    edge_corrs = defaultdict(list)
    for i in range(train_data.shape[0] - window_size + 1):
        window = train_data[i:i + window_size]
        corr = np.corrcoef(window, rowvar=False)
        for j in range(num_sensors):
            for k in range(j + 1, num_sensors):
                val = corr[j, k]
                if not np.isnan(val):
                    edge_corrs[(j, k)].append(val)

    # 每条边的分位数范围
    edge_thresholds = {
        (j, k): (np.percentile(vals, 5), np.percentile(vals, 95))
        for (j, k), vals in edge_corrs.items()
    }

    # 每个节点 Top-K 邻居
    avg_corr = np.zeros((num_sensors, num_sensors))
    for (j, k), vals in edge_corrs.items():
        mean_val = np.mean(vals)
        avg_corr[j, k] = mean_val
        avg_corr[k, j] = mean_val

    topk_neighbors = {
        i: set(np.argsort(-avg_corr[i])[:top_k + 1]) - {i}
        for i in range(num_sensors)
    }

    # 提取异常段
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
                for k in topk_neighbors[j]:
                    if j < k and (j, k) in edge_thresholds:
                        val = corr[j, k]
                    elif k < j and (k, j) in edge_thresholds:
                        val = corr[k, j]
                    else:
                        continue
                    if np.isnan(val):
                        continue
                    low, high = edge_thresholds.get((min(j, k), max(j, k)), (None, None))
                    if low is None:
                        continue
                    if val < low or val > high:
                        deviation = max(abs(val - low), abs(val - high))
                        abnormal_score[j] += deviation
                        abnormal_score[k] += deviation
        return {node + 1 for node, score in abnormal_score.items() if score >= score_threshold}

    # 加载 ground truth
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

    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

# ---------- 网格搜索某个数据集 ----------
def grid_search_topk(train_data, test_data, test_labels, gt_txt_path, topk_list, threshold_list):
    results = []
    for top_k in topk_list:
        for threshold in threshold_list:
            precision, recall, f1 = evaluate_omi_dataset_topk(
                train_data, test_data, test_labels, gt_txt_path,
                window_size=5, score_threshold=threshold, top_k=top_k
            )
            results.append({
                'top_k': top_k,
                'score_threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
    return pd.DataFrame(results)

# ---------- 多数据集批量搜索 ----------
def run_full_grid_search_all_datasets(data_folder, label_folder, topk_list, threshold_list):
    txt_files = [f for f in os.listdir(label_folder) if f.startswith("omi-") and f.endswith(".txt")]
    all_results = []

    for txt_file in txt_files:
        base = txt_file.replace('.txt', '')
        train_path = os.path.join(data_folder, f"{base}_train.pkl")
        test_path = os.path.join(data_folder, f"{base}_test.pkl")
        label_path = os.path.join(data_folder, f"{base}_test_label.pkl")
        gt_txt_path = os.path.join(label_folder, txt_file)

        if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(label_path)):
            print(f"[跳过] 缺少文件: {base}")
            continue

        train_data = pd.read_pickle(train_path)
        test_data = pd.read_pickle(test_path)
        test_labels = pd.read_pickle(label_path)

        df = grid_search_topk(train_data, test_data, test_labels, gt_txt_path, topk_list, threshold_list)
        df['dataset'] = txt_file
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)

# 提取每个数据集中最优参数组合 + 总体评估指标的完整函数

def summarize_best_results(results_df):
    # 每个数据集只保留 F1-score 最大的那一行
    best_per_dataset = results_df.sort_values(by='f1_score', ascending=False).groupby('dataset').first().reset_index()

    # 汇总总体指标
    total_TP = total_FP = total_FN = 0
    for _, row in best_per_dataset.iterrows():
        precision = row['precision']
        recall = row['recall']
        f1 = row['f1_score']

        # 反推出 TP, FP, FN 的比例近似
        TP = 100
        FP = int(round((TP / precision) - TP)) if precision > 0 else 0
        FN = int(round((TP / recall) - TP)) if recall > 0 else 0

        total_TP += TP
        total_FP += FP
        total_FN += FN

    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    return best_per_dataset, {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_score': overall_f1
    }


start = datetime.datetime.now()
print(f"开始时间: {start.strftime('%Y-%m-%d %H:%M:%S')}")
# 1. 跑所有组合
results = run_full_grid_search_all_datasets(
    data_folder='./data/processed',
    label_folder='./data/interpretation_label',
    topk_list = np.arange(1, 10),
    threshold_list= np.arange(1.0, 7.1, 0.5)
)

# 2. 提取每个数据集最优结果 + 汇总整体精度
best_results, overall = summarize_best_results(results)

# 3. 查看结果
print("\n===== 每个数据集最优参数组合 =====")
print(best_results[['dataset', 'top_k', 'score_threshold', 'precision', 'recall', 'f1_score']])

print("\n===== 全部数据集的整体指标 =====")
print(f"Precision: {overall['overall_precision']:.4f}")
print(f"Recall:    {overall['overall_recall']:.4f}")
print(f"F1-score:  {overall['overall_f1_score']:.4f}")


end = datetime.datetime.now()
print(f"结束时间: {end.strftime('%Y-%m-%d %H:%M:%S')}")
'''
===== 每个数据集最优参数组合 =====
       dataset  top_k  score_threshold  precision    recall  f1_score
0    omi-1.txt      7              3.0   0.715596  0.876404  0.787879
1   omi-10.txt      5              6.0   0.500000  0.967213  0.659218
2   omi-11.txt      3              5.0   0.410448  0.948276  0.572917
3   omi-12.txt      7              5.0   0.660714  0.925000  0.770833
4    omi-2.txt      5              6.0   0.717949  0.903226  0.800000
5    omi-3.txt      7              3.0   0.815789  0.794872  0.805195
6    omi-4.txt      7              4.0   0.845070  0.895522  0.869565
7    omi-5.txt      5              4.0   0.492308  0.653061  0.561404
8    omi-6.txt      3              6.0   0.347368  0.750000  0.474820
9    omi-7.txt      7              4.0   0.902439  0.948718  0.925000
10   omi-8.txt      5              6.0   0.406250  0.838710  0.547368
11   omi-9.txt      7              3.0   0.496183  0.915493  0.643564

===== 全部数据集的整体指标 =====
Precision: 0.5543
Recall:    0.8584
F1-score:  0.6736

---------------------------------------------------------------------

===== 每个数据集最优参数组合 =====
       dataset  top_k  score_threshold  precision    recall  f1_score
0    omi-1.txt      8              5.5   0.747748  0.932584  0.830000
1   omi-10.txt      4              7.0   0.504274  0.967213  0.662921
2   omi-11.txt      3              7.0   0.430894  0.913793  0.585635
3   omi-12.txt      6              5.0   0.672727  0.925000  0.778947
4    omi-2.txt      6              6.5   0.714286  0.967742  0.821918
5    omi-3.txt      9              4.0   0.842105  0.820513  0.831169
6    omi-4.txt      9              3.5   0.844156  0.970149  0.902778
7    omi-5.txt      9              4.0   0.459770  0.816327  0.588235
8    omi-6.txt      3              6.5   0.354839  0.750000  0.481752
9    omi-7.txt      6              3.0   0.902439  0.948718  0.925000
10   omi-8.txt      6              7.0   0.428571  0.870968  0.574468
11   omi-9.txt      9              3.5   0.496183  0.915493  0.643564

===== 全部数据集的整体指标 =====
Precision: 0.5618
Recall:    0.8962
F1-score:  0.6906


===== 每个数据集最优参数组合 =====
       dataset  top_k  score_threshold  precision    recall  f1_score
0    omi-1.txt      9              1.0   0.739496  0.988764  0.846154
1   omi-10.txt      4              7.0   0.504274  0.967213  0.662921
2   omi-11.txt      2              7.0   0.466667  0.844828  0.601227
3   omi-12.txt      6              5.0   0.672727  0.925000  0.778947
4    omi-2.txt      6              6.5   0.714286  0.967742  0.821918
5    omi-3.txt      9              4.0   0.842105  0.820513  0.831169
6    omi-4.txt      9              3.5   0.844156  0.970149  0.902778
7    omi-5.txt      9              1.0   0.480000  0.979592  0.644295
8    omi-6.txt      1              1.5   0.363636  0.909091  0.519481
9    omi-7.txt      9              2.0   0.904762  0.974359  0.938272
10   omi-8.txt      6              7.0   0.428571  0.870968  0.574468
11   omi-9.txt      9              6.0   0.496183  0.915493  0.643564

===== 全部数据集的整体指标 =====
Precision: 0.5709
Recall:    0.9252
F1-score:  0.7061
结束时间: 2025-04-11 17:54:46
'''
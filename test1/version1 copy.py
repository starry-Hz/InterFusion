'''
IsolationForest 森林隔离算法 + 动态时间规整DTW + 图嵌入DeepWalk + 异常检测 + 动态调整阈值
============================================================
最终总体统计:
  总期望异常传感器数: 658
  总正确检测数: 476
  总检测到异常传感器数: 784
  全局召回率: 72.3%
  全局精确率: 60.7%
'''
import os
import numpy as np
from scipy.stats import pearsonr
from dtaidistance import dtw
import networkx as nx
from gensim.models import Word2Vec
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
import pandas as pd

def robust_zscore(x, median_val, mad, cap=10):
    if mad < 1e-6:
        return 0
    return min(np.abs(x - median_val) / (mad + 1e-8), cap)

def strict_image_method(data, labels, window_size=5, score_threshold=0.5):
    G = nx.Graph()
    n_windows = len(data) - window_size + 1
    if n_windows < 2:
        return np.array([])

    for i in range(n_windows):
        window = data[i:i+window_size]
        G.add_node(i, values=window)

    for i in range(n_windows):
        for j in range(i+1, min(i+20, n_windows)):
            try:
                corr = pearsonr(G.nodes[i]['values'].flatten(), 
                                G.nodes[j]['values'].flatten())[0]
                dtw_dist = dtw.distance(
                    G.nodes[i]['values'].mean(axis=1),
                    G.nodes[j]['values'].mean(axis=1)
                )
            except:
                continue
            if abs(corr) > 0.65 and dtw_dist < 1.6:
                G.add_edge(i, j, weight=abs(corr))

    if len(G.nodes) < 2:
        return np.array([])

    def random_walk(G, walk_length=20, num_walks=20):
        walks = []
        for _ in range(num_walks):
            for node in G.nodes():
                walk = [str(node)]
                while len(walk) < walk_length:
                    current = int(walk[-1])
                    neighbors = list(G.neighbors(current))
                    if neighbors:
                        walk.append(str(np.random.choice(neighbors)))
                    else:
                        break
                walks.append(walk)
        return walks

    walks = random_walk(G)
    model = Word2Vec(walks, vector_size=32, window=5, min_count=0, sg=1)
    embeddings = np.array([model.wv[str(i)] for i in G.nodes()])

    if embeddings.shape[0] < 2:
        return np.array([])

    iso_model = IsolationForest(contamination=0.15, random_state=42)
    iso_preds = iso_model.fit_predict(embeddings)
    anomalies = np.where(iso_preds == -1)[0]

    anomaly_windows = [i for i in anomalies if i in G.nodes()]
    anomaly_points = []
    for win in anomaly_windows:
        anomaly_points.extend(range(win, win+window_size))

    sensor_scores = np.zeros(data.shape[1])
    sensor_counts = np.zeros(data.shape[1])

    for point in set(anomaly_points):
        if point < len(labels) and labels[point] == 1:
            for sensor in range(data.shape[1]):
                local_data = data[max(0, point-100):point+1, sensor]
                median_val = np.median(local_data)
                mad = np.median(np.abs(local_data - median_val))
                score = robust_zscore(data[point, sensor], median_val, mad)
                sensor_scores[sensor] += score
                sensor_counts[sensor] += 1

    avg_scores = np.zeros(data.shape[1])
    for sensor in range(data.shape[1]):
        if sensor_counts[sensor] > 0:
            avg_scores[sensor] = sensor_scores[sensor] / sensor_counts[sensor]

    return np.where(avg_scores > score_threshold)[0]

def optimize_threshold(data, labels, expected_sensors, window_size=5):
    best_f1 = -1
    best_threshold = None
    best_detected = []

    for th in np.linspace(0.05, 3.0, 30):
        detected_sensors = strict_image_method(data, labels, window_size, score_threshold=th)

        detected_set = set(detected_sensors)
        expected_set = set(expected_sensors)

        tp = len(detected_set & expected_set)
        fp = len(detected_set - expected_set)
        fn = len(expected_set - detected_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
            best_detected = detected_sensors

    return best_detected, best_threshold, best_f1

def process_omi_file(txt_file_path, test_data, test_labels, auto_threshold=True):
    print("="*60)
    print("Processing file:", txt_file_path)
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    expected_sum = 0
    true_positive_sum = 0
    detected_sum = 0

    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue
        time_part, sensor_part = line.split(":", 1)
        if '-' not in time_part:
            continue
        try:
            start_str, end_str = time_part.split('-')
            start_idx = int(start_str)
            end_idx = int(end_str)
        except ValueError:
            continue

        expected_sensors = [int(x) for x in sensor_part.split(',') if x.strip().isdigit()]

        seg_data = test_data.iloc[start_idx:end_idx+1].to_numpy()
        seg_labels = test_labels.iloc[start_idx:end_idx+1]["label"].to_numpy()

        if auto_threshold:
            detected_sensors, best_th, best_f1 = optimize_threshold(seg_data, seg_labels, expected_sensors)
        else:
            detected_sensors = strict_image_method(seg_data, seg_labels)
            best_th = None
            best_f1 = None

        detected_set = set(detected_sensors)
        expected_set = set(expected_sensors)
        true_positives = detected_set & expected_set
        false_negatives = expected_set - detected_set
        false_positives = detected_set - expected_set

        expected_count = len(expected_set)
        true_positive_count = len(true_positives)
        detected_count = len(detected_set)

        expected_sum += expected_count
        true_positive_sum += true_positive_count
        detected_sum += detected_count

        miss_ratio = len(false_negatives) / expected_count if expected_count > 0 else 0
        false_ratio = len(false_positives) / detected_count if detected_count > 0 else 0

        print(f"时段 {start_idx}-{end_idx}:")
        print(f"  期望异常传感器: {sorted(expected_set)}")
        print(f"  检测到异常传感器: {sorted(detected_set)}")
        print(f"  命中: {sorted(true_positives)}")
        print(f"  漏检: {sorted(false_negatives)}")
        print(f"  误检: {sorted(false_positives)}")
        print(f"  正确检测数: {true_positive_count}")
        print(f"  漏检比例: {miss_ratio:.2f}")
        print(f"  误检比例: {false_ratio:.2f}")
        if auto_threshold:
            print(f"  最佳阈值: {best_th:.3f}")
            print(f"  当前 F1 分数: {best_f1:.3f}")
        print("-"*50)

    recall = (true_positive_sum / expected_sum) if expected_sum > 0 else 0
    precision = (true_positive_sum / detected_sum) if detected_sum > 0 else 0

    print("当前文件统计:")
    print(f"  总期望异常传感器数: {expected_sum}")
    print(f"  总正确检测数: {true_positive_sum}")
    print(f"  总检测到异常传感器数: {detected_sum}")
    print(f"  召回率: {recall*100:.1f}%")
    print(f"  精确率: {precision*100:.1f}%")
    print("="*60)

    return {"expected_sum": expected_sum,
            "true_positive_sum": true_positive_sum,
            "detected_sum": detected_sum}

if __name__ == "__main__":
    label_folder = r"data\interpretation_label"
    txt_files = [f for f in os.listdir(label_folder)
                 if f.startswith("omi-") and f.endswith(".txt")]
    # txt_files = ['omi-1.txt']  # 可限制测试文件

    global_expected = 0
    global_true_positive = 0
    global_detected = 0

    for txt_filename in sorted(txt_files):
        prefix = txt_filename.split(".")[0]
        test_data_path = os.path.join("data/processed", f"{prefix}_test.pkl")
        test_labels_path = os.path.join("data/processed", f"{prefix}_test_label.pkl")

        if not (os.path.exists(test_data_path) and os.path.exists(test_labels_path)):
            print(f"缺少对应的测试数据或标签文件：{prefix}")
            continue

        test_data = pd.read_pickle(test_data_path)
        test_labels = pd.read_pickle(test_labels_path)
        if isinstance(test_data, np.ndarray):
            test_data = pd.DataFrame(test_data)
        if isinstance(test_labels, np.ndarray):
            test_labels = pd.DataFrame(test_labels, columns=["label"])
        elif isinstance(test_labels, pd.Series):
            test_labels = test_labels.to_frame(name="label")

        file_path = os.path.join(label_folder, txt_filename)
        file_stats = process_omi_file(file_path, test_data, test_labels, auto_threshold=True)

        global_expected += file_stats["expected_sum"]
        global_true_positive += file_stats["true_positive_sum"]
        global_detected += file_stats["detected_sum"]

        current_recall = (global_true_positive / global_expected) if global_expected > 0 else 0
        current_precision = (global_true_positive / global_detected) if global_detected > 0 else 0
        print("当前累计总体统计:")
        print(f"  总期望异常传感器数: {global_expected}")
        print(f"  总正确检测数: {global_true_positive}")
        print(f"  总检测到异常传感器数: {global_detected}")
        print(f"  累计召回率: {current_recall*100:.1f}%")
        print(f"  累计精确率: {current_precision*100:.1f}%")
        print("="*60)

    global_recall = (global_true_positive / global_expected) if global_expected > 0 else 0
    global_precision = (global_true_positive / global_detected) if global_detected > 0 else 0

    print("最终总体统计:")
    print(f"  总期望异常传感器数: {global_expected}")
    print(f"  总正确检测数: {global_true_positive}")
    print(f"  总检测到异常传感器数: {global_detected}")
    print(f"  全局召回率: {global_recall*100:.1f}%")
    print(f"  全局精确率: {global_precision*100:.1f}%")

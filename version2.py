import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr
from dtaidistance import dtw
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import community as community_louvain  # 需要安装 python-louvain
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def robust_zscore(x, median_val, mad):
    """鲁棒 Z 分数：(x - median) / (MAD + 1e-8)"""
    return np.abs(x - median_val) / (mad + 1e-8)

def build_sensor_graph(data, corr_threshold=0.4, dtw_threshold=2.0):
    """
    构建传感器关系图：
    - 节点：每个传感器（共 n 个）
    - 边：基于传感器时间序列的相似性（Pearson + DTW）
    
    参数：
        data : np.ndarray, 形状为 (m, n) 的时序数据（m 时间点，n 传感器）
    """
    n_sensors = data.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_sensors))
    
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            x_i = data[:, i]
            x_j = data[:, j]
            
            # 计算 Pearson 相关系数和 DTW 距离
            # Pearson 相关系数：衡量“整体线性趋势是否一致”
            # DTW（动态时间规整）：衡量“形状是否相似（哪怕时间上错开）”
            # 例如，两个测量同一设备的传感器可能因为传输延迟导致数据偏移几秒，DTW仍能识别它们的相似性，而Pearson相关系数可能较低。
            corr = pearsonr(x_i, x_j)[0]
            dtw_dist = dtw.distance(x_i, x_j)
            
            # 添加边条件
            if abs(corr) > corr_threshold and dtw_dist < dtw_threshold:
                G.add_edge(i, j, weight=abs(corr))
    return G

def detect_anomaly_sensors(data, labels, method="kmeans", score_threshold=0.5):
    """
    检测异常传感器：
    1. 构建传感器关系图。
    2. 使用社区检测或聚类找到异常传感器组。
    3. 计算鲁棒得分并筛选最终异常传感器。
    
    参数：
        method : "louvain"（社区检测）或 "kmeans"（聚类）
    """
    # 1. 构建传感器关系图
    G = build_sensor_graph(data)
    print(f"图中节点数: {G.number_of_nodes()}")
    print(f"图中边数: {G.number_of_edges()}")

    
    # 2. 检测异常传感器组
    if method == "louvain":
        partition = community_louvain.best_partition(G)
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)
        # 假设异常社区是节点数最少的社区
        anomaly_sensors = min(communities.values(), key=len)
    elif method == "kmeans":
        # 使用 DeepWalk 生成节点嵌入
        def random_walk(G, walk_length=10, num_walks=20):
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
        model = Word2Vec(walks, vector_size=16, window=5, min_count=0, sg=1)
        embeddings = np.array([model.wv[str(i)] for i in G.nodes()])

        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        print('clusters:', clusters)
        # np.bincount(clusters) 返回每个类别的样本数,np.argmin()返回最小值的索引
        anomaly_sensors = np.where(clusters == np.argmin(np.bincount(clusters)))[0]
        # anomaly_sensors = clusters
    else:
        raise ValueError("Method must be 'louvain' or 'kmeans'")
    
    # 3. 计算异常传感器的鲁棒得分
    sensor_scores = np.zeros(data.shape[1])     # 初始化传感器得分数组
    anomaly_points = np.where(labels == 1)[0]   # 获取异常数据点的索引
    
    for sensor in anomaly_sensors:
        x = data[:, sensor]
        median_val = np.median(x)
        mad = np.median(np.abs(x - median_val))
        
        for point in anomaly_points:
            score = robust_zscore(data[point, sensor], median_val, mad)
            sensor_scores[sensor] += score
    
    avg_scores = sensor_scores / (len(anomaly_points) + 1e-8)
    # return np.where(avg_scores > score_threshold)[0]
    return anomaly_sensors

def process_omi_file(txt_file_path, test_data, test_labels):
    """
    处理单个标注文件并评估检测效果。
    返回统计结果字典。
    """
    print("=" * 60)
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
        start_idx, end_idx = map(int, time_part.split('-'))
        expected_sensors = [int(x) for x in sensor_part.split(',') if x.strip() != '']
        
        # 提取当前时段的数据和标签
        seg_data = test_data.iloc[start_idx:end_idx+1].values
        seg_labels = test_labels.iloc[start_idx:end_idx+1]["label"].values
        
        # 检测异常传感器
        detected_sensors = detect_anomaly_sensors(seg_data, seg_labels, method="kmeans")
        
        # 统计结果
        expected_set = set(expected_sensors)
        detected_set = set(detected_sensors)
        true_positives = expected_set & detected_set
        false_negatives = expected_set - detected_set
        false_positives = detected_set - expected_set
        
        expected_sum += len(expected_set)
        true_positive_sum += len(true_positives)
        detected_sum += len(detected_set)
        
        print(f"时段 {start_idx}-{end_idx}:")
        print(f"  期望异常传感器: {sorted(expected_set)}")
        print(f"  检测到异常传感器: {sorted(detected_set)}")
        print(f"  命中: {sorted(true_positives)}")
        print(f"  漏检: {sorted(false_negatives)}")
        print(f"  误检: {sorted(false_positives)}")
        print("-" * 50)
    
    # 计算召回率和精确率
    recall = true_positive_sum / expected_sum if expected_sum > 0 else 0
    precision = true_positive_sum / detected_sum if detected_sum > 0 else 0
    
    print("当前文件统计:")
    print(f"  总期望异常传感器数: {expected_sum}")
    print(f"  总正确检测数: {true_positive_sum}")
    print(f"  总检测到异常传感器数: {detected_sum}")
    print(f"  召回率: {recall * 100:.1f}%")
    print(f"  精确率: {precision * 100:.1f}%")
    print("=" * 60)
    
    return {
        "expected_sum": expected_sum,
        "true_positive_sum": true_positive_sum,
        "detected_sum": detected_sum
    }

if __name__ == "__main__":
    # 配置路径
    label_folder = "data/interpretation_label"
    txt_files = [f for f in os.listdir(label_folder) 
                 if f.startswith("omi-") and f.endswith(".txt")]
    txt_files = ['omi-1.txt']
    global_expected = 0
    global_true_positive = 0
    global_detected = 0
    
    for txt_filename in sorted(txt_files):
        prefix = txt_filename.split(".")[0]  # 例如 "omi-1"
        test_data_path = f"data/processed/{prefix}_test.pkl"
        test_labels_path = f"data/processed/{prefix}_test_label.pkl"
        
        if not (os.path.exists(test_data_path) and os.path.exists(test_labels_path)):
            print(f"缺少对应的测试数据或标签文件：{prefix}")
            continue
        
        # 加载数据
        test_data = pd.read_pickle(test_data_path)
        test_labels = pd.read_pickle(test_labels_path)
        if isinstance(test_data, np.ndarray):
            test_data = pd.DataFrame(test_data)
        if isinstance(test_labels, np.ndarray):
            test_labels = pd.DataFrame(test_labels, columns=["label"])
        elif isinstance(test_labels, pd.Series):
            test_labels = test_labels.to_frame(name="label")
        
        # 处理文件并累计统计
        file_path = os.path.join(label_folder, txt_filename)
        file_stats = process_omi_file(file_path, test_data, test_labels)
        
        global_expected += file_stats["expected_sum"]
        global_true_positive += file_stats["true_positive_sum"]
        global_detected += file_stats["detected_sum"]
        
        # 输出当前累计结果
        current_recall = global_true_positive / global_expected if global_expected > 0 else 0
        current_precision = global_true_positive / global_detected if global_detected > 0 else 0
        
        print("当前累计统计:")
        print(f"  总期望异常传感器数: {global_expected}")
        print(f"  总正确检测数: {global_true_positive}")
        print(f"  总检测到异常传感器数: {global_detected}")
        print(f"  累计召回率: {current_recall * 100:.1f}%")
        print(f"  累计精确率: {current_precision * 100:.1f}%")
        print("=" * 60)
    
    # 最终全局统计
    global_recall = global_true_positive / global_expected if global_expected > 0 else 0
    global_precision = global_true_positive / global_detected if global_detected > 0 else 0
    
    print("最终全局统计:")
    print(f"  总期望异常传感器数: {global_expected}")
    print(f"  总正确检测数: {global_true_positive}")
    print(f"  总检测到异常传感器数: {global_detected}")
    print(f"  全局召回率: {global_recall * 100:.1f}%")
    print(f"  全局精确率: {global_precision * 100:.1f}%")
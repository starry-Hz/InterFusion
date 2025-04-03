import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr
from dtaidistance import dtw
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import community as community_louvain
import logging

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("grid_search.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def robust_zscore(x, median_val, mad):
    return np.abs(x - median_val) / (mad + 1e-8)

def build_sensor_graph(data, corr_threshold=0.4, dtw_threshold=2.0):
    n_sensors = data.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_sensors))

    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            x_i = data[:, i]
            x_j = data[:, j]

            try:
                corr = pearsonr(x_i, x_j)[0]
            except:
                corr = 0
            dtw_dist = dtw.distance(x_i, x_j)

            if abs(corr) > corr_threshold and dtw_dist < dtw_threshold:
                G.add_edge(i, j, weight=abs(corr))
    return G

def detect_anomaly_sensors(data, labels, method="kmeans", score_threshold=0.5):
    G = build_sensor_graph(data)
    logging.info(f"图中节点数: {G.number_of_nodes()}，边数: {G.number_of_edges()}")

    if method == "louvain":
        partition = community_louvain.best_partition(G)
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)
        anomaly_sensors = min(communities.values(), key=len)
    elif method == "kmeans":
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
        anomaly_sensors = np.where(clusters == np.argmin(np.bincount(clusters)))[0]
    else:
        raise ValueError("Method must be 'louvain' or 'kmeans'")

    sensor_scores = np.zeros(data.shape[1])
    anomaly_points = np.where(labels == 1)[0]

    for sensor in anomaly_sensors:
        x = data[:, sensor]
        median_val = np.median(x)
        mad = np.median(np.abs(x - median_val))

        for point in anomaly_points:
            score = robust_zscore(data[point, sensor], median_val, mad)
            sensor_scores[sensor] += score

    avg_scores = sensor_scores / (len(anomaly_points) + 1e-8)
    return np.where(avg_scores > score_threshold)[0]

def grid_search_graph_params(label_file, test_data, test_labels):
    corr_range = [0.3, 0.4, 0.5]
    dtw_range = [1.2, 1.6, 2.0]
    score_range = np.round(np.arange(0.0, 1.01, 0.01), 2)   # 后面的2表示保留两位小数
    method = "kmeans"

    best_hit = 0
    best_expected = 0
    best_detected = 0

    for corr_thres in corr_range:
        for dtw_thres in dtw_range:
            def build_sensor_graph_custom(data):
                return build_sensor_graph(data, corr_threshold=corr_thres, dtw_threshold=dtw_thres)

            for score_threshold in score_range:
                total_expected = 0
                total_hit = 0
                total_detected = 0

                def detect_anomaly_sensors_custom(data, labels):
                    G = build_sensor_graph_custom(data)
                    return detect_anomaly_sensors(data, labels, method=method, score_threshold=score_threshold)

                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    time_part, sensor_part = line.split(":", 1)
                    start_idx, end_idx = map(int, time_part.split('-'))
                    expected_sensors = [int(x) for x in sensor_part.split(',') if x.strip() != '']

                    seg_data = test_data.iloc[start_idx:end_idx + 1].values
                    seg_labels = test_labels.iloc[start_idx:end_idx + 1]["label"].values

                    detected_sensors = detect_anomaly_sensors_custom(seg_data, seg_labels)
                    expected_set = set(expected_sensors)
                    detected_set = set(detected_sensors)

                    total_expected += len(expected_set)
                    total_hit += len(expected_set & detected_set)
                    total_detected += len(detected_set)

                recall = total_hit / total_expected if total_expected > 0 else 0
                precision = total_hit / total_detected if total_detected > 0 else 0

                logging.info(f"✅ corr={corr_thres}, dtw={dtw_thres}, score_thres={score_threshold:.2f} => "
                             f"召回率: {recall:.2%}, 精确率: {precision:.2%}，检测到: {total_detected}，命中: {total_hit}")

                if total_hit > best_hit:
                    best_hit = total_hit
                    best_expected = total_expected
                    best_detected = total_detected

    return {
        "expected_sum": best_expected,
        "true_positive_sum": best_hit,
        "detected_sum": best_detected
    }

if __name__ == "__main__":
    label_folder = "data/interpretation_label"
    txt_files = [f for f in os.listdir(label_folder) if f.startswith("omi-") and f.endswith(".txt")]
    txt_files = ['omi-1.txt']  # 如果只测试 omi-1.txt

    global_expected = 0
    global_true_positive = 0
    global_detected = 0

    for txt_filename in sorted(txt_files):
        prefix = txt_filename.split(".")[0]
        test_data_path = f"data/processed/{prefix}_test.pkl"
        test_labels_path = f"data/processed/{prefix}_test_label.pkl"

        if not (os.path.exists(test_data_path) and os.path.exists(test_labels_path)):
            logging.warning(f"缺少对应的测试数据或标签文件：{prefix}")
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
        file_stats = grid_search_graph_params(file_path, test_data, test_labels)

        global_expected += file_stats["expected_sum"]
        global_true_positive += file_stats["true_positive_sum"]
        global_detected += file_stats["detected_sum"]

        current_recall = global_true_positive / global_expected if global_expected > 0 else 0
        current_precision = global_true_positive / global_detected if global_detected > 0 else 0

        logging.info("当前累计统计:")
        logging.info(f"  总期望异常传感器数: {global_expected}")
        logging.info(f"  总正确检测数: {global_true_positive}")
        logging.info(f"  总检测到异常传感器数: {global_detected}")
        logging.info(f"  累计召回率: {current_recall * 100:.1f}%")
        logging.info(f"  累计精确率: {current_precision * 100:.1f}%")
        logging.info("=" * 60)

    global_recall = global_true_positive / global_expected if global_expected > 0 else 0
    global_precision = global_true_positive / global_detected if global_detected > 0 else 0

    logging.info("最终全局统计:")
    logging.info(f"  总期望异常传感器数: {global_expected}")
    logging.info(f"  总正确检测数: {global_true_positive}")
    logging.info(f"  总检测到异常传感器数: {global_detected}")
    logging.info(f"  全局召回率: {global_recall * 100:.1f}%")
    logging.info(f"  全局精确率: {global_precision * 100:.1f}%")

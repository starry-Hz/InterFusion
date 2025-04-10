import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import networkx as nx
from dtaidistance import dtw
from gensim.models import Word2Vec
from collections import defaultdict
import logging
import warnings

# 日志设置
current_date = datetime.now().strftime('%Y%m%d')
log_dir = './log'
os.makedirs(log_dir, exist_ok=True)
log_filename = f'{log_dir}/train_{current_date}.log'
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=log_filename,
                    filemode='a',
                    encoding='utf-8')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=RuntimeWarning)

class SensorRelationAnalyzer:
    def __init__(self, window_size=20, corr_thresh=0.6, dtw_thresh=1.5):
        self.window_size = window_size
        self.corr_thresh = corr_thresh
        self.dtw_thresh = dtw_thresh
        self.normal_graph = None
        self.anomaly_graph = None

    def build_relation_graphs(self, data, labels):
        logger.info(f"输入数据形状: {data.shape}, 传感器数量: {data.shape[1]}")

        normal_windows = self._extract_windows(data, labels, target_label=0)
        anomaly_windows = self._extract_windows(data, labels, target_label=1)

        logger.info(f"正常窗口数: {len(normal_windows)}, 异常窗口数: {len(anomaly_windows)}")

        self.normal_graph = self._build_graph(normal_windows)
        self.anomaly_graph = self._build_graph(anomaly_windows)

        logger.info(f"正常图: {len(self.normal_graph.nodes())} 节点, {len(self.normal_graph.edges())} 边")
        logger.info(f"异常图: {len(self.anomaly_graph.nodes())} 节点, {len(self.anomaly_graph.edges())} 边")

    def _extract_windows(self, data, labels, target_label=0):
        windows = []
        expected_dim = data.shape[1]

        for i in range(len(data) - self.window_size + 1):
            if labels[i + self.window_size - 1] == target_label:
                win = data[i:i + self.window_size]
                if win.shape[1] == expected_dim:
                    windows.append(win)
                else:
                    logger.warning(f"跳过维度异常窗口: 第{i}个窗口 shape={win.shape}")
        if not windows:
            logger.warning("没有符合条件的窗口被提取")

        windows = np.array(windows)
        logger.info(f"提取窗口数组形状: {windows.shape}")
        return windows

    def _build_graph(self, windows):
        if len(windows) == 0:
            logger.warning("无窗口数据，返回空图")
            return nx.Graph()

        n_sensors = windows.shape[2]
        logger.info(f"构建图中，传感器数量: {n_sensors}")

        G = nx.Graph()
        G.add_nodes_from(range(n_sensors))

        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                try:
                    corrs = []
                    valid_windows = 0

                    for win in windows:
                        if np.all(win[:, i] == win[0, i]) or np.all(win[:, j] == win[0, j]):
                            continue

                        corr = np.corrcoef(win[:, i], win[:, j])[0, 1]
                        dist = dtw.distance(win[:, i], win[:, j])

                        if not np.isnan(corr) and abs(corr) > self.corr_thresh and dist < self.dtw_thresh:
                            corrs.append(corr)
                            valid_windows += 1

                    if corrs:
                        avg_corr = np.mean(corrs)
                        G.add_edge(i, j, weight=avg_corr, count=valid_windows)

                except Exception as e:
                    logger.error(f"处理传感器对({i},{j})时出错: {str(e)}")

        return G



class DeepWalkDetector:
    def __init__(self, embed_size=32, window_size=20, corr_thresh=0.6, dtw_thresh=1.5):
        self.embed_size = embed_size
        self.window_size = window_size
        self.corr_thresh = corr_thresh
        self.dtw_thresh = dtw_thresh
        self.normal_embeddings = None
        self.anomaly_embeddings = None
        self.normal_graph = None
        self.anomaly_graph = None

    def train(self, normal_graph, anomaly_graph):
        logger.info("训练 DeepWalk 模型...")
        self.normal_model = self._train_deepwalk(normal_graph, "正常")
        self.anomaly_model = self._train_deepwalk(anomaly_graph, "异常")

        self.normal_embeddings = self._get_embeddings(normal_graph, self.normal_model)
        self.anomaly_embeddings = self._get_embeddings(anomaly_graph, self.anomaly_model)

    def _train_deepwalk(self, graph, graph_type):
        if len(graph.edges()) == 0:
            logger.warning(f"{graph_type}图没有边，跳过训练")
            return None

        logger.info(f"训练{graph_type}模型，节点数: {len(graph.nodes())}, 边数: {len(graph.edges())}")

        walks = []
        for _ in range(20):
            for node in graph.nodes():
                walk = [str(node)]
                while len(walk) < 20:
                    current = int(walk[-1])
                    neighbors = list(graph.neighbors(current))
                    if neighbors:
                        walk.append(str(np.random.choice(neighbors)))
                    else:
                        break
                walks.append(walk)

        return Word2Vec(walks, vector_size=self.embed_size, window=5, min_count=0, sg=1)

    def _get_embeddings(self, graph, model):
        if model is None:
            return np.zeros((len(graph.nodes()), self.embed_size))
        return np.array([model.wv[str(i)] for i in graph.nodes()])

    def detect_anomalies(self, test_data, threshold=0.8):
        logger.info("开始异常检测...")
        anomalies = []
        for i in range(len(test_data) - self.window_size + 1):
            window = test_data[i:i + self.window_size]
            window_anomalies = self._detect_window_anomalies(window, threshold)
            anomalies.append(window_anomalies)
        return anomalies

    def _detect_window_anomalies(self, window, threshold):
        n_sensors = window.shape[1]
        current_graph = nx.Graph()
        current_graph.add_nodes_from(range(n_sensors))

        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                try:
                    corr = np.corrcoef(window[:, i], window[:, j])[0, 1]
                    dist = dtw.distance(window[:, i], window[:, j])

                    if not np.isnan(corr) and abs(corr) > self.corr_thresh and dist < self.dtw_thresh:
                        current_graph.add_edge(i, j, weight=corr)

                except Exception as e:
                    logger.error(f"窗口检测中处理传感器对({i},{j})出错: {str(e)}")

        anomaly_scores = np.zeros(n_sensors)
        for sensor in range(n_sensors):
            try:
                if sensor not in current_graph.nodes():
                    continue
                neighbors = list(current_graph.neighbors(sensor))
                if not neighbors:
                    continue

                neighbor_feat = np.mean([current_graph[sensor][n]['weight'] for n in neighbors])

                normal_diff = 1.0
                anomaly_diff = 1.0

                if sensor in self.normal_graph.nodes() and list(self.normal_graph.neighbors(sensor)):
                    normal_neighbor_feat = np.mean([
                        self.normal_graph[sensor][n]['weight'] for n in self.normal_graph.neighbors(sensor)])
                    normal_diff = np.abs(neighbor_feat - normal_neighbor_feat)

                if sensor in self.anomaly_graph.nodes() and list(self.anomaly_graph.neighbors(sensor)):
                    anomaly_neighbor_feat = np.mean([
                        self.anomaly_graph[sensor][n]['weight'] for n in self.anomaly_graph.neighbors(sensor)])
                    anomaly_diff = np.abs(neighbor_feat - anomaly_neighbor_feat)

                anomaly_scores[sensor] = normal_diff / (anomaly_diff + 1e-6)

            except Exception as e:
                logger.error(f"计算传感器{sensor}异常得分时出错: {str(e)}")

        return np.where(anomaly_scores > threshold)[0]


def safe_load_pickle(path):
    logger.info(f"加载文件: {path}")
    data = pd.read_pickle(path)
    return data.values if isinstance(data, pd.DataFrame) else data

def load_and_preprocess(txt_file):
    logger.info("加载并预处理数据...")
    prefix = txt_file.split('.')[0]
    data_path = os.path.join("data/processed", f"{prefix}_test.pkl")
    label_path = os.path.join("data/processed", f"{prefix}_test_label.pkl")

    data = safe_load_pickle(data_path)
    labels = safe_load_pickle(label_path).reshape(-1)

    logger.info(f"原始数据形状: {data.shape}, 标签形状: {labels.shape}")

    train_size = int(len(data) * 0.7)
    train_data, test_data = data[:train_size], data[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]

    logger.info(f"训练集: {train_data.shape}, 测试集: {test_data.shape}")
    return train_data, train_labels, test_data, test_labels

def parse_ground_truth(txt_file):
    ground_truth = defaultdict(list)
    with open(os.path.join("data/interpretation_label", txt_file), 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            time_part, sensors = line.strip().split(':')
            start, end = map(int, time_part.split('-'))
            sensors = list(map(int, sensors.split(','))) if sensors else []
            for t in range(start, end + 1):
                ground_truth[t] = sensors
    return ground_truth

def evaluate(ground_truth, predictions, test_data):
    logger.info("开始评估...")
    stats = {'expected': 0, 'detected': 0, 'true_positives': 0, 'false_positives': 0}

    for t in range(len(test_data)):
        true_anomalies = ground_truth.get(t, [])
        pred_anomalies = predictions[t] if t < len(predictions) else []

        stats['expected'] += len(true_anomalies)
        stats['detected'] += len(pred_anomalies)
        stats['true_positives'] += len(set(true_anomalies) & set(pred_anomalies))
        stats['false_positives'] += len(set(pred_anomalies) - set(true_anomalies))

    recall = stats['true_positives'] / stats['expected'] if stats['expected'] else 0
    precision = stats['true_positives'] / stats['detected'] if stats['detected'] else 0

    logger.info(f"召回率: {recall * 100:.1f}%, 精确率: {precision * 100:.1f}%")
    logger.info(f"TP: {stats['true_positives']}, FP: {stats['false_positives']}, Detected: {stats['detected']}, Expected: {stats['expected']}")
    return stats

if __name__ == "__main__":
    WINDOW_SIZE = 20
    CORR_THRESH = 0.65
    DTW_THRESH = 1.6
    ANOMALY_THRESH = 0.90

    txt_files = [f for f in os.listdir("data/interpretation_label") if f.startswith("omi-") and f.endswith(".txt")]
    txt_file = txt_files[0] if txt_files else None

    if not txt_file:
        logger.error("未找到匹配的txt文件")
    else:
        try:
            train_data, train_labels, test_data, test_labels = load_and_preprocess(txt_file)

            analyzer = SensorRelationAnalyzer(WINDOW_SIZE, CORR_THRESH, DTW_THRESH) # 初始化传感器关系分析器
            analyzer.build_relation_graphs(train_data, train_labels)    # 构建正常和异常图

            detector = DeepWalkDetector()
            detector.window_size = WINDOW_SIZE
            detector.corr_thresh = CORR_THRESH
            detector.dtw_thresh = DTW_THRESH
            detector.normal_graph = analyzer.normal_graph
            detector.anomaly_graph = analyzer.anomaly_graph
            detector.train(analyzer.normal_graph, analyzer.anomaly_graph)

            predictions = detector.detect_anomalies(test_data, ANOMALY_THRESH)
            ground_truth = parse_ground_truth(txt_file)
            evaluate(ground_truth, predictions, test_data)

        except Exception as e:
            logger.exception(f"程序运行出错: {str(e)}")
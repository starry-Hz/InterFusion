import os
import numpy as np
import pandas as pd
import networkx as nx
from dtaidistance import dtw
from gensim.models import Word2Vec
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

class SensorRelationAnalyzer:
    def __init__(self, window_size=20, corr_thresh=0.6, dtw_thresh=1.5):
        self.window_size = window_size
        self.corr_thresh = corr_thresh
        self.dtw_thresh = dtw_thresh
        self.normal_graph = None
        self.anomaly_graph = None
    
    def build_relation_graphs(self, data, labels):
        """构建正常和异常关系图"""
        print(f"输入数据形状: {data.shape}, 传感器数量: {data.shape[1]}")
        
        normal_windows = self._extract_windows(data, labels, target_label=0)
        anomaly_windows = self._extract_windows(data, labels, target_label=1)
        
        print(f"正常窗口数: {len(normal_windows)}, 异常窗口数: {len(anomaly_windows)}")
        
        self.normal_graph = self._build_graph(normal_windows)
        self.anomaly_graph = self._build_graph(anomaly_windows)
        
        print(f"正常图: {len(self.normal_graph.nodes())}节点, {len(self.normal_graph.edges())}边")
        print(f"异常图: {len(self.anomaly_graph.nodes())}节点, {len(self.anomaly_graph.edges())}边")
    
    def _extract_windows(self, data, labels, target_label=0):
        """提取指定标签的滑动窗口"""
        windows = []
        for i in range(len(data) - self.window_size + 1):
            if labels[i + self.window_size - 1] == target_label:
                windows.append(data[i:i+self.window_size])
        return np.array(windows) if windows else np.array([])
    
    def _build_graph(self, windows):
        """从窗口数据构建关系图"""
        if len(windows) == 0:
            return nx.Graph()
        
        n_sensors = windows.shape[1]
        print(f"构建图中，传感器数量: {n_sensors}")
        
        G = nx.Graph()
        G.add_nodes_from(range(n_sensors))
        
        for i in range(n_sensors):
            for j in range(i+1, n_sensors):
                try:
                    # 计算所有窗口中该传感器对的平均相关性
                    corrs = []
                    dtw_dists = []
                    valid_windows = 0
                    
                    for win in windows:
                        # 处理全零或常数值的情况
                        if np.all(win[:, i] == win[0, i]) or np.all(win[:, j] == win[0, j]):
                            continue
                            
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            corr = np.corrcoef(win[:, i], win[:, j])[0, 1]
                            dist = dtw.distance(win[:, i], win[:, j])
                            
                        if not np.isnan(corr) and abs(corr) > self.corr_thresh and dist < self.dtw_thresh:
                            corrs.append(corr)
                            dtw_dists.append(dist)
                            valid_windows += 1
                    
                    if corrs:
                        avg_corr = np.mean(corrs)
                        G.add_edge(i, j, weight=avg_corr, count=valid_windows)
                        
                except Exception as e:
                    print(f"处理传感器对({i},{j})时出错: {str(e)}")
                    continue
        
        return G

class DeepWalkDetector:
    def __init__(self, embed_size=32):
        self.embed_size = embed_size
        self.normal_embeddings = None
        self.anomaly_embeddings = None
    
    def train(self, normal_graph, anomaly_graph):
        """训练正常和异常的DeepWalk模型"""
        print("\n训练DeepWalk模型...")
        self.normal_model = self._train_deepwalk(normal_graph, "正常")
        self.anomaly_model = self._train_deepwalk(anomaly_graph, "异常")
        
        # 提取所有传感器嵌入
        self.normal_embeddings = self._get_embeddings(normal_graph, self.normal_model)
        self.anomaly_embeddings = self._get_embeddings(anomaly_graph, self.anomaly_model)
    
    def _train_deepwalk(self, graph, graph_type):
        """训练单个DeepWalk模型"""
        if len(graph.edges()) == 0:
            print(f"{graph_type}图没有边，跳过训练")
            return None
        
        print(f"训练{graph_type}模型，节点数: {len(graph.nodes())}, 边数: {len(graph.edges())}")
        
        walks = []
        for _ in range(20):  # 每个节点20次随机游走
            for node in graph.nodes():
                walk = [str(node)]
                while len(walk) < 20:  # 每次游走20步
                    current = int(walk[-1])
                    neighbors = list(graph.neighbors(current))
                    if neighbors:
                        walk.append(str(np.random.choice(neighbors)))
                    else:
                        break
                walks.append(walk)
        
        return Word2Vec(walks, vector_size=self.embed_size, window=5, min_count=0, sg=1)
    
    def _get_embeddings(self, graph, model):
        """获取所有传感器的嵌入向量"""
        if model is None:
            return np.zeros((len(graph.nodes()), self.embed_size))
        return np.array([model.wv[str(i)] for i in graph.nodes()])
    
    def detect_anomalies(self, test_data, threshold=0.8):
        """检测测试数据中的异常传感器"""
        print("\n开始异常检测...")
        anomalies = []
        for i in range(len(test_data) - self.window_size + 1):
            window = test_data[i:i+self.window_size]
            window_anomalies = self._detect_window_anomalies(window, threshold)
            anomalies.append(window_anomalies)
        return anomalies
    
    def _detect_window_anomalies(self, window, threshold):
        """检测单个窗口中的异常传感器"""
        n_sensors = window.shape[1]
        current_graph = nx.Graph()
        current_graph.add_nodes_from(range(n_sensors))
        
        # 构建当前窗口的关系图
        for i in range(n_sensors):
            for j in range(i+1, n_sensors):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        corr = np.corrcoef(window[:, i], window[:, j])[0, 1]
                        dist = dtw.distance(window[:, i], window[:, j])
                    
                    if not np.isnan(corr) and abs(corr) > self.corr_thresh and dist < self.dtw_thresh:
                        current_graph.add_edge(i, j, weight=corr)
                        
                except Exception as e:
                    print(f"窗口检测中处理传感器对({i},{j})出错: {str(e)}")
                    continue
        
        # 计算每个传感器与正常/异常模式的相似度
        anomaly_scores = np.zeros(n_sensors)
        for sensor in range(n_sensors):
            try:
                if sensor not in current_graph.nodes():
                    continue
                    
                neighbors = list(current_graph.neighbors(sensor))
                if not neighbors:
                    continue
                    
                # 计算邻居平均相关性特征
                neighbor_feat = np.mean([current_graph[sensor][n]['weight'] for n in neighbors])
                
                # 计算与正常模式的差异
                if sensor in self.normal_graph.nodes() and list(self.normal_graph.neighbors(sensor)):
                    normal_neighbor_feat = np.mean(
                        [self.normal_graph[sensor][n]['weight'] for n in self.normal_graph.neighbors(sensor)]
                    )
                    normal_diff = np.abs(neighbor_feat - normal_neighbor_feat)
                else:
                    normal_diff = 1.0
                
                # 计算与异常模式的差异
                if sensor in self.anomaly_graph.nodes() and list(self.anomaly_graph.neighbors(sensor)):
                    anomaly_neighbor_feat = np.mean(
                        [self.anomaly_graph[sensor][n]['weight'] for n in self.anomaly_graph.neighbors(sensor)]
                    )
                    anomaly_diff = np.abs(neighbor_feat - anomaly_neighbor_feat)
                else:
                    anomaly_diff = 1.0
                
                anomaly_scores[sensor] = normal_diff / (anomaly_diff + 1e-6)
                
            except Exception as e:
                print(f"计算传感器{sensor}异常得分时出错: {str(e)}")
                continue
        
        return np.where(anomaly_scores > threshold)[0]

def safe_load_pickle(path):
    """安全加载pickle文件"""
    data = pd.read_pickle(path)
    if isinstance(data, pd.DataFrame):
        return data.values
    return data

def load_and_preprocess(txt_file):
    """加载并预处理数据"""
    print("\n加载数据...")
    prefix = txt_file.split('.')[0]
    data_path = os.path.join("data/processed", f"{prefix}_test.pkl")
    label_path = os.path.join("data/processed", f"{prefix}_test_label.pkl")
    
    # 加载数据
    data = safe_load_pickle(data_path)
    labels = safe_load_pickle(label_path).reshape(-1)
    
    print(f"原始数据形状: {data.shape}, 标签形状: {labels.shape}")
    
    # 划分训练测试集
    train_size = int(len(data) * 0.7)
    train_data, test_data = data[:train_size], data[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]
    
    print(f"训练集: {train_data.shape}, 测试集: {test_data.shape}")
    
    return train_data, train_labels, test_data, test_labels

def parse_ground_truth(txt_file):
    """解析真实异常标注"""
    ground_truth = defaultdict(list)
    with open(os.path.join("data/interpretation_label", txt_file), 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            time_part, sensors = line.strip().split(':')
            start, end = map(int, time_part.split('-'))
            sensors = list(map(int, sensors.split(','))) if sensors else []
            for t in range(start, end+1):
                ground_truth[t] = sensors
    return ground_truth

def evaluate(ground_truth, predictions, test_data):
    """评估检测结果"""
    stats = {
        'expected': 0,
        'detected': 0,
        'true_positives': 0,
        'false_positives': 0
    }
    
    for t in range(len(test_data)):
        true_anomalies = ground_truth.get(t, [])
        pred_anomalies = predictions[t] if t < len(predictions) else []
        
        stats['expected'] += len(true_anomalies)
        stats['detected'] += len(pred_anomalies)
        
        true_set = set(true_anomalies)
        pred_set = set(pred_anomalies)
        
        stats['true_positives'] += len(true_set & pred_set)
        stats['false_positives'] += len(pred_set - true_set)
    
    recall = stats['true_positives'] / stats['expected'] if stats['expected'] > 0 else 0
    precision = stats['true_positives'] / stats['detected'] if stats['detected'] > 0 else 0
    
    print("\n评估结果:")
    print(f"  总期望异常传感器数: {stats['expected']}")
    print(f"  总检测到异常传感器数: {stats['detected']}")
    print(f"  正确检测数: {stats['true_positives']}")
    print(f"  误检数: {stats['false_positives']}")
    print(f"  召回率: {recall*100:.1f}%")
    print(f"  精确率: {precision*100:.1f}%")
    
    return stats

if __name__ == "__main__":
    # 参数配置
    WINDOW_SIZE = 20
    CORR_THRESH = 0.65
    DTW_THRESH = 1.6
    ANOMALY_THRESH = 0.75
    
    # 选择要处理的文件
    txt_files = [f for f in os.listdir("data/interpretation_label") 
                if f.startswith("omi-") and f.endswith(".txt")]
    txt_file = txt_files[0]  # 以第一个文件为例
    
    try:
        # 1. 加载和预处理数据
        train_data, train_labels, test_data, test_labels = load_and_preprocess(txt_file)
        
        # 2. 构建关系图
        analyzer = SensorRelationAnalyzer(WINDOW_SIZE, CORR_THRESH, DTW_THRESH)
        analyzer.build_relation_graphs(train_data, train_labels)
        
        # 3. 训练检测器
        detector = DeepWalkDetector()
        detector.train(analyzer.normal_graph, analyzer.anomaly_graph)
        
        # 4. 在测试集上检测
        predictions = detector.detect_anomalies(test_data, ANOMALY_THRESH)
        
        # 5. 解析真实标注并评估
        ground_truth = parse_ground_truth(txt_file)
        evaluate(ground_truth, predictions, test_data)
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
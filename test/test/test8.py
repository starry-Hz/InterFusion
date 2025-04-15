# 改进后代码：深度学习 + 图结构模型，识别异常传感器
import os
import numpy as np
import pandas as pd
import networkx as nx
from dtaidistance import dtw
from collections import defaultdict
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import warnings
from datetime import datetime

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

class TimeWindowGraphBuilder:
    def __init__(self, window_size=20, corr_thresh=0.6, dtw_thresh=1.5):
        self.window_size = window_size
        self.corr_thresh = corr_thresh
        self.dtw_thresh = dtw_thresh

    def build_graph(self, data):
        G = nx.Graph()
        n_windows = len(data) - self.window_size + 1
        for t in range(n_windows):
            window = data[t:t + self.window_size]
            G.add_node(t, values=window.copy())

        for i in range(n_windows):
            for j in range(i + 1, n_windows):
                v1 = G.nodes[i]['values'].flatten()
                v2 = G.nodes[j]['values'].flatten()

                corr = np.corrcoef(v1, v2)[0, 1]
                dist = dtw.distance(v1, v2)

                if not np.isnan(corr) and abs(corr) > self.corr_thresh and dist < self.dtw_thresh:
                    G.add_edge(i, j, weight=abs(corr))
        return G

class DeepWalkEmbedding:
    def __init__(self, embed_size=32):
        self.embed_size = embed_size

    def learn_embeddings(self, graph):
        walks = []
        for _ in range(10):
            for node in graph.nodes():
                walk = [str(node)]
                while len(walk) < 10:
                    cur = int(walk[-1])
                    neighbors = list(graph.neighbors(cur))
                    if neighbors:
                        walk.append(str(np.random.choice(neighbors)))
                    else:
                        break
                walks.append(walk)
        model = Word2Vec(walks, vector_size=self.embed_size, window=5, min_count=0, sg=1)
        embeddings = {int(k): model.wv[k] for k in model.wv.key_to_index}
        return embeddings

class AnomalyClassifier(nn.Module):
    def __init__(self, input_size):
        super(AnomalyClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

def parse_interpret_label(txt_file):
    gt = defaultdict(list)
    with open(os.path.join("data/interpretation_label", txt_file), 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            time_part, sensors = line.strip().split(':')
            start, end = map(int, time_part.split('-'))
            sensor_list = list(map(int, sensors.split(','))) if sensors else []
            for t in range(start, end + 1):
                gt[t] = sensor_list
    return gt

if __name__ == "__main__":
    WINDOW_SIZE = 20
    CORR_THRESH = 0.7
    DTW_THRESH = 1.2

    txt_files = [f for f in os.listdir("data/interpretation_label") if f.startswith("omi-") and f.endswith(".txt")]
    txt_file = txt_files[0] if txt_files else None

    if not txt_file:
        logger.error("未找到标注文件")
    else:
        prefix = txt_file.split('.')[0]
        data_path = os.path.join("data/processed", f"{prefix}_test.pkl")
        label_path = os.path.join("data/processed", f"{prefix}_test_label.pkl")

        data = pd.read_pickle(data_path)
        labels = pd.read_pickle(label_path).reshape(-1)
        logger.info(f"数据维度: {data.shape}, 标签维度: {labels.shape}")

        builder = TimeWindowGraphBuilder(WINDOW_SIZE, CORR_THRESH, DTW_THRESH)
        graph = builder.build_graph(data)

        embeddings = DeepWalkEmbedding().learn_embeddings(graph)
        logger.info(f"获得嵌入数量: {len(embeddings)}")

        n = len(labels) - WINDOW_SIZE + 1
        cutoff = int(n * 0.7)
        X_train = []
        y_train = []

        for i in range(cutoff):
            if i in embeddings:
                X_train.append(embeddings[i])
                y_train.append(labels[i + WINDOW_SIZE - 1])

        model = AnomalyClassifier(input_size=32)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        for epoch in range(50):
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info("训练完成")

        ground_truth = parse_interpret_label(txt_file)
        X_test, test_indices = [], []
        for i in range(cutoff, n):
            if i in embeddings:
                X_test.append(embeddings[i])
                test_indices.append(i + WINDOW_SIZE - 1)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_test).detach().numpy().flatten()

        TP, FP = 0, 0
        for i, pred in zip(test_indices, preds):
            if pred > 0.5:
                if i in ground_truth:
                    TP += len(ground_truth[i])
                else:
                    FP += 1

        logger.info(f"最终检测结果：正确检测异常传感器总数（TP）: {TP}, 误报时刻数（FP）: {FP}")
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from fastdtw import fastdtw
from scipy.stats import pearsonr
from gensim.models import Word2Vec
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# === 1. 加载数据 ===
def load_data(x):
    base_path = 'data/processed'
    train = pickle.load(open(f"{base_path}/omi-{x}_train.pkl", "rb"))
    test = pickle.load(open(f"{base_path}/omi-{x}_test.pkl", "rb"))
    test_label = pickle.load(open(f"{base_path}/omi-{x}_test_label.pkl", "rb"))
    return train, test, test_label

def load_interpretation_label(x):
    path = f"data/interpretation_label/omi-{x}.txt"
    result = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip(): continue
            time_range, sensors = line.strip().split(":")
            start, end = map(int, time_range.split("-"))
            result[(start, end)] = list(map(int, sensors.split(",")))
    return result

# === 2. 构建滑动窗口图结构 ===
def sliding_windows(data, window_size=30, step=5):
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i+window_size])
    return np.array(windows)

def similarity(window1, window2):
    w1 = window1.flatten()
    w2 = window2.flatten()
    pearson = pearsonr(w1, w2)[0]
    dtw_dist, _ = fastdtw(w1, w2)
    return pearson - 0.01 * dtw_dist  # 权重可调

def build_graph(windows):
    G = nx.Graph()
    for i in range(len(windows)):
        G.add_node(i)
    for i in tqdm(range(len(windows))):
        for j in range(i+1, len(windows)):
            sim = similarity(windows[i], windows[j])
            if sim > 0.7:  # 设置阈值
                G.add_edge(i, j, weight=sim)
    return G

# === 3. DeepWalk + Word2Vec ===
def deepwalk_random_walk(G, path_length=10, num_walks=5):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)]
            while len(walk) < path_length:
                neighbors = list(G.neighbors(int(walk[-1])))
                if neighbors:
                    next_node = str(np.random.choice(neighbors))
                    walk.append(next_node)
                else:
                    break
            walks.append(walk)
    return walks

def train_word2vec(walks, vector_size=64, window=5, min_count=0):
    model = Word2Vec(walks, vector_size=vector_size, window=window, min_count=min_count, sg=1)
    return model

# === 4. 构建检测模型 ===
class SensorAnomalyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# === 5. 模型训练 ===
def train_model(model, X, y, epochs=10, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X, dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# === 6. 评估 ===
def evaluate(model, X, y_true, threshold=0.5):
    model.eval()
    inputs = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs).numpy()
    y_pred = (outputs > threshold).astype(int)
    return y_pred

# === 7. 对齐interpretation_label评估 ===
def match_interpretation(preds, interp, start_index, window_size):
    results = []

    for (start, end), true_sensors in interp.items():
        for i in range(start, end + 1):
            win_idx = (i - start_index) // window_size
            if 0 <= win_idx < len(preds):
                pred_sensors = np.where(preds[win_idx] == 1)[0].tolist()

                tp = len(set(pred_sensors) & set(true_sensors))  # 真正
                fp = len(set(pred_sensors) - set(true_sensors))  # 假正
                fn = len(set(true_sensors) - set(pred_sensors))  # 假负

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                results.append({
                    "start": start,
                    "end": end,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1
                })
    return results

def summarize_results(results):
    total_tp = sum(r["TP"] for r in results)
    total_fp = sum(r["FP"] for r in results)
    total_fn = sum(r["FN"] for r in results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n====== 📊 总体评估结果 ======")
    print(f"总正确检测传感器数 (TP): {total_tp}")
    print(f"总错误检测传感器数 (FP): {total_fp}")
    print(f"总漏检传感器数   (FN): {total_fn}")
    print(f"总体准确率 Precision: {precision:.4f}")
    print(f"总体召回率 Recall:    {recall:.4f}")
    print(f"总体 F1 分数:          {f1:.4f}")


# === 主流程函数 ===
def main(x):
    print(f"处理 omi-{x}...")
    train, test, test_label = load_data(x)
    interp_label = load_interpretation_label(x)
    window_size = 30
    step = 5

    train_windows = sliding_windows(train, window_size, step)
    G = build_graph(train_windows)
    walks = deepwalk_random_walk(G)
    w2v_model = train_word2vec(walks)

    # 构造训练集向量 + 模拟标签（这里简单假设）
    embed_dim = w2v_model.vector_size
    X_train = []
    for i in range(len(train_windows)):
        if str(i) in w2v_model.wv:
            X_train.append(w2v_model.wv[str(i)])
    X_train = np.array(X_train)
    y_train = np.random.randint(0, 2, size=(X_train.shape[0], 19))  # 模拟标签，真实应标注

    # 训练检测模型
    model = SensorAnomalyModel(input_dim=embed_dim, output_dim=19)
    train_model(model, X_train, y_train)

    # 处理测试集
    test_windows = sliding_windows(test, window_size, step)
    X_test = []
    for i in range(len(test_windows)):
        if str(i) in w2v_model.wv:
            X_test.append(w2v_model.wv[str(i)])
        else:
            X_test.append(np.zeros(embed_dim))
    X_test = np.array(X_test)

    preds = evaluate(model, X_test, y_train[:len(X_test)])  # 示例用 train 标签，可替换
    results = match_interpretation(preds, interp_label, start_index=0, window_size=step)

    # 打印每段详细结果
    for r in results:
        print(f"[{r['start']}-{r['end']}] TP: {r['TP']}, FP: {r['FP']}, FN: {r['FN']}, "
              f"Precision: {r['Precision']:.2f}, Recall: {r['Recall']:.2f}, F1: {r['F1']:.2f}")

    # 打印总体统计
    summarize_results(results)


# === 批量运行 ===
for x in range(1, 2):  # 假设你有 omi-1 到 omi-5
    try:
        main(x)
    except Exception as e:
        print(f"omi-{x} 处理失败: {e}")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

# ------------------- 数据加载 -------------------
def load_test_data(x):
    test = pickle.load(open(f"data/processed/omi-{x}_test.pkl", "rb"))
    return test

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

# ------------------- 滑动窗口 -------------------
def sliding_windows(data, window_size=30, step=5):
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    return np.array(windows)

# ------------------- 构建图 -------------------
def build_graph(windows, top_k=5):
    node_features = [w.flatten() for w in windows]
    n = len(node_features)
    edges = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i != j:
                sim = 1 - cosine(node_features[i], node_features[j])
                sims.append((j, sim))
        top = sorted(sims, key=lambda x: -x[1])[:top_k]
        for j, _ in top:
            edges.append((i, j))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

# ------------------- GCN 模型 -------------------
class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.conv1(x, edge_index))
        z = self.conv2(h, edge_index)
        return z

# ------------------- GCN 嵌入 → 异常传感器识别 -------------------
def detect_gcn_anomalous_sensors(embeddings, normal_len, sensor_dim=19):
    dim = min(sensor_dim, embeddings.shape[1])
    pca = PCA(n_components=dim)
    reduced = pca.fit_transform(embeddings.detach().cpu().numpy())
    normal_mean = np.mean(reduced[:normal_len], axis=0)
    abnormal_part = reduced[normal_len:]
    deltas = np.abs(abnormal_part - normal_mean)
    avg_delta = np.mean(deltas, axis=0)
    threshold = np.percentile(avg_delta, 80)
    anomalous_sensors = np.where(avg_delta >= threshold)[0].tolist()
    return anomalous_sensors

# ------------------- 区段评估 -------------------
def evaluate(pred_sensors, true_sensors):
    tp = len(set(pred_sensors) & set(true_sensors))
    fp = len(set(pred_sensors) - set(true_sensors))
    fn = len(set(true_sensors) - set(pred_sensors))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return tp, fp, fn, precision, recall, f1

# ------------------- 主流程 -------------------
def main(x):
    test = load_test_data(x)
    interp = load_interpretation_label(x)
    window_size = 30
    step = 5
    total_tp, total_fp, total_fn = 0, 0, 0
    print(f"\n=========== 🔍 开始处理 omi-{x} ===========")

    for (start, end), true_sensors in interp.items():
        if start < 100: continue
        segment = test[start - 100:end]
        windows = sliding_windows(segment, window_size=window_size, step=step)
        if len(windows) < 5: continue  # too few windows
        graph_data = build_graph(windows)
        model = SimpleGCN(graph_data.x.shape[1])
        model.eval()
        with torch.no_grad():
            embeddings = model(graph_data)
        normal_len = (100 - window_size) // step + 1
        pred_sensors = detect_gcn_anomalous_sensors(embeddings, normal_len)
        tp, fp, fn, p, r, f1 = evaluate(pred_sensors, true_sensors)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(f"[{start}-{end}] TP: {tp}  FP: {fp}  FN: {fn}  P: {p:.2f}  R: {r:.2f}  F1: {f1:.2f}")

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
    print(f"=========== ✅ 处理完成 omi-{x} ===========")

# 🚀 执行 GCN 检测
main(1)

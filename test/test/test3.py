import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from dtaidistance import dtw
import networkx as nx
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader

class SensorDataset(Dataset):
    def __init__(self, data, labels, window_size=20):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx):
        window_data = self.data[idx:idx+self.window_size]
        window_label = self.labels[idx+self.window_size-1]  # 使用窗口最后一个标签
        return window_data.permute(1, 0), window_label  # [n_sensors, window_size]

def load_and_split(txt_file_path, test_size=0.3):
    """加载数据并划分训练测试集"""
    data = pd.read_csv(txt_file_path, delimiter='\t', header=None).values
    label_path = txt_file_path.replace('.txt', '.label')
    labels = pd.read_csv(label_path, delimiter='\t', header=None).values
    
    # 按时间顺序划分
    n_train = int(len(data) * (1 - test_size))
    train_data = data[:n_train]
    train_labels = labels[:n_train]
    test_data = data[n_train:]
    test_labels = labels[n_train:]
    
    return train_data, train_labels, test_data, test_labels

class RelationLearner:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.normal_graph = None
        self.anomaly_graph = None
    
    def build_graphs(self, dataset):
        """从数据集构建正常和异常图"""
        normal_segments = self._extract_segments(dataset, target_label=0)
        anomaly_segments = self._extract_segments(dataset, target_label=1)
        
        self.normal_graph = self._build_single_graph(normal_segments)
        self.anomaly_graph = self._build_single_graph(anomaly_segments)
    
    def _extract_segments(self, dataset, target_label=0):
        """提取连续时段"""
        segments = []
        current_segment = []
        
        for i in range(len(dataset)):
            data, label = dataset[i]
            if label.mean() == target_label:  # 多数投票决定窗口标签
                current_segment.append(data.numpy())
            else:
                if len(current_segment) >= self.window_size:
                    segments.append(np.stack(current_segment))
                current_segment = []
        
        return segments
    
    def _build_single_graph(self, segments):
        """构建单个关联图"""
        if not segments:
            return nx.Graph()
        
        n_sensors = segments[0].shape[1]
        G = nx.Graph()
        G.add_nodes_from(range(n_sensors))
        
        corr_matrix = np.zeros((n_sensors, n_sensors))
        count_matrix = np.zeros((n_sensors, n_sensors))
        
        for seg in segments:
            for i in range(n_sensors):
                for j in range(i+1, n_sensors):
                    pearson = np.corrcoef(seg[:, i, :], seg[:, j, :])[0, 1]
                    dtw_dist = dtw.distance(seg[:, i, :].flatten(), seg[:, j, :].flatten())
                    
                    if abs(pearson) > 0.6 and dtw_dist < 2.0:
                        corr_matrix[i, j] += pearson
                        count_matrix[i, j] += 1
        
        for i in range(n_sensors):
            for j in range(i+1, n_sensors):
                if count_matrix[i, j] > 0:
                    avg_corr = corr_matrix[i, j] / count_matrix[i, j]
                    G.add_edge(i, j, weight=avg_corr)
        
        return G
    


class DeepWalkFeatureExtractor:
    def __init__(self, embed_size=32):
        self.embed_size = embed_size
    
    def extract_features(self, graph):
        """从图结构提取DeepWalk特征"""
        if len(graph.edges()) == 0:
            return torch.zeros(len(graph.nodes()), self.embed_size)
        
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
        
        model = Word2Vec(walks, vector_size=self.embed_size, 
                        window=5, min_count=0, sg=1, workers=4)
        
        features = []
        for i in range(len(graph.nodes())):
            features.append(model.wv[str(i)])
        
        return torch.FloatTensor(np.array(features))
    
class AnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def train_supervised_model(train_features, train_labels, val_features, val_labels, n_epochs=50):
    """训练监督模型"""
    model = AnomalyDetector(input_dim=train_features.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for feats, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(feats).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features).squeeze()
            val_loss = criterion(val_outputs, val_labels)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def process_file(txt_file_path):
    # 1. 加载并划分数据
    train_data, train_labels, test_data, test_labels = load_and_split(txt_file_path)
    
    # 2. 创建数据集
    train_dataset = SensorDataset(train_data, train_labels)
    test_dataset = SensorDataset(test_data, test_labels)
    
    # 3. 学习关联关系
    relation_learner = RelationLearner()
    relation_learner.build_graphs(train_dataset)
    
    # 4. 提取特征
    feature_extractor = DeepWalkFeatureExtractor()
    normal_feats = feature_extractor.extract_features(relation_learner.normal_graph)
    anomaly_feats = feature_extractor.extract_features(relation_learner.anomaly_graph)
    
    # 5. 准备训练数据
    train_features = torch.cat([normal_feats, anomaly_feats])
    train_labels = torch.cat([
        torch.zeros(len(normal_feats)), 
        torch.ones(len(anomaly_feats))
    ])
    
    # 6. 训练监督模型
    model = train_supervised_model(train_features, train_labels, 
                                 normal_feats[:len(normal_feats)//5],  # 用部分正常数据做验证
                                 torch.zeros(len(normal_feats)//5))
    
    # 7. 测试评估
    test_anomaly_scores = evaluate_on_test(model, test_dataset, feature_extractor)
    print_performance(test_anomaly_scores, test_labels[len(test_labels)-len(test_anomaly_scores):])

def evaluate_on_test(model, test_dataset, feature_extractor):
    """在测试集上评估"""
    model.eval()
    anomaly_scores = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            window_data, _ = test_dataset[i]
            
            # 为当前窗口构建临时图
            G = nx.Graph()
            n_sensors = window_data.shape[0]
            G.add_nodes_from(range(n_sensors))
            
            for i in range(n_sensors):
                for j in range(i+1, n_sensors):
                    pearson = np.corrcoef(window_data[i], window_data[j])[0,1]
                    if abs(pearson) > 0.6:
                        G.add_edge(i, j, weight=pearson)
            
            # 提取特征并预测
            feats = feature_extractor.extract_features(G)
            scores = model(feats).squeeze()
            anomaly_scores.append(scores.numpy())
    
    return np.stack(anomaly_scores)

def print_performance(scores, true_labels, threshold=0.5):
    """打印性能指标"""
    preds = (scores > threshold).astype(int)
    true_labels = true_labels[:len(preds)]  # 对齐长度
    
    tp = np.sum((preds == 1) & (true_labels == 1))
    fp = np.sum((preds == 1) & (true_labels == 0))
    fn = np.sum((preds == 0) & (true_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    import glob
    
    # 处理所有TXT文件
    txt_files = glob.glob("data/*.txt")
    for txt_file in txt_files:
        print(f"\nProcessing {os.path.basename(txt_file)}...")
        process_file(txt_file)

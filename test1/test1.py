import os
import numpy as np
from scipy.stats import pearsonr
from dtaidistance import dtw
import networkx as nx
from gensim.models import Word2Vec
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):   # x为输入数据,形状为(batch_size, input_dim)
        return self.classifier(x)   # 经过网络计算后的概率值 (batch_size, 1)，表示每个样本是异常的概率

def deepwalk_feature_extraction(data, window_size=5):
    """使用DeepWalk获取时间窗口的嵌入特征"""
    G = nx.Graph()
    n_windows = len(data) - window_size + 1
    
    # 构建图结构
    for i in range(n_windows):
        window = data[i:i+window_size]
        G.add_node(i, values=window)
    
    # 添加边
    for i in range(n_windows):
        for j in range(i+1, min(i+20, n_windows)):
            corr = pearsonr(G.nodes[i]['values'].flatten(), 
                           G.nodes[j]['values'].flatten())[0]
            dtw_dist = dtw.distance(G.nodes[i]['values'].mean(axis=1),
                                   G.nodes[j]['values'].mean(axis=1))
            if abs(corr) > 0.65 and dtw_dist < 1.6:
                G.add_edge(i, j, weight=abs(corr))
    
    # DeepWalk嵌入
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
    
    return embeddings

def prepare_dataset(data, labels, window_size=5):
    """准备训练数据集"""
    features = []
    targets = []
    
    # 获取每个时间段的DeepWalk特征
    embeddings = deepwalk_feature_extraction(data, window_size)
    
    # 为每个传感器创建样本
    for sensor in range(data.shape[1]):
        # 获取该传感器在所有窗口中的平均值作为特征
        sensor_features = []
        for i in range(len(embeddings)):
            window_data = data[i:i+window_size, sensor]
            stats = [
                np.mean(window_data),
                np.std(window_data),
                np.min(window_data),
                np.max(window_data),
                np.median(window_data)
            ]
            combined = np.concatenate([embeddings[i], stats])
            sensor_features.append(combined)
        
        # 对应窗口的标签(如果窗口中任何时间点异常则为1)
        sensor_labels = [
            int(any(labels[i:i+window_size] == 1)) 
            for i in range(len(embeddings))
        ]
        
        features.extend(sensor_features)
        targets.extend(sensor_labels)
    
    return np.array(features), np.array(targets)

def train_and_evaluate(model, train_loader, test_loader, epochs=50):
    """训练和评估模型"""
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练阶段
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
    
    # 评估阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.float())
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets.unsqueeze(1)).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return model

def process_file_with_model(txt_file_path, test_data, test_labels, model, window_size=5):
    """使用训练好的模型处理文件"""
    print("="*60)
    print("Processing file:", txt_file_path)
    
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    expected_sum = 0
    true_positive_sum = 0
    detected_sum = 0
    
    for line in lines:
        line = line.strip()
        if not line or ':' not in line or '-' not in line:
            continue
            
        time_part, sensor_part = line.split(":", 1)
        start_idx, end_idx = map(int, time_part.split('-'))
        
        expected_sensors = []
        if sensor_part.strip():
            expected_sensors = [int(x) for x in sensor_part.split(',') if x.strip() != '']
        
        # 提取该时段数据
        seg_data = test_data.iloc[start_idx:end_idx+1].to_numpy()
        seg_labels = test_labels.iloc[start_idx:end_idx+1]["label"].to_numpy()
        
        # 获取DeepWalk特征
        embeddings = deepwalk_feature_extraction(seg_data, window_size)
        if len(embeddings) == 0:
            continue
        
        # 对每个传感器进行预测
        detected_sensors = []
        model.eval()
        with torch.no_grad():
            for sensor in range(seg_data.shape[1]):
                # 创建输入特征
                inputs = []
                for i in range(len(embeddings)):
                    window_data = seg_data[i:i+window_size, sensor]
                    stats = [
                        np.mean(window_data),
                        np.std(window_data),
                        np.min(window_data),
                        np.max(window_data),
                        np.median(window_data)
                    ]
                    combined = np.concatenate([embeddings[i], stats])
                    inputs.append(combined)
                
                inputs = torch.FloatTensor(np.array(inputs))
                outputs = model(inputs)
                
                # 如果任何窗口预测为异常，则标记该传感器为异常
                if (outputs > 0.5).any():
                    detected_sensors.append(sensor)
        
        # 统计结果
        detected_set = set(detected_sensors)
        expected_set = set(expected_sensors)
        true_positives = detected_set & expected_set
        
        expected_sum += len(expected_set)
        true_positive_sum += len(true_positives)
        detected_sum += len(detected_set)
        
        print(f"时段 {start_idx}-{end_idx}:")
        print(f"  期望异常传感器: {sorted(expected_set)}")
        print(f"  检测到异常传感器: {sorted(detected_set)}")
        print(f"  命中: {sorted(true_positives)}")
        print("-"*50)
    
    # 计算指标
    recall = true_positive_sum / expected_sum if expected_sum > 0 else 0
    precision = true_positive_sum / detected_sum if detected_sum > 0 else 0
    
    print("当前文件统计:")
    print(f"  总期望异常传感器数: {expected_sum}")
    print(f"  总正确检测数: {true_positive_sum}")
    print(f"  总检测到异常传感器数: {detected_sum}")
    print(f"  召回率: {recall*100:.1f}%")
    print(f"  精确率: {precision*100:.1f}%")
    print("="*60)
    
    return {
        "expected_sum": expected_sum,
        "true_positive_sum": true_positive_sum,
        "detected_sum": detected_sum
    }

if __name__ == "__main__":
    # 加载数据
    label_folder = r"data\interpretation_label"
    txt_files = [f for f in os.listdir(label_folder)
                if f.startswith("omi-") and f.endswith(".txt")]
    
    # 只使用第一个文件进行演示
    txt_filename = txt_files[0]
    prefix = txt_filename.split(".")[0]
    test_data_path = os.path.join("data/processed", f"{prefix}_test.pkl")
    test_labels_path = os.path.join("data/processed", f"{prefix}_test_label.pkl")
    
    test_data = pd.read_pickle(test_data_path)
    test_labels = pd.read_pickle(test_labels_path)
    
    # 转换为DataFrame格式
    if isinstance(test_data, np.ndarray):
        test_data = pd.DataFrame(test_data)
    if isinstance(test_labels, np.ndarray):
        test_labels = pd.DataFrame(test_labels, columns=["label"])
    elif isinstance(test_labels, pd.Series):
        test_labels = test_labels.to_frame(name="label")
    
    # 获取数据总长度
    total_samples = len(test_data)

    # 计算70%的位置
    train_size = int(total_samples * 0.7)

    # 使用前70%作为训练
    X, y = prepare_dataset(test_data.iloc[:train_size].values,
                        test_labels.iloc[:train_size]["label"].values)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 转换为PyTorch Dataset
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    model = AnomalyDetector(input_dim=X.shape[1])
    
    # 训练和评估模型
    trained_model = train_and_evaluate(model, train_loader, test_loader)
    
    # 在完整文件上测试
    file_path = os.path.join(label_folder, txt_filename)
    stats = process_file_with_model(file_path, test_data, test_labels, trained_model)
    
    # 输出最终结果
    recall = stats["true_positive_sum"] / stats["expected_sum"] if stats["expected_sum"] > 0 else 0
    precision = stats["true_positive_sum"] / stats["detected_sum"] if stats["detected_sum"] > 0 else 0
    
    print("最终统计:")
    print(f"  总期望异常传感器数: {stats['expected_sum']}")
    print(f"  总正确检测数: {stats['true_positive_sum']}")
    print(f"  总检测到异常传感器数: {stats['detected_sum']}")
    print(f"  召回率: {recall*100:.1f}%")
    print(f"  精确率: {precision*100:.1f}%")
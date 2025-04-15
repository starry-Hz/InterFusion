import os
import numpy as np
from scipy.stats import pearsonr
from dtaidistance import dtw
import networkx as nx
from gensim.models import Word2Vec
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        
        # 主分支路径
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU()
        )
        
        # 残差连接路径
        self.shortcut = nn.Sequential()
        if input_dim != hidden_dim//2:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, hidden_dim//2),
                nn.BatchNorm1d(hidden_dim//2)
            )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
        # 初始化最佳阈值
        self.best_threshold = 0.5
        
    def forward(self, x):
        # 主分支处理
        main_out = self.main(x)
        
        # 残差连接
        shortcut_out = self.shortcut(x)
        out = main_out + shortcut_out
        
        # 输出概率
        return self.output(out)
    
    def optimize_threshold(self, val_loader, device='cpu'):
        """根据验证集F1分数优化阈值"""
        self.eval()
        y_true = []
        y_prob = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = self(inputs)
                
                y_true.extend(targets.cpu().numpy())
                y_prob.extend(outputs.cpu().numpy())
        
        y_true = np.array(y_true)
        y_prob = np.array(y_prob).flatten()
        
        # 搜索最佳阈值
        thresholds = np.linspace(0.1, 0.9, 50)
        best_f1 = -1
        best_thresh = 0.5
        
        for thresh in thresholds:
            y_pred = (y_prob > thresh).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        self.best_threshold = best_thresh
        print(f"Optimized threshold: {best_thresh:.4f} (F1={best_f1:.4f})")
        return best_thresh
    
    # def predict(self, x, threshold=None):
    #     """使用优化后的阈值进行预测"""
    #     if threshold is None:
    #         threshold = self.best_threshold
        
    #     with torch.no_grad():
    #         proba = self(x)
    #         return (proba > threshold).float()
    def predict(self, x, threshold=None):
        """使用优化后的阈值进行预测"""
        if threshold is None:
            threshold = self.best_threshold
        
        with torch.no_grad():
            proba = self(x)
            return (proba > threshold).long()  # 改为long类型

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
    
    # 检查图是否为空
    if len(G.nodes()) == 0:
        return np.array([])
    
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
    
    # 检查walks是否有效
    if not walks or all(len(walk) == 0 for walk in walks):
        return np.array([])
    
    # 训练Word2Vec模型（添加词汇表构建）
    model = Word2Vec(
        sentences=walks,
        vector_size=32,
        window=5,
        min_count=1,  # 改为1，确保至少出现1次的节点被包含
        sg=1,
        workers=4
    )
    
    # 获取嵌入向量
    embeddings = []
    for i in range(n_windows):
        if str(i) in model.wv:
            embeddings.append(model.wv[str(i)])
        else:
            # 对于未出现在词汇表中的节点，使用零向量
            embeddings.append(np.zeros(32))
    
    return np.array(embeddings)

def prepare_dataset(data, labels, window_size=5):
    """准备训练数据集"""
    features = []
    targets = []
    
    # 获取DeepWalk特征
    embeddings = deepwalk_feature_extraction(data, window_size)
    if len(embeddings) == 0:
        return np.array([]), np.array([])
    
    # 为每个传感器创建样本
    for sensor in range(data.shape[1]):
        # 获取该传感器在所有窗口中的统计特征
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

def train_model(model, train_loader, val_loader, epochs=50, device='cpu'):
    """训练模型并优化阈值"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 定期验证并优化阈值
        if (epoch+1) % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets.unsqueeze(1)).item()
            
            # 优化阈值
            best_thresh = model.optimize_threshold(val_loader, device)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} | "
                  f"Best Threshold: {best_thresh:.4f}")
    
    return model

def evaluate_model(model, test_loader, device='cpu'):
    """评估模型性能"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model.predict(inputs, model.best_threshold)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    
    # 转换为整数类型
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32).flatten()
    
    # 计算指标
    accuracy = (y_true == y_pred).mean()
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    actual_positives = np.sum(y_true == 1)
    
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (actual_positives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Using Threshold: {model.best_threshold:.4f}")

def process_file_with_model(txt_file_path, test_data, test_labels, model, window_size=5, device='cpu'):
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
        
        try:
            # 获取DeepWalk特征
            embeddings = deepwalk_feature_extraction(seg_data, window_size)
            if len(embeddings) == 0:
                print(f"时段 {start_idx}-{end_idx}: 无法提取特征 (数据不足或无法构建图)")
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
                    
                    if not inputs:  # 如果没有任何输入特征
                        continue
                        
                    inputs = torch.FloatTensor(np.array(inputs)).to(device)
                    outputs = model.predict(inputs)
                    
                    # 如果任何窗口预测为异常，则标记该传感器为异常
                    if outputs.any():
                        detected_sensors.append(sensor)
        
        except Exception as e:
            print(f"时段 {start_idx}-{end_idx} 处理出错: {str(e)}")
            continue
        
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
    print(f"  使用阈值: {model.best_threshold:.4f}")
    print("="*60)
    
    return {
        "expected_sum": expected_sum,
        "true_positive_sum": true_positive_sum,
        "detected_sum": detected_sum
    }

if __name__ == "__main__":
    # 配置参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    window_size = 5
    batch_size = 64
    epochs = 50
    
    # 加载数据
    label_folder = r"data\interpretation_label"
    txt_files = [f for f in os.listdir(label_folder)
                if f.startswith("omi-") and f.endswith(".txt")]


    # 使用第一个文件进行演示
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

    print(f"数据集大小: {test_data.shape}") # (4320, 19)
    print(f"标签集大小: {test_labels.shape}")   # (4320, 1)
    
    # 准备数据集 (使用前70%训练，后30%测试)
    total_samples = len(test_data)
    train_size = int(total_samples * 0.7)
    
    train_data = test_data.iloc[:train_size]
    train_labels = test_labels.iloc[:train_size]
    
    test_data = test_data.iloc[train_size:]
    test_labels = test_labels.iloc[train_size:]
    
    # 准备训练和验证集
    X, y = prepare_dataset(train_data.values, train_labels["label"].values, window_size)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 转换为PyTorch Dataset
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    input_dim = X_train.shape[1]
    model = AnomalyDetector(input_dim)
    
    # 训练模型
    print("开始训练模型...")
    model = train_model(model, train_loader, val_loader, epochs, device)
    
    # 评估模型
    print("\n评估验证集性能...")
    evaluate_model(model, val_loader, device)
    
    # 在完整文件上测试
    file_path = os.path.join(label_folder, txt_filename)
    stats = process_file_with_model(file_path, test_data, test_labels, model, window_size, device)
    
    # 输出最终结果
    recall = stats["true_positive_sum"] / stats["expected_sum"] if stats["expected_sum"] > 0 else 0
    precision = stats["true_positive_sum"] / stats["detected_sum"] if stats["detected_sum"] > 0 else 0
    
    print("\n最终统计:")
    print(f"  总期望异常传感器数: {stats['expected_sum']}")
    print(f"  总正确检测数: {stats['true_positive_sum']}")
    print(f"  总检测到异常传感器数: {stats['detected_sum']}")
    print(f"  召回率: {recall*100:.1f}%")
    print(f"  精确率: {precision*100:.1f}%")
    print(f"  使用阈值: {model.best_threshold:.4f}")
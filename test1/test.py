import numpy as np
import networkx as nx
from karateclub import DeepWalk
from sklearn.ensemble import IsolationForest
import pandas as pd
import time
import datetime

print("start", datetime.datetime.now())
start = time.time()

# 1. 数据加载
data = pd.read_pickle('data/processed/omi-1_test.pkl')
print("Data shape:", data.shape)

# 确保数据保持为DataFrame格式
if isinstance(data, np.ndarray):
    data = pd.DataFrame(data)
    
T, N = data.shape  # 时间步数, 传感器数

# 2. 构建传感器关系图
G = nx.Graph()

# 添加传感器节点（使用列名或索引作为节点ID）
for n in range(N):
    # 获取传感器名称（如果存在），否则使用数字索引
    node_name = data.columns[n] if hasattr(data, 'columns') else n
    G.add_node(node_name, features=data.iloc[:, n].values)

# 基于相关性添加边（使用滑动窗口平均相关性）
window_size = min(24, T)  # 24小时窗口或总时间步数
corr_threshold = 0.7  # 相关性阈值

for i in range(N):
    for j in range(i+1, N):
        # 获取节点名称
        node_i = data.columns[i] if hasattr(data, 'columns') else i
        node_j = data.columns[j] if hasattr(data, 'columns') else j
        
        # 计算滑动窗口相关性
        rolling_corr = data.iloc[:, i].rolling(window_size).corr(data.iloc[:, j])
        avg_corr = rolling_corr.mean()
        
        if avg_corr > corr_threshold:
            G.add_edge(node_i, node_j, weight=avg_corr)

# 3. DeepWalk嵌入
print("Training DeepWalk...")
model = DeepWalk(walk_number=10, walk_length=80, dimensions=64)
model.fit(G)
embeddings = model.get_embedding()

# 4. 异常检测（使用孤立森林）
print("Detecting anomalies...")
clf = IsolationForest(contamination=0.05, random_state=42)
anomaly_scores = clf.fit_predict(embeddings)

# 5. 识别异常传感器
node_list = list(G.nodes())
abnormal_sensors = [node_list[i] for i in np.where(anomaly_scores == -1)[0]]  # -1表示异常
print("Abnormal sensors detected:", abnormal_sensors)

# 6. 时间维度分析（可选）
# 对每个异常传感器，分析其异常时间段
for sensor in abnormal_sensors:
    sensor_idx = data.columns.get_loc(sensor) if hasattr(data, 'columns') else sensor
    sensor_data = data.iloc[:, sensor_idx]
    
    # 使用滑动Z-score检测异常时间段
    rolling_mean = sensor_data.rolling(window=24).mean()
    rolling_std = sensor_data.rolling(window=24).std()
    z_scores = (sensor_data - rolling_mean) / rolling_std
    abnormal_times = np.where(np.abs(z_scores) > 3)[0]  # Z-score > 3
    
    if len(abnormal_times) > 0:
        print(f"Sensor {sensor} abnormal at time steps: {abnormal_times}")

end = time.time()
print("end", datetime.datetime.now())
print('Total time:', end - start)
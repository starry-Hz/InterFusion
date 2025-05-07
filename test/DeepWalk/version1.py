'''
data\interpretation_label\omi
实验结果最终总体统计:
  总期望异常传感器数: 658
  总正确检测数: 550
  总检测到异常传感器数: 1192
  全局召回率: 83.6%
  全局精确率: 46.1%

===================================

data\interpretation_label\machine
最终总体统计:
  总期望异常传感器数: 1167
  总正确检测数: 765
  总检测到异常传感器数: 2764
  全局召回率: 65.6%
  全局精确率: 27.7%
'''

import os
import numpy as np
from scipy.stats import pearsonr
from dtaidistance import dtw
import networkx as nx
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import pandas as pd

def robust_zscore(x, median_val, mad):
    # 鲁棒 Z 分数：(x - median) / (MAD + 1e-8), 避免除零错误
    return np.abs(x - median_val) / (mad + 1e-8)

def strict_image_method(data, labels, window_size=5, score_threshold=0.5):
    """
    改进后的异常检测方法：
    1. 构建图结构：节点为连续时间窗口(窗口内数据存储在节点属性 'values'),
       边的条件：窗口展平后的 Pearson 相关系数绝对值 > 0.65,
       且各窗口沿时间方向均值序列的 DTW 距离 < 1.6。
    2. 利用随机游走和 Word2Vec(DeepWalk 思想)获得每个窗口的低维嵌入表示,
       随机游走次数设为20次,游走步长延长到20。
    3. 用 KMeans 聚类(聚类数设为2,数量较少的簇视为异常窗口),
       将异常窗口映射回原始时间点,再对真实异常点(labels==1)利用鲁棒统计(中位数和 MAD)计算局部异常得分。
       最后,对每个传感器计算参与异常点的平均得分,当平均得分超过 score_threshold 时,认为该传感器异常。
    """
    # 1. 构建图结构
    G = nx.Graph()
    
    n_windows = len(data) - window_size + 1 # 窗口数量

    if n_windows < 2:
        return np.array([])
    for i in range(n_windows):
        window = data[i:i+window_size]
        G.add_node(i, values=window)
    
    # 添加边：比较窗口 i 与 i+1 到 i+20 内的窗口
    for i in range(n_windows):
        for j in range(i+1, min(i+20, n_windows)):
            # ​Pearson Correlation（相关性）衡量两个窗口的 ​整体数据模式是否相似​
            corr = pearsonr(G.nodes[i]['values'].flatten(), 
                            G.nodes[j]['values'].flatten())[0]
            # DTW Distance（动态时间规整）衡量两个窗口的 ​时间动态变化是否相似​（即使时间上有微小偏移）
            # DTW 计算的是两个时间序列在弹性对齐下的最小累积距离，用于衡量时间模式的相似性。
            # 在异常检测中，DTW 帮助发现时间窗口之间的局部相似性，即使它们的长度或相位不同。
            dtw_dist = dtw.distance(G.nodes[i]['values'].mean(axis=1),
                                    G.nodes[j]['values'].mean(axis=1))
            if abs(corr) > 0.65 and dtw_dist < 1.6:
                G.add_edge(i, j, weight=abs(corr))
    # 图 G（节点代表时间窗口，边代表窗口间的相似性）
    
    # 2. DeepWalk：随机游走获得节点序列,再用 Word2Vec 得到嵌入
    def random_walk(G, walk_length=20, num_walks=20):
        walks = []
        for _ in range(num_walks):  # 每个节点进行 `num_walks` 次随机游走
            for node in G.nodes():   # 遍历所有节点
                walk = [str(node)]   # 初始化游走序列（起点）
                while len(walk) < walk_length:  # 直到游走长度达到 `walk_length`
                    current = int(walk[-1])     # 当前节点
                    neighbors = list(G.neighbors(current))  # 获取邻居
                    if neighbors:  # 如果有邻居，随机选择一个继续游走
                        walk.append(str(np.random.choice(neighbors)))
                    else:  # 如果没有邻居（孤立节点），终止游走
                        break
                walks.append(walk)  # 保存本次游走序列
        return walks
    
    # DeepWalk 的核心思想是 ​通过随机游走生成节点序列，然后用 Word2Vec 训练嵌入向量
    walks = random_walk(G)
    model = Word2Vec(walks, vector_size=32, window=5, min_count=0, sg=1)
    embeddings = np.array([model.wv[str(i)] for i in G.nodes()])
    
    # 3. 异常检测：利用 KMeans 聚类,选出数量较少的簇作为异常窗口
    if embeddings.shape[0] < 2:
        return np.array([])
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(embeddings)   # 每个窗口的聚类标签（0 或 1）
    anomalies = np.where(clusters == np.argmin(np.bincount(clusters)))[0]   # 异常窗口
    
    # 将异常窗口映射回原始时间点
    anomaly_windows = [i for i in anomalies if i in G.nodes()]
    anomaly_points = [] # 异常时间点
    for win in anomaly_windows:
        anomaly_points.extend(range(win, win+window_size))
    
    # 针对每个异常点(仅统计真实异常点 labels==1),计算每个传感器的局部异常得分
    sensor_scores = np.zeros(data.shape[1])
    sensor_counts = np.zeros(data.shape[1])
    for point in set(anomaly_points):
        if point < len(labels) and labels[point] == 1:
            for sensor in range(data.shape[1]):
                # 提取当前传感器在异常点附近的数据窗口（前100个时间步到当前点）
                local_data = data[max(0, point-100):point+1, sensor]

                # 计算中位数和MAD
                median_val = np.median(local_data)  # 中位数
                mad = np.median(np.abs(local_data - median_val))    # MAD是绝对偏差的中位数，数据与中位数偏差的绝对值的中位数
                
                # 计算鲁棒Z分数并累加到对应传感器
                score = robust_zscore(data[point, sensor], median_val, mad)
                sensor_scores[sensor] += score
                sensor_counts[sensor] += 1
    # print('sensor_scores:', sensor_scores)
    # print('sensor_counts:', sensor_counts)
    # 计算每个传感器的平均得分
    avg_scores = np.zeros(data.shape[1])
    for sensor in range(data.shape[1]):
        if sensor_counts[sensor] > 0:
            avg_scores[sensor] = sensor_scores[sensor] / sensor_counts[sensor]
    
    # 返回平均得分大于 score_threshold 的传感器索引 
    return np.where(avg_scores > score_threshold)[0]

def process_omi_file(txt_file_path, test_data, test_labels):
    """
    处理单个 omi- 文件：
      1. 解析每行记录(格式：start-end:expected_sensor_list)；
      2. 对每个时段从 test_data 和 test_labels 中提取数据,调用 strict_image_method 进行检测；
      3. 输出每个时段的检测结果及对比,同时返回该文件所有时段的统计数据。
    返回一个字典,包含：
      - expected_sum: 期望异常传感器总数
      - true_positive_sum: 命中总数
      - detected_sum: 检测到的异常传感器总数
    """
    print("="*60)
    print("Processing file:", txt_file_path)
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print("lines:", lines)

    expected_sum = 0
    true_positive_sum = 0
    detected_sum = 0
     
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ':' not in line:
            print("格式错误,缺少冒号：", line)
            continue
        time_part, sensor_part = line.split(":", 1)
        if '-' not in time_part:
            print("格式错误,缺少 '-'：", line)
            continue
        try:
            start_str, end_str = time_part.split('-')
            start_idx = int(start_str)
            end_idx = int(end_str)
        except ValueError:
            print("时段解析错误：", line)
            continue
        if sensor_part.strip():
            try:
                expected_sensors = [int(x) for x in sensor_part.split(',') if x.strip() != '']
            except ValueError:
                print("传感器列表解析错误：", line)
                expected_sensors = []
        else:
            expected_sensors = []
        
        # 提取该时段数据(转换为 numpy 数组)
        seg_data = test_data.iloc[start_idx:end_idx+1].to_numpy()
        seg_labels = test_labels.iloc[start_idx:end_idx+1]["label"].to_numpy()
        
        detected_sensors = strict_image_method(seg_data, seg_labels)
        
        detected_set = set(detected_sensors)
        expected_set = set(expected_sensors)
        true_positives = detected_set & expected_set
        false_negatives = expected_set - detected_set
        false_positives = detected_set - expected_set
        
        expected_count = len(expected_set)
        true_positive_count = len(true_positives)
        detected_count = len(detected_set)
        
        expected_sum += expected_count
        true_positive_sum += true_positive_count
        detected_sum += detected_count
        
        miss_ratio = len(false_negatives) / expected_count if expected_count > 0 else 0
        false_ratio = len(false_positives) / detected_count if detected_count > 0 else 0
        
        print(f"时段 {start_idx}-{end_idx}:")
        print(f"  期望异常传感器: {sorted(expected_set)}")
        print(f"  检测到异常传感器: {sorted(detected_set)}")
        print(f"  命中: {sorted(true_positives)}")
        print(f"  漏检: {sorted(false_negatives)}")
        print(f"  误检: {sorted(false_positives)}")
        print(f"  正确检测数: {true_positive_count}")
        print(f"  漏检比例: {miss_ratio:.2f}")
        print(f"  误检比例: {false_ratio:.2f}")
        print("-"*50)
    
    # 计算当前文件的召回率和精确率
    recall = (true_positive_sum / expected_sum) if expected_sum > 0 else 0
    precision = (true_positive_sum / detected_sum) if detected_sum > 0 else 0
    
    print("当前文件统计:")
    print(f"  总期望异常传感器数: {expected_sum}")
    print(f"  总正确检测数: {true_positive_sum}")
    print(f"  总检测到异常传感器数: {detected_sum}")
    print(f"  召回率: {recall*100:.1f}%")
    print(f"  精确率: {precision*100:.1f}%")
    print("="*60)
    
    return {"expected_sum": expected_sum,
            "true_positive_sum": true_positive_sum,
            "detected_sum": detected_sum}

if __name__ == "__main__":
    # 指定包含 omi- 开头 txt 文件的文件夹(仅处理 omi- 文件)
    label_folder = './data/interpretation_label'
    txt_files = [f for f in os.listdir(label_folder)
                 if f.startswith("omi-") and f.endswith(".txt")]
    # print("处理文件列表：", txt_files)
    txt_files = ['els-1.txt']
    global_expected = 0
    global_true_positive = 0
    global_detected = 0
    
    for txt_filename in sorted(txt_files):
        prefix = txt_filename.split(".")[0]  # 例如 "omi-1"
        test_data_path = os.path.join("data/processed", f"{prefix}_test.pkl")
        test_labels_path = os.path.join("data/processed", f"{prefix}_test_label.pkl")
        
        if not (os.path.exists(test_data_path) and os.path.exists(test_labels_path)):
            print(f"缺少对应的测试数据或标签文件：{prefix}")
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
        file_stats = process_omi_file(file_path, test_data, test_labels)
        
        global_expected += file_stats["expected_sum"]
        global_true_positive += file_stats["true_positive_sum"]
        global_detected += file_stats["detected_sum"]
        
        # 输出当前文件后的全局统计
        current_recall = (global_true_positive / global_expected) if global_expected > 0 else 0
        current_precision = (global_true_positive / global_detected) if global_detected > 0 else 0
        print("当前累计总体统计:")
        print(f"  总期望异常传感器数: {global_expected}")
        print(f"  总正确检测数: {global_true_positive}")
        print(f"  总检测到异常传感器数: {global_detected}")
        print(f"  累计召回率: {current_recall*100:.1f}%")
        print(f"  累计精确率: {current_precision*100:.1f}%")
        print("="*60)
    
    # 最后输出全局总体统计
    global_recall = (global_true_positive / global_expected) if global_expected > 0 else 0
    global_precision = (global_true_positive / global_detected) if global_detected > 0 else 0
    
    print("最终总体统计:")
    print(f"  总期望异常传感器数: {global_expected}")
    print(f"  总正确检测数: {global_true_positive}")
    print(f"  总检测到异常传感器数: {global_detected}")
    print(f"  全局召回率: {global_recall*100:.1f}%")
    print(f"  全局精确率: {global_precision*100:.1f}%")


def robust_zscore(x, median_val, mad):
    """鲁棒 Z 分数：(x - median) / (MAD + 1e-8)"""
    return np.abs(x - median_val) / (mad + 1e-8)

def build_sensor_graph(data, corr_threshold=0.4, dtw_threshold=2.0):
    """
    构建传感器关系图：
    - 节点：每个传感器（共 n 个）
    - 边：基于传感器时间序列的相似性（Pearson + DTW）
    
    参数：
        data : np.ndarray, 形状为 (m, n) 的时序数据（m 时间点，n 传感器）
    """
    n_sensors = data.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_sensors))
    
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            x_i = data[:, i]
            x_j = data[:, j]
            
            # 计算 Pearson 相关系数和 DTW 距离
            # Pearson 相关系数：衡量“整体线性趋势是否一致”
            # DTW（动态时间规整）：衡量“形状是否相似（哪怕时间上错开）”
            # 例如，两个测量同一设备的传感器可能因为传输延迟导致数据偏移几秒，DTW仍能识别它们的相似性，而Pearson相关系数可能较低。
            corr = pearsonr(x_i, x_j)[0]
            dtw_dist = dtw.distance(x_i, x_j)
            
            # 添加边条件
            if abs(corr) > corr_threshold and dtw_dist < dtw_threshold:
                G.add_edge(i, j, weight=abs(corr))
    return G

def detect_anomaly_sensors(data, labels, method="kmeans", score_threshold=0.5):
    """
    检测异常传感器：
    1. 构建传感器关系图。
    2. 使用社区检测或聚类找到异常传感器组。
    3. 计算鲁棒得分并筛选最终异常传感器。
    
    参数：
        method : "louvain"（社区检测）或 "kmeans"（聚类）
    """
    # 1. 构建传感器关系图
    G = build_sensor_graph(data)
    print(f"图中节点数: {G.number_of_nodes()}")
    print(f"图中边数: {G.number_of_edges()}")

    
    # 2. 检测异常传感器组
    if method == "louvain":
        partition = community_louvain.best_partition(G)
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)
        # 假设异常社区是节点数最少的社区
        anomaly_sensors = min(communities.values(), key=len)
    elif method == "kmeans":
        # 使用 DeepWalk 生成节点嵌入
        def random_walk(G, walk_length=10, num_walks=20):
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
        model = Word2Vec(walks, vector_size=16, window=5, min_count=0, sg=1)
        embeddings = np.array([model.wv[str(i)] for i in G.nodes()])

        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        print('clusters:', clusters)
        # np.bincount(clusters) 返回每个类别的样本数,np.argmin()返回最小值的索引
        anomaly_sensors = np.where(clusters == np.argmin(np.bincount(clusters)))[0]
        # anomaly_sensors = clusters
    else:
        raise ValueError("Method must be 'louvain' or 'kmeans'")
    
    # 3. 计算异常传感器的鲁棒得分
    sensor_scores = np.zeros(data.shape[1])     # 初始化传感器得分数组
    anomaly_points = np.where(labels == 1)[0]   # 获取异常数据点的索引
    
    for sensor in anomaly_sensors:
        x = data[:, sensor]
        median_val = np.median(x)
        mad = np.median(np.abs(x - median_val))
        
        for point in anomaly_points:
            score = robust_zscore(data[point, sensor], median_val, mad)
            sensor_scores[sensor] += score
    
    avg_scores = sensor_scores / (len(anomaly_points) + 1e-8)
    # return np.where(avg_scores > score_threshold)[0]
    return anomaly_sensors
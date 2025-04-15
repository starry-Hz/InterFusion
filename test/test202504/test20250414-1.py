import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import warnings
import datetime
import logging
from PIL import Image
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

dataset = 'omi'
log_dir = f'./log/{dataset}'
os.makedirs(log_dir, exist_ok=True)
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_filename = f'{log_dir}/gridsearch_{current_date}.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)  # 全局设置为 INFO
logger.setLevel(logging.DEBUG)              # 只保留自己 logger 的 DEBUG
logger.info("===== 启动 Grid Search 流程 =====")

def visualize_anomaly_graph(test_data, start, end, edge_thresholds, topk_neighbors,
                            ground_truth_nodes, detected_nodes, save_path):
    """
    节点颜色：
    🟠 橙色（orange）     -> 检测异常
    🔵 蓝色（blue）       -> 实际异常
    🟪 紫色（purple）     -> GT + 检测都异常
    ⚪ 浅灰色（lightgray）-> 正常

    边颜色：
    🔴 红色（red）        -> 异常边
    ⚫ 深灰色（dimgray）   -> 正常边
    """
    window = test_data[start:end+1]
    corr = np.corrcoef(window, rowvar=False)
    num_sensors = corr.shape[0]
    G = nx.Graph()
    for i in range(num_sensors):
        G.add_node(i + 1)
    for j in range(num_sensors):
        for k in topk_neighbors[j]:
            key = (min(j, k), max(j, k))
            if key not in edge_thresholds:
                continue
            val = corr[j, k] if j < k else corr[k, j]
            if np.isnan(val):
                continue
            low, high = edge_thresholds[key]
            is_abnormal = val < low or val > high
            G.add_edge(j + 1, k + 1, color='red' if is_abnormal else 'dimgray')
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    node_colors = []
    for node in G.nodes():
        if node in ground_truth_nodes and node in detected_nodes:
            node_colors.append('purple')
        elif node in ground_truth_nodes:
            node_colors.append('blue')
        elif node in detected_nodes:
            node_colors.append('orange')
        else:
            node_colors.append('lightgray')
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=1.8, iterations=150)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
            node_size=500, font_size=10, width=2)
    plt.title(f"Segment {start}-{end}")
    legend_elements = [
        mpatches.Patch(color='orange', label='检测异常'),
        mpatches.Patch(color='blue', label='GT异常'),
        mpatches.Patch(color='purple', label='正确检测'),
        mpatches.Patch(color='lightgray', label='正常'),
        mpatches.Patch(color='red', label='异常边'),
        mpatches.Patch(color='dimgray', label='正常边')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize='small')
    plt.savefig(save_path)
    plt.close()

def evaluate_omi_dataset_topk(train_data, test_data, test_labels, gt_txt_path,
                              window_size=5, score_threshold=3.0, top_k=5,
                              save_fig_dir=None, score_mode='deviation', use_topk=True):
    num_sensors = train_data.shape[1]
    edge_corrs = defaultdict(list)
    for i in range(train_data.shape[0] - window_size + 1):
        window = train_data[i:i + window_size]
        corr = np.corrcoef(window, rowvar=False)
        for j in range(num_sensors):
            for k in range(j + 1, num_sensors):
                val = abs(corr[j, k])
                if not np.isnan(val):
                    edge_corrs[(j, k)].append(val)
    edge_thresholds = {
        (j, k): (np.percentile(vals, 5), np.percentile(vals, 95))
        for (j, k), vals in edge_corrs.items()
    }
    avg_corr = np.zeros((num_sensors, num_sensors))
    for (j, k), vals in edge_corrs.items():
        mean_val = np.mean(vals)
        avg_corr[j, k] = mean_val
        avg_corr[k, j] = mean_val
    topk_neighbors = {
        i: set(np.argsort(-avg_corr[i])[:top_k + 1]) - {i}
        for i in range(num_sensors)
    }
    def detect_anomalous_nodes(start, end):
        abnormal_score = defaultdict(float)
        for i in range(start, end - window_size + 2):
            window = test_data[i:i + window_size]
            corr = np.corrcoef(window, rowvar=False)
            for j in range(num_sensors):
                neighbors = topk_neighbors[j] if use_topk else set(range(num_sensors)) - {j}
                for k in neighbors:
                    key = (min(j, k), max(j, k))
                    if key not in edge_thresholds:
                        continue
                    val = corr[j, k] if j < k else corr[k, j]
                    if np.isnan(val):
                        continue
                    low, high = edge_thresholds[key]
                    mean_val = avg_corr[j, k]
                    range_val = high - low
                    if val < low or val > high:
                        if score_mode == 'deviation':
                            deviation = max(abs(val - low), abs(val - high))
                        elif score_mode == 'mean_ratio':
                            deviation = abs(val / mean_val) if mean_val != 0 else 0
                        elif score_mode == 'range_ratio':
                            deviation = abs(val / range_val) if range_val != 0 else 0
                        elif score_mode == 'value_times_range':
                            deviation = abs(val * range_val)
                        else:
                            raise ValueError(f"未知异常分数计算模式：{score_mode}")
                        abnormal_score[j] += deviation
                        abnormal_score[k] += deviation
        return {node + 1 for node, score in abnormal_score.items() if score >= score_threshold}

    gt_segments = []
    with open(gt_txt_path, 'r') as f:
        for line in f:
            time_range, sensors_str = line.strip().split(':')
            start, end = map(int, time_range.split('-'))
            sensors = list(map(int, sensors_str.split(',')))
            gt_segments.append({'start': start, 'end': end, 'sensors': sensors})

    if save_fig_dir:
        os.makedirs(save_fig_dir, exist_ok=True)

    total_TP = total_FP = total_FN = 0
    for segment in gt_segments:
        start, end = segment['start'], segment['end']
        gt_sensors = set(segment['sensors'])
        predicted = detect_anomalous_nodes(start, end)
        TP = len(predicted & gt_sensors)
        FP = len(predicted - gt_sensors)
        FN = len(gt_sensors - predicted)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        if save_fig_dir:
            save_path = os.path.join(save_fig_dir, f"{start}-{end}.png")
            visualize_anomaly_graph(test_data, start, end, edge_thresholds, topk_neighbors,
                                    ground_truth_nodes=gt_sensors,
                                    detected_nodes=predicted,
                                    save_path=save_path)
    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def grid_search_with_visualization(train_data, test_data, test_labels, gt_txt_path,
                                   topk_list, threshold_list,
                                   save_base_dir=None,
                                   score_mode='deviation', use_topk=True):
    results = []
    best_f1 = -1
    best_combo = None
    for top_k in topk_list:
        for threshold in threshold_list:
            save_fig_dir = None
            if save_base_dir:
                save_fig_dir = os.path.join(save_base_dir, f"topk{top_k}_thr{threshold:.1f}")
            precision, recall, f1 = evaluate_omi_dataset_topk(
                train_data, test_data, test_labels, gt_txt_path,
                window_size=5,
                score_threshold=threshold,
                top_k=top_k,
                save_fig_dir=save_fig_dir,
                score_mode=score_mode,
                use_topk=use_topk
            )
            results.append({
                'top_k': top_k,
                'score_threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            if f1 > best_f1:
                best_f1 = f1
                best_combo = results[-1]
    return pd.DataFrame(results), best_combo

def run_batch_grid_search(data_folder, label_folder, output_dir,
                          topk_list, threshold_list,
                          score_mode='deviation', use_topk=True):
    all_results = []
    txt_files = [f for f in os.listdir(label_folder) if f.startswith(f"{dataset}-") and f.endswith(".txt")]
    for txt_file in txt_files:
        base = txt_file.replace(".txt", "")
        train_path = os.path.join(data_folder, f"{base}_train.pkl")
        test_path = os.path.join(data_folder, f"{base}_test.pkl")
        label_path = os.path.join(data_folder, f"{base}_test_label.pkl")
        gt_txt_path = os.path.join(label_folder, txt_file)
        if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(label_path)):
            logger.warning(f"[{base}] 缺少文件，跳过")
            continue
        train_data = pd.read_pickle(train_path)
        test_data = pd.read_pickle(test_path)
        test_labels = pd.read_pickle(label_path)
        logger.info(f"[{base}] 启动网格搜索...")
        save_fig_dir = os.path.join(output_dir, base)
        df, best = grid_search_with_visualization(
            train_data, test_data, test_labels, gt_txt_path,
            topk_list, threshold_list,
            save_base_dir=save_fig_dir,
            score_mode=score_mode,
            use_topk=use_topk
        )
        df['dataset'] = base
        all_results.append(df)
        logger.info(f"[{base}] 最佳参数: top_k={best['top_k']} threshold={best['score_threshold']} F1={best['f1_score']:.4f}")
    return pd.concat(all_results, ignore_index=True)

if __name__ == "__main__":
    start = datetime.datetime.now()
    print(f"开始时间: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    topk_list = np.arange(1, 10)
    threshold_list = np.arange(0.0, 7.1, 0.5)
    result_df = run_batch_grid_search(
        data_folder='./data/processed',
        label_folder='./data/interpretation_label',
        output_dir=f'./figures/{dataset}/grid',
        topk_list=topk_list,
        threshold_list=threshold_list,
        score_mode='range_ratio',
        use_topk=False
    )
    best_results = result_df.sort_values(by='f1_score', ascending=False).groupby('dataset').first().reset_index()
    print("\n===== 每个数据集最优组合 =====")
    print(best_results[['dataset', 'top_k', 'score_threshold', 'precision', 'recall', 'f1_score']])
    end = datetime.datetime.now()
    print(f"结束时间: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"运行完成，用时：{end - start}")

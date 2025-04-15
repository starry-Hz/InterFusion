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
import matplotlib.patches as mpatches  # 新增导入，用于图例
# 设置全局字体为 SimHei（黑体）
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False # 解决坐标轴负号显示问题

warnings.filterwarnings("ignore")
dataset = 'omi'
# dataset = 'machine'

# ========== 日志设置 ==========
log_dir = f'./log/{dataset}'
os.makedirs(log_dir, exist_ok=True)
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_filename = f'{log_dir}/{current_date}.log'
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

logger.info("python test20250414-0.py")
logger.info("train 下界:5% 上界:95%,pearson相关系数取绝对值,生成Gif图画")
logger.info("===== 日志初始化完成 =====")

# ---------- 可视化函数 ----------
# 每个传感器是一个节点；节点之间的边表示两个传感器的相关性；异常边和异常节点用颜色突出显示；
def visualize_anomaly_graph(test_data, start, end, edge_thresholds, topk_neighbors,
                            ground_truth_nodes, detected_nodes, save_path):
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
            G.add_edge(j + 1, k + 1, color='red' if is_abnormal else 'gray')

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
    # pos = nx.spring_layout(G, seed=42)
    pos = nx.spring_layout(G, seed=42, k=1.2, iterations=100)

    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
            node_size=500, font_size=10, width=2)
    plt.title(f"Segment {start}-{end}")

    # 添加图例
    legend_elements = [
        mpatches.Patch(color='orange', label='检测异常'),
        mpatches.Patch(color='blue', label='GT异常'),   # ground truth异常
        mpatches.Patch(color='purple', label='正确检测'),
        mpatches.Patch(color='lightgray', label='正常'),
        mpatches.Patch(color='red', label='异常边'),
        mpatches.Patch(color='gray', label='正常边')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize='small')

    plt.savefig(save_path)
    plt.close()

'''
节点颜色：
🟠 orange  -> 检测异常
🔵 blue    -> 实际异常
🟪 purple  -> GT + 检测都异常
⚪ gray    -> 正常

边颜色：
🔴 red     -> 异常边
⚫ gray    -> 正常边
'''

def create_gif_from_images(image_folder, gif_path, duration=500):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".png"):
            img_path = os.path.join(image_folder, filename)
            images.append(Image.open(img_path))
    if images:
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

# ---------- 单个数据集评估函数 ----------
def evaluate_omi_dataset_topk(train_data, test_data, test_labels, gt_txt_path,
                              window_size=5, score_threshold=3.0, top_k=5, save_fig_dir=None):
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
                for k in topk_neighbors[j]:
                    key = (min(j, k), max(j, k))
                    if key not in edge_thresholds:
                        continue
                    val = corr[j, k] if j < k else corr[k, j]
                    if np.isnan(val):
                        continue
                    low, high = edge_thresholds[key]
                    if val < low or val > high:
                        deviation = max(abs(val - low), abs(val - high))
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
            visualize_anomaly_graph(
                test_data, start, end, edge_thresholds, topk_neighbors,
                ground_truth_nodes=gt_sensors,
                detected_nodes=predicted,
                save_path=save_path
            )

    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

# ---------- 汇总最优结果 + 全局指标 ----------
def summarize_best_results(results_df):
    best_per_dataset = results_df.sort_values(by='f1_score', ascending=False).groupby('dataset').first().reset_index()

    total_TP = total_FP = total_FN = 0
    for _, row in best_per_dataset.iterrows():
        precision = row['precision']
        recall = row['recall']
        TP = 100
        FP = int(round((TP / precision) - TP)) if precision > 0 else 0
        FN = int(round((TP / recall) - TP)) if recall > 0 else 0
        total_TP += TP
        total_FP += FP
        total_FN += FN

    overall_precision = total_TP / (total_TP + total_FP)
    overall_recall = total_TP / (total_TP + total_FN)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)

    logger.info("\n===== 每个数据集最优组合 =====\n" + best_per_dataset.to_string(index=False))
    logger.info(f"\n===== 全部数据集指标 =====\nPrecision: {overall_precision:.4f}  Recall: {overall_recall:.4f}  F1: {overall_f1:.4f}")

    return best_per_dataset, {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_score': overall_f1
    }

# ---------- 主流程 ----------
if __name__ == "__main__":
    start = datetime.datetime.now()
    print(f"开始时间: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"开始时间: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    txt_files = [f for f in os.listdir('./data/interpretation_label') if f.startswith(f"{dataset}-") and f.endswith(".txt")]
    all_results = []

    for txt_file in txt_files:
        base = txt_file.replace('.txt', '')
        train_path = os.path.join('./data/processed', f"{base}_train.pkl")
        test_path = os.path.join('./data/processed', f"{base}_test.pkl")
        label_path = os.path.join('./data/processed', f"{base}_test_label.pkl")
        gt_txt_path = os.path.join('./data/interpretation_label', txt_file)

        if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(label_path)):
            logger.warning(f"[{base}] 缺少文件，跳过")
            continue

        train_data = pd.read_pickle(train_path)
        test_data = pd.read_pickle(test_path)
        test_labels = pd.read_pickle(label_path)

        save_fig_dir = os.path.join('./figures', dataset, base)
        logger.info(f"[{base}] 开始异常检测并生成图像...")

        precision, recall, f1 = evaluate_omi_dataset_topk(
            train_data, test_data, test_labels, gt_txt_path,
            window_size=5, score_threshold=3.0, top_k=5,
            save_fig_dir=save_fig_dir
        )

        gif_path = os.path.join('./figures', dataset, f"{base}.gif")
        create_gif_from_images(save_fig_dir, gif_path, duration=600)

        logger.info(f"[{base}] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        all_results.append({
            'dataset': base,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

    results_df = pd.DataFrame(all_results)
    best_results, overall = summarize_best_results(results_df)

    print("\n===== 每个数据集最优参数组合 =====")
    print(best_results[['dataset', 'precision', 'recall', 'f1_score']])

    print("\n===== 全部数据集的整体指标 =====")
    print(f"Precision: {overall['overall_precision']:.4f}")
    print(f"Recall:    {overall['overall_recall']:.4f}")
    print(f"F1-score:  {overall['overall_f1_score']:.4f}")

    end = datetime.datetime.now()
    print(f"结束时间: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结束时间: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"运行完成，用时：{end - start}")

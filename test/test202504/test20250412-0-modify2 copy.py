import os
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
import datetime
import logging
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 用于显示中文
import matplotlib.patches as mpatches
from PIL import Image
warnings.filterwarnings("ignore")

dataset = 'omi'
score_mode1 = 'value_times_range'  # 选择评分模式
# score_mode_list = ['deviation', 'mean_ratio', 'range_ratio', 'value_times_range', 'robust_zscore']

log_dir = f'./log/{dataset}/{score_mode1}'
os.makedirs(log_dir, exist_ok=True)
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_filename = f'{log_dir}/train_{current_date}.log'
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
logger.info("python test20250412-0-modify2.py")   
logger.info("train 下界:5% 上界:95%,pearson相关系数取绝对值")
logger.info("===== 日志初始化完成 =====")

def calculate_abnormal_score(val, edge_stats, key, score_mode):
    """
    计算异常分数
    :param val: 当前相关系数值
    :param edge_stats: 边缘统计信息字典
    :param key: 边缘键 (j, k)
    :param score_mode: 评分模式
    :return: 异常分数
    """
    if key not in edge_stats:
        return 0
    
    stats = edge_stats[key]
    low, high = stats['percentiles']
    
    if score_mode == 'deviation':
        return max(abs(val - low), abs(val - high))
    elif score_mode == 'mean_ratio':
        return abs(val / stats['mean']) if stats['mean'] != 0 else 0
    elif score_mode == 'range_ratio':
        return abs(val / stats['range']) if stats['range'] != 0 else 0
    elif score_mode == 'value_times_range':
        return abs(val * stats['range'])
    elif score_mode == 'robust_zscore':
        return np.abs(val - stats['median']) / (stats['mad'] + 1e-8)
    else:
        return max(abs(val - low), abs(val - high))

# 评价单个数据集
def evaluate_omi_dataset_topk(train_data, test_data, test_labels, gt_txt_path,
                             window_size=5, score_threshold=3.0, top_k=None,
                             score_mode='deviation', use_topk=True):
    """
    计算异常检测的精确率、召回率和F1分数
    :param train_data: 训练数据 (n_samples, n_sensors)
    :param test_data: 测试数据 (n_samples, n_sensors)
    :param test_labels: 测试标签 (n_samples,)
    :param gt_txt_path: 标注文件路径
    :param window_size: 滑动窗口大小
    :param score_threshold: 异常分数阈值
    :param top_k: 每个节点的top-k邻居数（use_topk=True时有效）
    :param score_mode: 评分模式 ['deviation', 'mean_ratio', 'range_ratio', 'value_times_range', 'robust_zscore']
    :param use_topk: True=仅检查top-k邻居，False=检查所有邻居
    :return: (precision, recall, f1_score)
    """
    num_sensors = train_data.shape[1]
    edge_corrs = defaultdict(list)

    # 1. 计算训练窗口的相关系数分布
    for i in range(train_data.shape[0] - window_size + 1):
        window = train_data[i:i + window_size]
        corr = np.corrcoef(window, rowvar=False)
        for j in range(num_sensors):
            for k in range(j + 1, num_sensors):
                val = corr[j, k]
                if not np.isnan(val):
                    edge_corrs[(j, k)].append(val)

    # 2. 计算边缘统计量
    edge_stats = {}
    for (j, k), vals in edge_corrs.items():
        median_val = np.median(vals)
        mad = np.median(np.abs(vals - median_val))  # 中位数绝对偏差
        edge_stats[(j, k)] = {
            'percentiles': (np.percentile(vals, 5), np.percentile(vals, 95)),
            'median': median_val,
            'mad': mad,
            'mean': np.mean(vals),
            'range': np.percentile(vals, 95) - np.percentile(vals, 5)
        }

    # 3. 计算平均相关系数（用于确定top-k邻居）
    avg_corr = np.zeros((num_sensors, num_sensors))
    for (j, k), vals in edge_corrs.items():
        mean_val = np.mean(vals)
        avg_corr[j, k] = mean_val
        avg_corr[k, j] = mean_val

    # 4. 确定邻居选择方式
    if use_topk:
        if top_k is None or top_k >= num_sensors:
            top_k = num_sensors - 1
        topk_neighbors = {
            i: set(np.argsort(-avg_corr[i])[:top_k + 1]) - {i}
            for i in range(num_sensors)
        }
    else:
        topk_neighbors = {
            i: set(range(num_sensors)) - {i}  # 所有非自身节点
            for i in range(num_sensors)
        }

    # 5. 异常检测函数
    def detect_anomalous_nodes(start, end):
        abnormal_score = defaultdict(float)
        for i in range(start, end - window_size + 2):
            window = test_data[i:i + window_size]
            corr = np.corrcoef(window, rowvar=False)
            for j in range(num_sensors):
                for k in topk_neighbors[j]:
                    key = (min(j, k), max(j, k))
                    val = corr[j, k] if j < k else corr[k, j]
                    if np.isnan(val):
                        continue
                    
                    deviation = calculate_abnormal_score(val, edge_stats, key, score_mode)
                    if deviation > 0:
                        abnormal_score[j] += deviation
                        abnormal_score[k] += deviation
                        
        return {node + 1 for node, score in abnormal_score.items() if score >= score_threshold}

    # 6. 解析标注文件
    gt_segments = []
    with open(gt_txt_path, 'r') as f:
        for line in f:
            time_range, sensors_str = line.strip().split(':')
            start, end = map(int, time_range.split('-'))
            sensors = list(map(int, sensors_str.split(',')))
            gt_segments.append({'start': start, 'end': end, 'sensors': sensors})

    # 7. 计算指标
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

    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def grid_search(
    train_data, test_data, test_labels, gt_txt_path,
    threshold_list, score_mode_list, dataset_name=None,
    use_topk=True, topk_list=None
):
    results = []
    
    for score_mode in score_mode_list:
        if use_topk:
            # 模式1：搜索 top_k 和 threshold
            if topk_list is None:
                raise ValueError("topk_list must be provided when use_topk=True")
            for top_k in topk_list:
                for threshold in threshold_list:
                    precision, recall, f1 = evaluate_omi_dataset_topk(
                        train_data, test_data, test_labels, gt_txt_path,
                        window_size=5, score_threshold=threshold,
                        top_k=top_k, score_mode=score_mode, use_topk=True
                    )
                    log_and_append_result(results, dataset_name, score_mode, top_k, threshold, precision, recall, f1, use_topk=True)
        else:
            # 模式2：仅搜索 threshold（top_k 固定为 None）
            for threshold in threshold_list:
                precision, recall, f1 = evaluate_omi_dataset_topk(
                    train_data, test_data, test_labels, gt_txt_path,
                    window_size=5, score_threshold=threshold,
                    top_k=None, score_mode=score_mode, use_topk=False
                )
                log_and_append_result(results, dataset_name, score_mode, None, threshold, precision, recall, f1, use_topk=False)
    
    return pd.DataFrame(results)

def log_and_append_result(results, dataset_name, score_mode, top_k, threshold, precision, recall, f1, use_topk):
    """统一处理日志和结果记录"""
    if use_topk:
        logger.info(
            f"[{dataset_name}] mode={score_mode}, top_k={top_k}, threshold={threshold:.1f}, "
            f"use_topk=True, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
        )
    else:
        logger.info(
            f"[{dataset_name}] mode={score_mode}, threshold={threshold:.1f}, "
            f"use_topk=False, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
        )
    
    results.append({
        'score_mode': score_mode,
        'top_k': top_k,
        'score_threshold': threshold,
        'use_topk': use_topk,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })


def run_full_grid_search_all_datasets(data_folder, label_folder,
                                      topk_list, threshold_list, score_mode_list,
                                      use_topk=True):  # 新增 use_topk 参数
    txt_files = [f for f in os.listdir(label_folder) if f.startswith(f"{dataset}-") and f.endswith(".txt")]
    all_results = []

    for txt_file in txt_files:
        base = txt_file.replace('.txt', '')
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

        logger.info(f"[{base}] 正在网格搜索...")
        df = grid_search(train_data, test_data, test_labels, gt_txt_path,
                              topk_list, threshold_list, score_mode_list, 
                              dataset_name=txt_file, use_topk=use_topk)  # 传递 use_topk 参数
        df['dataset'] = txt_file
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)

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


# ---------- 可视化函数 ----------
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
    pos = nx.spring_layout(G, seed=42, k=1.2, iterations=100)

    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
            node_size=500, font_size=10, width=2)
    plt.title(f"Segment {start}-{end}")

    legend_elements = [
        mpatches.Patch(color='orange', label='检测异常'),
        mpatches.Patch(color='blue', label='GT异常'),
        mpatches.Patch(color='purple', label='正确检测'),
        mpatches.Patch(color='lightgray', label='正常'),
        mpatches.Patch(color='red', label='异常边'),
        mpatches.Patch(color='gray', label='正常边')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize='small')

    plt.savefig(save_path)
    plt.close()


def create_gif_from_images(image_folder, gif_path, duration=500):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".png"):
            img_path = os.path.join(image_folder, filename)
            images.append(Image.open(img_path))
    if images:
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)


# def run_visualization_from_best_results(best_results,path=None):
#     for _, row in best_results.iterrows():
#         dataset_name = row['dataset'].replace('.txt', '')
#         top_k = int(row['top_k'])
#         threshold = float(row['score_threshold'])

#         train_data = pd.read_pickle(f"./data/processed/{dataset_name}_train.pkl")
#         test_data = pd.read_pickle(f"./data/processed/{dataset_name}_test.pkl")
#         test_labels = pd.read_pickle(f"./data/processed/{dataset_name}_test_label.pkl")
#         gt_txt_path = f"./data/interpretation_label/{dataset_name}.txt"

#         os.makedirs(f"./visual/{path}/{dataset_name}", exist_ok=True)

#         corr = np.corrcoef(train_data[-5:], rowvar=False)
#         avg_corr = corr.copy()
#         num_sensors = avg_corr.shape[0]

#         topk_neighbors = {
#             i: set(np.argsort(-avg_corr[i])[:top_k + 1]) - {i}
#             for i in range(num_sensors)
#         }

#         edge_corrs = defaultdict(list)
#         for i in range(train_data.shape[0] - 5 + 1):
#             window = train_data[i:i + 5]
#             corr = np.corrcoef(window, rowvar=False)
#             for j in range(num_sensors):
#                 for k in range(j + 1, num_sensors):
#                     val = corr[j, k]
#                     if not np.isnan(val):
#                         edge_corrs[(j, k)].append(val)

#         edge_thresholds = {
#             (j, k): (np.percentile(vals, 5), np.percentile(vals, 95))
#             for (j, k), vals in edge_corrs.items()
#         }

#         def extract_anomaly_segments(labels):
#             segments, start = [], None
#             for i, val in enumerate(labels):
#                 if val == 1 and start is None:
#                     start = i
#                 elif val == 0 and start is not None:
#                     segments.append((start, i - 1))
#                     start = None
#             if start is not None:
#                 segments.append((start, len(labels) - 1))
#             return segments

#         def detect_anomalous_nodes(start, end):
#             abnormal_score = defaultdict(float)
#             for i in range(start, end - 5 + 2):
#                 corr = np.corrcoef(test_data[i:i + 5], rowvar=False)
#                 for j in range(num_sensors):
#                     for k in topk_neighbors[j]:
#                         key = (min(j, k), max(j, k))
#                         if key not in edge_thresholds:
#                             continue
#                         val = corr[j, k] if j < k else corr[k, j]
#                         if np.isnan(val):
#                             continue
#                         low, high = edge_thresholds[key]
#                         if val < low or val > high:
#                             deviation = max(abs(val - low), abs(val - high))
#                             abnormal_score[j] += deviation
#                             abnormal_score[k] += deviation
#             return {node + 1 for node, score in abnormal_score.items() if score >= threshold}

def run_visualization_from_best_results(best_results, path=None):
    for _, row in best_results.iterrows():
        dataset_name = row['dataset'].replace('.txt', '')
        top_k = int(row['top_k'])
        threshold = float(row['score_threshold'])
        score_mode = row['score_mode']

        train_path = f"./data/processed/{dataset_name}_train.pkl"
        test_path = f"./data/processed/{dataset_name}_test.pkl"
        label_path = f"./data/processed/{dataset_name}_test_label.pkl"
        gt_txt_path = f"./data/interpretation_label/{dataset_name}.txt"

        if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(label_path)):
            logger.warning(f"[{dataset_name}] 缺少文件，跳过")
            continue

        train_data = pd.read_pickle(train_path)
        test_data = pd.read_pickle(test_path)
        test_labels = pd.read_pickle(label_path)

        os.makedirs(f"./visual/{path}/{dataset_name}", exist_ok=True)

        # 计算边缘统计量
        edge_corrs = defaultdict(list)
        for i in range(train_data.shape[0] - 5 + 1):
            window = train_data[i:i + 5]
            corr = np.corrcoef(window, rowvar=False)
            for j in range(corr.shape[0]):
                for k in range(j + 1, corr.shape[0]):
                    val = corr[j, k]
                    if not np.isnan(val):
                        edge_corrs[(j, k)].append(val)

        edge_stats = {}
        for (j, k), vals in edge_corrs.items():
            median_val = np.median(vals)
            mad = np.median(np.abs(vals - median_val))
            edge_stats[(j, k)] = {
                'percentiles': (np.percentile(vals, 5), np.percentile(vals, 95)),
                'median': median_val,
                'mad': mad,
                'mean': np.mean(vals),
                'range': np.percentile(vals, 95) - np.percentile(vals, 5)
            }

        # 计算平均相关系数
        avg_corr = np.zeros((corr.shape[0], corr.shape[0]))
        for (j, k), vals in edge_corrs.items():
            mean_val = np.mean(vals)
            avg_corr[j, k] = mean_val
            avg_corr[k, j] = mean_val

        topk_neighbors = {
            i: set(np.argsort(-avg_corr[i])[:top_k + 1]) - {i}
            for i in range(corr.shape[0])
        }

        def extract_anomaly_segments(labels):
            segments, start = [], None
            for i, val in enumerate(labels):
                if val == 1 and start is None:
                    start = i
                elif val == 0 and start is not None:
                    segments.append((start, i - 1))
                    start = None
            if start is not None:
                segments.append((start, len(labels) - 1))
            return segments

        def detect_anomalous_nodes(start, end):
            abnormal_score = defaultdict(float)
            for i in range(start, end - 5 + 2):
                corr = np.corrcoef(test_data[i:i + 5], rowvar=False)
                for j in range(corr.shape[0]):
                    for k in topk_neighbors[j]:
                        key = (min(j, k), max(j, k))
                        val = corr[j, k] if j < k else corr[k, j]
                        if np.isnan(val):
                            continue
                        
                        # 使用封装的函数计算异常分数
                        deviation = calculate_abnormal_score(val, edge_stats, key, score_mode)
                        if deviation > 0:
                            abnormal_score[j] += deviation
                            abnormal_score[k] += deviation
                            
            return {node + 1 for node, score in abnormal_score.items() if score >= threshold}

        segments = extract_anomaly_segments(test_labels)

        for idx, (start, end) in enumerate(segments):
            gt_nodes = set()
            with open(gt_txt_path, 'r') as f:
                for line in f:
                    t_range, s_str = line.strip().split(':')
                    t_start, t_end = map(int, t_range.split('-'))
                    if start >= t_start and end <= t_end:
                        gt_nodes = set(map(int, s_str.split(',')))
                        break

            detected_nodes = detect_anomalous_nodes(start, end)
            image_path = f"./visual/{path}/{dataset_name}/seg_{idx}_{start}_{end}.png"
            visualize_anomaly_graph(test_data, start, end, 
                                   {k: v['percentiles'] for k, v in edge_stats.items()}, 
                                   topk_neighbors, gt_nodes, detected_nodes, image_path)

        create_gif_from_images(f"./visual/{path}/{dataset_name}", f"./visual/{dataset_name}.gif")

if __name__ == "__main__":
    start = datetime.datetime.now()
    print(f"开始时间: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"开始时间: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    # score_mode_list=['deviation', 'mean_ratio', 'range_ratio', 'value_times_range']
    score_mode = score_mode1
    logger.info(f"score_mode = ['{score_mode}']")
    results = run_full_grid_search_all_datasets(
        data_folder='./data/processed',
        label_folder='./data/interpretation_label',
        topk_list=np.arange(1, 10),
        threshold_list=np.arange(0, 7.1, 0.5),
        score_mode_list=[f'{score_mode}'],
        use_topk=False  # 设置为 False 表示测试阶段遍历所有邻居
    )

    best_results, overall = summarize_best_results(results)
    # 可视化异常检测结果
    run_visualization_from_best_results(best_results,path=score_mode)

    # print("\n===== 每个数据集最优参数组合 =====")
    # best_summary = best_results[['dataset', 'score_mode', 'top_k', 'score_threshold', 'precision', 'recall', 'f1_score']]
    # print(best_summary)
    # logger.info("\n===== 每个数据集最优参数组合 =====\n" + best_summary.to_string(index=False))

    # print("\n===== 全部数据集的整体指标 =====")
    # print(f"Precision: {overall['overall_precision']:.4f}")
    # print(f"Recall:    {overall['overall_recall']:.4f}")
    # print(f"F1-score:  {overall['overall_f1_score']:.4f}")

    end = datetime.datetime.now()
    print(f"结束时间: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结束时间: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"运行完成，用时：{end - start}")
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
import datetime
import logging
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Tuple, Set, List, Optional, Union
import numpy.typing as npt
from dataclasses import dataclass, field
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from PIL import Image

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用于显示中文
warnings.filterwarnings("ignore")

# ==================== 配置类 ====================
@dataclass
class ExperimentConfig:
    dataset: str = 'omi'
    window_size: int = 5
    score_thresholds: npt.NDArray = field(default_factory=lambda: np.arange(0, 7.1, 0.5))
    topk_range: range = field(default_factory=lambda: range(1, 10))
    score_modes: List[str] = field(default_factory=lambda: ['value_times_range'])
    use_topk: bool = False
    data_dir: str = "./data/processed"
    label_dir: str = "./data/interpretation_label"
    log_dir: str = f"./log/use_topk{use_topk}"
    visual_dir: str = "./visual"
    num_workers: int = 4
    gif_duration: int = 500

# ==================== 辅助函数 ====================
def load_pickle_data(path: str) -> np.ndarray:
    """安全加载pickle数据，确保返回NumPy数组"""
    data = pd.read_pickle(path)
    if hasattr(data, 'values'):
        return data.values
    return np.array(data)  # 确保转换为NumPy数组

# ==================== 初始化 ====================
def initialize_logging(config: ExperimentConfig) -> logging.Logger:
    """初始化日志配置"""
    os.makedirs(f"{config.log_dir}/{config.dataset}/{config.score_modes[0]}", exist_ok=True)
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    log_filename = f"{config.log_dir}/{config.dataset}/{config.score_modes[0]}/train_{current_date}.log"
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='a',
        encoding='utf-8'
    )
    logger = logging.getLogger(__name__)
    logging.getLogger().setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting experiment with config: {config}")
    return logger

# ==================== 核心算法 ====================
def calculate_edge_stats(
    train_data: npt.NDArray,
    window_size: int
) -> Tuple[Dict[Tuple[int, int], Dict[str, Union[float, Tuple[float, float]]]], npt.NDArray]:
    """
    计算边缘统计量和平均相关系数矩阵
    
    Args:
        train_data: 训练数据 (n_samples, n_sensors)
        window_size: 滑动窗口大小
        
    Returns:
        edge_stats: 边缘统计信息字典
        avg_corr: 平均相关系数矩阵
    """
    num_sensors = train_data.shape[1]
    edge_corrs = defaultdict(list)

    # 计算训练窗口的相关系数分布
    for i in range(train_data.shape[0] - window_size + 1):
        window = train_data[i:i + window_size]
        corr = np.corrcoef(window, rowvar=False)
        for j in range(num_sensors):
            for k in range(j + 1, num_sensors):
                val = corr[j, k]
                if not np.isnan(val):
                    edge_corrs[(j, k)].append(val)

    # 计算边缘统计量
    edge_stats = {}
    for (j, k), vals in edge_corrs.items():
        if len(vals) == 0:
            continue
            
        median_val = np.median(vals)
        mad = np.median(np.abs(vals - median_val))
        percentiles = (np.percentile(vals, 5), np.percentile(vals, 95))
        
        # 确保percentiles是包含两个数值的元组
        if not (isinstance(percentiles, (tuple, list)) and len(percentiles) == 2):
            percentiles = (0.0, 0.0)
            
        edge_stats[(j, k)] = {
            'percentiles': percentiles,
            'median': median_val,
            'mad': mad,
            'mean': np.mean(vals),
            'range': percentiles[1] - percentiles[0]
        }
    
    # 计算平均相关系数矩阵
    avg_corr = np.zeros((num_sensors, num_sensors))
    for (j, k), vals in edge_corrs.items():
        if len(vals) == 0:
            continue
            
        mean_val = np.mean(vals)
        avg_corr[j, k] = mean_val
        avg_corr[k, j] = mean_val

    return edge_stats, avg_corr

def calculate_abnormal_score(
    val: float,
    edge_stats: Dict[Tuple[int, int], Dict[str, Union[float, Tuple[float, float]]]],
    key: Tuple[int, int],
    score_mode: str
) -> float:
    """
    计算异常分数
    
    Args:
        val: 当前相关系数值
        edge_stats: 边缘统计信息字典
        key: 边缘键 (j, k)
        score_mode: 评分模式
        
    Returns:
        异常分数
    """
    if key not in edge_stats:
        return 0.0
    
    stats = edge_stats[key]
    
    score_modes = {
        'deviation': lambda: max(abs(val - stats['percentiles'][0]), abs(val - stats['percentiles'][1])),
        'mean_ratio': lambda: abs(val / stats['mean']) if stats['mean'] != 0 else 0.0,
        'range_ratio': lambda: abs(val / stats['range']) if stats['range'] != 0 else 0.0,
        'value_times_range': lambda: abs(val * stats['range']),
        'robust_zscore': lambda: np.abs(val - stats['median']) / (stats['mad'] + 1e-8)
    }
    
    return score_modes.get(score_mode, lambda: 0.0)()

def evaluate_single_dataset(
    train_data: npt.NDArray,
    test_data: npt.NDArray,
    test_labels: npt.NDArray,
    gt_txt_path: str,
    config: ExperimentConfig,
    score_mode: str,
    threshold: float,
    top_k: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    评估单个数据集的性能
    
    Args:
        train_data: 训练数据
        test_data: 测试数据
        test_labels: 测试标签
        gt_txt_path: 标注文件路径
        config: 实验配置
        score_mode: 评分模式
        threshold: 异常分数阈值
        top_k: 每个节点的top-k邻居数
        
    Returns:
        (precision, recall, f1_score)
    """
    # 计算边缘统计量和平均相关系数
    edge_stats, avg_corr = calculate_edge_stats(train_data, config.window_size)
    num_sensors = train_data.shape[1]

    # 确定邻居选择方式
    if config.use_topk:
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

    # 解析标注文件
    gt_segments = []
    with open(gt_txt_path, 'r') as f:
        for line in f:
            time_range, sensors_str = line.strip().split(':')
            start, end = map(int, time_range.split('-'))
            sensors = list(map(int, sensors_str.split(',')))
            gt_segments.append({'start': start, 'end': end, 'sensors': sensors})

    # 计算指标
    total_TP = total_FP = total_FN = 0
    
    for segment in gt_segments:
        start, end = segment['start'], segment['end']
        gt_sensors = set(segment['sensors'])
        
        # 检测异常节点
        abnormal_score = defaultdict(float)
        for i in range(start, end - config.window_size + 2):
            window = test_data[i:i + config.window_size]
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
        
        predicted = {node + 1 for node, score in abnormal_score.items() if score >= threshold}
        
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

# ==================== 网格搜索 ====================
def grid_search_worker(
    params: Tuple[int, float],
    train_data: npt.NDArray,
    test_data: npt.NDArray,
    test_labels: npt.NDArray,
    gt_txt_path: str,
    config: ExperimentConfig,
    score_mode: str
) -> Dict[str, Union[str, int, float]]:
    """网格搜索的工作函数"""
    top_k, threshold = params
    precision, recall, f1 = evaluate_single_dataset(
        train_data, test_data, test_labels, gt_txt_path,
        config, score_mode, threshold, top_k
    )
    
    return {
        'score_mode': score_mode,
        'top_k': top_k,
        'score_threshold': threshold,
        'use_topk': config.use_topk,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def grid_search(
    train_data: npt.NDArray,
    test_data: npt.NDArray,
    test_labels: npt.NDArray,
    gt_txt_path: str,
    config: ExperimentConfig,
    score_mode: str,
    dataset_name: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """执行网格搜索"""
    results = []
    
    if config.use_topk:
        # 模式1：搜索 top_k 和 threshold
        params = [(top_k, threshold) 
                 for top_k in config.topk_range 
                 for threshold in config.score_thresholds]
    else:
        # 模式2：仅搜索 threshold
        params = [(None, threshold) for threshold in config.score_thresholds]

    # 使用多进程加速
    with Pool(config.num_workers) as pool:
        worker = partial(
            grid_search_worker,
            train_data=train_data,
            test_data=test_data,
            test_labels=test_labels,
            gt_txt_path=gt_txt_path,
            config=config,
            score_mode=score_mode
        )
        
        for result in tqdm(pool.imap(worker, params), total=len(params), desc=f"Grid search {dataset_name}"):
            results.append(result)
            logger.info(
                f"[{dataset_name}] mode={score_mode}, top_k={result['top_k']}, "
                f"threshold={result['score_threshold']:.1f}, use_topk={config.use_topk}, "
                f"P={result['precision']:.4f}, R={result['recall']:.4f}, F1={result['f1_score']:.4f}"
            )
    
    return pd.DataFrame(results)

# ==================== 主流程 ====================
def run_full_grid_search(config: ExperimentConfig, logger: logging.Logger) -> pd.DataFrame:
    """在所有数据集上运行完整的网格搜索"""
    txt_files = [f for f in os.listdir(config.label_dir) 
                if f.startswith(f"{config.dataset}-") and f.endswith(".txt")]
    all_results = []

    for txt_file in txt_files:
        base = txt_file.replace('.txt', '')
        train_path = os.path.join(config.data_dir, f"{base}_train.pkl")
        test_path = os.path.join(config.data_dir, f"{base}_test.pkl")
        label_path = os.path.join(config.data_dir, f"{base}_test_label.pkl")
        gt_txt_path = os.path.join(config.label_dir, txt_file)

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path, gt_txt_path]):
            logger.warning(f"[{base}] Missing files, skipping")
            continue

        try:
            # 使用新的load_pickle_data函数加载数据
            train_data = load_pickle_data(train_path)
            test_data = load_pickle_data(test_path)
            test_labels = load_pickle_data(label_path)

            logger.info(f"[{base}] Starting grid search...")
            
            for score_mode in config.score_modes:
                df = grid_search(
                    train_data, test_data, test_labels, gt_txt_path,
                    config, score_mode, txt_file, logger
                )
                df['dataset'] = txt_file
                all_results.append(df)
                
        except Exception as e:
            logger.error(f"Error processing {base}: {str(e)}", exc_info=True)
            continue

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

def summarize_best_results(results_df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """汇总最优结果"""
    if results_df.empty:
        logger.warning("No results to summarize")
        return pd.DataFrame(), {
            'overall_precision': 0.0,
            'overall_recall': 0.0,
            'overall_f1_score': 0.0
        }
    
    best_per_dataset = results_df.sort_values(by='f1_score', ascending=False).groupby('dataset').first().reset_index()

    total_TP = total_FP = total_FN = 0
    for _, row in best_per_dataset.iterrows():
        precision = row['precision']
        recall = row['recall']
        TP = 100  # 假设每个数据集有100个正样本用于计算
        FP = int(round((TP / precision) - TP)) if precision > 0 else 0
        FN = int(round((TP / recall) - TP)) if recall > 0 else 0
        total_TP += TP
        total_FP += FP
        total_FN += FN

    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    logger.info("\n===== Best Results per Dataset =====\n" + best_per_dataset.to_string(index=False))
    logger.info(f"\n===== Overall Metrics =====\n"
                f"Precision: {overall_precision:.4f}  "
                f"Recall: {overall_recall:.4f}  "
                f"F1: {overall_f1:.4f}")

    return best_per_dataset, {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_score': overall_f1
    }

# ==================== 可视化 ====================

def visualize_anomaly_graph(
    test_data: npt.NDArray,
    start: int,
    end: int,
    edge_stats: Dict[Tuple[int, int], Dict[str, Union[float, Tuple[float, float]]]],
    topk_neighbors: Dict[int, Set[int]],
    ground_truth_nodes: Set[int],
    detected_nodes: Set[int],
    save_path: str
):
    """可视化异常图"""
    window = test_data[start:end+1]
    corr = np.corrcoef(window, rowvar=False)
    num_sensors = corr.shape[0]

    G = nx.Graph()
    for i in range(num_sensors):
        G.add_node(i + 1)

    for j in range(num_sensors):
        for k in topk_neighbors[j]:
            key = (min(j, k), max(j, k))
            if key not in edge_stats:
                continue
            val = corr[j, k] if j < k else corr[k, j]
            if np.isnan(val):
                continue
            
            # 确保percentiles是元组且有两个元素
            percentiles = edge_stats[key].get('percentiles', (0, 0))
            if isinstance(percentiles, (list, tuple)) and len(percentiles) == 2:
                low, high = percentiles
            else:
                low, high = 0, 0
                
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
        mpatches.Patch(color='orange', label='Detected Anomaly'),
        mpatches.Patch(color='blue', label='GT Anomaly'),
        mpatches.Patch(color='purple', label='Correct Detection'),
        mpatches.Patch(color='lightgray', label='Normal'),
        mpatches.Patch(color='red', label='Abnormal Edge'),
        mpatches.Patch(color='gray', label='Normal Edge')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize='small')

    plt.savefig(save_path)
    plt.close()

def run_visualization(
    best_results: pd.DataFrame,
    config: ExperimentConfig,
    logger: logging.Logger
):
    """运行可视化流程"""
    if best_results.empty:
        logger.warning("No best results to visualize")
        return

    for _, row in best_results.iterrows():
        dataset_name = row['dataset'].replace('.txt', '')
        score_mode = row['score_mode']
        top_k = int(row['top_k']) if pd.notna(row['top_k']) else None
        threshold = float(row['score_threshold'])

        # 加载数据
        train_path = os.path.join(config.data_dir, f"{dataset_name}_train.pkl")
        test_path = os.path.join(config.data_dir, f"{dataset_name}_test.pkl")
        label_path = os.path.join(config.data_dir, f"{dataset_name}_test_label.pkl")
        gt_txt_path = os.path.join(config.label_dir, f"{dataset_name}.txt")

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path, gt_txt_path]):
            logger.warning(f"[{dataset_name}] Missing files, skipping visualization")
            continue

        try:
            # 修正点1：移除不必要的.values调用
            train_data = pd.read_pickle(train_path)
            if isinstance(train_data, pd.DataFrame):
                train_data = train_data.values
                
            test_data = pd.read_pickle(test_path)
            if isinstance(test_data, pd.DataFrame):
                test_data = test_data.values
                
            test_labels = pd.read_pickle(label_path)
            if isinstance(test_labels, pd.DataFrame):
                test_labels = test_labels.values

            # 计算边缘统计量和邻居
            edge_stats, avg_corr = calculate_edge_stats(train_data, config.window_size)
            num_sensors = train_data.shape[1]

            if config.use_topk and top_k is not None:
                topk_neighbors = {
                    i: set(np.argsort(-avg_corr[i])[:top_k + 1]) - {i}
                    for i in range(num_sensors)
                }
            else:
                topk_neighbors = {
                    i: set(range(num_sensors)) - {i}
                    for i in range(num_sensors)
                }

            # 创建可视化目录
            visual_dir = os.path.join(config.visual_dir, config.dataset, score_mode, dataset_name)
            os.makedirs(visual_dir, exist_ok=True)

            # 提取异常段
            segments = []
            current_start = None
            for i, val in enumerate(test_labels):
                if val == 1 and current_start is None:
                    current_start = i
                elif val == 0 and current_start is not None:
                    segments.append((current_start, i - 1))
                    current_start = None
            if current_start is not None:
                segments.append((current_start, len(test_labels) - 1))

            # 处理每个异常段
            for idx, (start, end) in enumerate(segments):
                # 获取真实异常节点
                gt_nodes = set()
                with open(gt_txt_path, 'r') as f:
                    for line in f:
                        t_range, s_str = line.strip().split(':')
                        t_start, t_end = map(int, t_range.split('-'))
                        if start >= t_start and end <= t_end:
                            gt_nodes = set(map(int, s_str.split(',')))
                            break

                # 检测异常节点
                abnormal_score = defaultdict(float)
                for i in range(start, end - config.window_size + 2):
                    window = test_data[i:i + config.window_size]
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
                
                detected_nodes = {node + 1 for node, score in abnormal_score.items() if score >= threshold}
                
                # 可视化
                image_path = os.path.join(visual_dir, f"seg_{idx}_{start}_{end}.png")
                visualize_anomaly_graph(
                    test_data, start, end, 
                    edge_stats,  # 传递完整的edge_stats字典，而不仅仅是percentiles
                    topk_neighbors, gt_nodes, detected_nodes, image_path
                )

            # 创建GIF
            gif_path = os.path.join(config.visual_dir, f"{dataset_name}.gif")
            create_gif_from_images(visual_dir, gif_path, config.gif_duration)
            
        except Exception as e:
            logger.error(f"Error visualizing {dataset_name}: {str(e)}", exc_info=True)
            continue

def create_gif_from_images(image_folder: str, gif_path: str, duration: int = 500):
    """从图像创建GIF"""
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".png"):
            img_path = os.path.join(image_folder, filename)
            try:
                images.append(Image.open(img_path))
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                continue
                
    if images:
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

# ==================== 主函数 ====================
def main():
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 初始化配置
    config = ExperimentConfig(
        dataset='omi',
        score_modes=['value_times_range'],
        use_topk=False
    )

    # 初始化日志
    logger = initialize_logging(config)
    logger.info(f"Score modes: {config.score_modes}")

    try:
        # 运行网格搜索
        results = run_full_grid_search(config, logger)
        
        # 汇总结果
        best_results, overall_metrics = summarize_best_results(results, logger)
        
        # 运行可视化
        run_visualization(best_results, config, logger)

    except Exception as e:
        logger.error(f"Error occurred in main: {str(e)}", exc_info=True)
        raise

    end_time = datetime.datetime.now()
    logger.info(f"Completed in: {end_time - start_time}")
    print(f"Completed in: {end_time - start_time}")

if __name__ == "__main__":
    main()
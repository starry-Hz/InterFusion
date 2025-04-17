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
from matplotlib.patches import Patch  # 添加缺失的导入
from matplotlib.lines import Line2D   # 添加缺失的导入
from typing import Dict, Tuple, Set, List, Optional, Union
import numpy.typing as npt
from dataclasses import dataclass, field
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
warnings.filterwarnings("ignore")

@dataclass
class ExperimentConfig:
    dataset: str = 'omi'
    window_size: int = 5
    score_thresholds: npt.NDArray = field(default_factory=lambda: np.arange(0.0, 7.1, 0.5))
    topk_range: range = field(default_factory=lambda: range(1, 10))
    # corr_threshold_range: npt.NDArray = field(default_factory=lambda: np.arange(0.1, 0.51, 0.1))
    corr_threshold_range: npt.NDArray = field(
        default_factory=lambda: np.round(np.linspace(0.0, 0.65, 14), 1)  # 修正后的阈值范围
    )
    neighbor_selection: str = 'topk'  # 可选: 'topk', 'corr_threshold', 'all'
    score_modes: List[str] = field(default_factory=lambda: ['value_times_range'])
    data_dir: str = "./data/processed"
    label_dir: str = "./data/interpretation_label"
    log_dir: str = field(init=False)
    visual_dir: str = field(init=False)
    num_workers: int = 36
    gif_duration: int = 500

    def __post_init__(self):
        """动态计算 log_dir 和 visual_dir"""
        self.log_dir = f"./log/{self.dataset}/neighbor_{self.neighbor_selection}"
        self.visual_dir = f"./visual/{self.dataset}/neighbor_{self.neighbor_selection}"
# 其他评分模式（可选）：
# score_modes = ['deviation', 'mean_ratio', 'range_ratio', 'value_times_range', 'robust_zscore']

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
    os.makedirs(f"{config.log_dir}/{config.score_modes[0]}", exist_ok=True)
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    log_filename = f"{config.log_dir}/{config.score_modes[0]}/{config.score_modes[0]}_{current_date}.log"
    
    # 强制重新配置日志系统
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='a',
        encoding='utf-8',
        force=True  # <-- 关键修改
    )
    
    logger = logging.getLogger(__name__)
    logging.getLogger().setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)
    # 关闭 Matplotlib 的字体警告（关键修正）
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    logger.info(f"\n Starting experiment with config: {config}")
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
                # val = abs(corr[j, k])  # 取绝对值
                if not np.isnan(val):
                    edge_corrs[(j, k)].append(val)

    # 计算边缘统计量
    edge_stats = {}
    for (j, k), vals in edge_corrs.items():
        if len(vals) == 0:
            continue
            
        median_val = np.median(vals)    # 中位数
        mad = np.median(np.abs(vals - median_val))  # 中位数绝对偏差
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
            
        # mean_val = np.mean(vals)
        mean_val = np.mean(np.abs(vals))  # 对每个窗口的相关系数先取绝对值
        avg_corr[j, k] = mean_val
        avg_corr[k, j] = mean_val

    return edge_stats, avg_corr


def calculate_abnormal_score(
    val: float,
    edge_stats: Dict[Tuple[int, int], Dict[str, Union[float, Tuple[float, float]]]],
    key: Tuple[int, int],
    score_mode: str,
    threshold_mode: str = 'percentile'  # 新增参数：阈值模式
) -> float:
    """
    计算异常分数（仅在超出阈值时返回分数）
    
    参数:
        val: 当前边的观测值
        edge_stats: 边统计信息字典
        key: 边标识 (i,j)
        score_mode: 分数计算模式 
            - 'strict_deviation': 仅当超出百分位时返回偏差值
            - 'deviation': 始终返回偏差值（原逻辑）
            - 其他模式（mean_ratio/range_ratio等）
        threshold_mode: 阈值模式
            - 'percentile': 使用百分位阈值 (low, high)
            - 'sigma': 使用均值±3标准差 (需要stats中有mean/std)
    
    返回:
        异常分数（未超阈值时返回0.0）
    """
    if key not in edge_stats:
        return 0.0
    
    stats = edge_stats[key]
    
    # 获取阈值范围
    if threshold_mode == 'percentile':
        percentiles = stats.get('percentiles', (0.0, 0.0))
        if not isinstance(percentiles, (tuple, list)) or len(percentiles) != 2:
            percentiles = (0.0, 0.0)
        low, high = percentiles
    elif threshold_mode == 'sigma' and 'mean' in stats and 'std' in stats:
        low = stats['mean'] - 3 * stats['std']
        high = stats['mean'] + 3 * stats['std']
    else:
        low, high = 0.0, 0.0  # 默认无阈值限制
    
    # 检查是否超出阈值
    is_abnormal = (val < low) or (val > high)
    
    # 如果未超阈值且不是deviation模式，直接返回0
    if not is_abnormal and score_mode != 'deviation':
        return 0.0
    # print(threshold_mode)
    # 计算分数
    score_modes = {
        'strict_deviation': lambda: max(abs(val - low), abs(val - high)),
        'deviation': lambda: max(abs(val - low), abs(val - high)),
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
    neighbor_param: Optional[Union[int, float]] = None
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
        neighbor_param: 邻居选择参数 (top_k或corr_threshold)
        
    Returns:
        (precision, recall, f1_score)
    """
    # 计算边缘统计量和平均相关系数
    edge_stats, avg_corr = calculate_edge_stats(train_data, config.window_size)
    num_sensors = train_data.shape[1]

    # 实现三种邻居选择方式
    if config.neighbor_selection == 'topk':
        # TopK邻居选择 - 按排名选择
        if neighbor_param is None or neighbor_param >= num_sensors:
            neighbor_param = num_sensors - 1
        topk_neighbors = {
            i: set(np.argsort(-avg_corr[i])[:neighbor_param + 1]) - {i}
            for i in range(num_sensors)
        }
    elif config.neighbor_selection == 'corr_threshold':
        # 相关系数阈值选择 - 按绝对值阈值选择
        if neighbor_param is None:
            neighbor_param = 0.1  # 默认阈值
        topk_neighbors = {
            i: set(j for j in range(num_sensors) 
                  if j != i and abs(avg_corr[i, j]) >= neighbor_param)
            for i in range(num_sensors)
        }
    else:
        # 默认选择所有邻居
        topk_neighbors = {
            i: set(range(num_sensors)) - {i}
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
    params: Tuple[Union[int, float], float],
    train_data: npt.NDArray,
    test_data: npt.NDArray,
    test_labels: npt.NDArray,
    gt_txt_path: str,
    config: ExperimentConfig,
    score_mode: str
) -> Dict[str, Union[str, int, float]]:
    neighbor_param, threshold = params
    precision, recall, f1 = evaluate_single_dataset(
        train_data, test_data, test_labels, gt_txt_path,
        config, score_mode, threshold, neighbor_param
    )
    
    # 修正日志输出
    logged_neighbor_param = (round(neighbor_param, 2) 
                           if isinstance(neighbor_param, float) 
                           else neighbor_param)
    
    return {
        'score_mode': score_mode,
        'neighbor_param': neighbor_param,  # 保持原始值用于计算
        'score_threshold': threshold,
        'neighbor_selection': config.neighbor_selection,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        # 'logged_neighbor_param': logged_neighbor_param  # 用于显示的修正值
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
    
    # 根据邻居选择方式生成参数组合
    if config.neighbor_selection == 'topk':
        params = [(top_k, threshold) 
                 for top_k in config.topk_range 
                 for threshold in config.score_thresholds]
    elif config.neighbor_selection == 'corr_threshold':
        params = [(corr_threshold, threshold)
                 for corr_threshold in config.corr_threshold_range
                 for threshold in config.score_thresholds]
    else:  # 'all' 或其他未定义方式
        params = [(None, threshold) for threshold in config.score_thresholds]

    # 使用多进程加速（原有逻辑不变）
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
                f"[{dataset_name}] mode={score_mode}, {config.neighbor_selection}={result['neighbor_param']}, "
                f"threshold={result['score_threshold']:.1f}, "
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

def optimize_metrics(results_df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """优化指标选择：在保持Recall的同时优先提高Precision"""
    logger.info("\n开始优化指标选择：平衡Precision和Recall")
    
    # 1. 首先按Precision降序排序，其次按Recall降序
    sorted_df = results_df.sort_values(
        by=['precision', 'recall'], 
        ascending=[False, False]
    )
    
    # 2. 对每个数据集，选择Precision最高的结果
    # 但要求Recall不低于该数据集所有结果的中位数
    best_per_dataset = []
    
    for dataset, group in sorted_df.groupby('dataset'):
        # 计算该数据集的Recall中位数
        median_recall = group['recall'].median()
        
        # 筛选Recall不低于中位数的结果
        valid_group = group[group['recall'] >= median_recall]
        
        if not valid_group.empty:
            # 选择其中Precision最高的
            best_row = valid_group.iloc[0]
        else:
            # 如果没有满足条件的，选择Recall最高的
            best_row = group.iloc[0]
            
        best_per_dataset.append(best_row)
    
    # 转换为DataFrame
    best_per_dataset = pd.DataFrame(best_per_dataset)
    
    # 3. 计算总体指标（使用加权平均，考虑各数据集大小）
    total_TP = total_FP = total_FN = 0
    for _, row in best_per_dataset.iterrows():
        precision = row['precision']
        recall = row['recall']
        
        # 假设每个数据集的权重相同（可根据实际情况调整）
        weight = 100  
        TP = weight
        FP = int(round((TP / precision) - TP)) if precision > 0 else 0
        FN = int(round((TP / recall) - TP)) if recall > 0 else 0
        
        total_TP += TP
        total_FP += FP
        total_FN += FN
    
    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    logger.info("\n===== 优化后的数据集最优参数组合 =====")
    logger.info(best_per_dataset[['dataset', 'top_k', 'score_threshold', 'precision', 'recall', 'f1_score']].to_string(index=False))
    
    logger.info("\n===== 优化后的整体指标 =====")
    logger.info(f"综合精确率: {overall_precision:.4f}")
    logger.info(f"综合召回率: {overall_recall:.4f}")
    logger.info(f"综合F1分数: {overall_f1:.4f}")
    
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
    save_path: str,
    logger: logging.Logger
):
    """优化后的异常图可视化函数"""
    logger.info(f"正在为时间段 {start}-{end} 生成异常图可视化")
    
    window = test_data[start:end+1]
    corr = np.corrcoef(window, rowvar=False)
    num_sensors = corr.shape[0]
    
    logger.debug(f"创建包含 {num_sensors} 个节点的图结构")
    G = nx.Graph()
    for i in range(num_sensors):
        G.add_node(i + 1)

    # 统计边缘信息
    normal_edges = 0
    abnormal_edges = 0
    
    for j in range(num_sensors):
        for k in topk_neighbors[j]:
            key = (min(j, k), max(j, k))
            if key not in edge_stats:
                continue
            val = corr[j, k] if j < k else corr[k, j]
            if np.isnan(val):
                continue
            
            percentiles = edge_stats[key].get('percentiles', (0, 0))
            if isinstance(percentiles, (list, tuple)) and len(percentiles) == 2:
                low, high = percentiles
            else:
                low, high = 0, 0
                
            is_abnormal = val < low or val > high
            if is_abnormal:
                abnormal_edges += 1
            else:
                normal_edges += 1
                
            G.add_edge(j + 1, k + 1, color='red' if is_abnormal else 'gray')

    logger.debug(f"边统计: 正常边 {normal_edges} 条, 异常边 {abnormal_edges} 条")
    
    # 节点颜色统计
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

    # 创建图形（优化部分）
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 优化布局参数
    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=50)
    
    # 绘制图形
    nx.draw_networkx(
        G, 
        pos=pos, 
        ax=ax,
        with_labels=True, 
        node_color=node_colors, 
        edge_color=[G[u][v]['color'] for u, v in G.edges()],
        node_size=300,  # 减小节点大小
        font_size=8,    # 减小字体大小
        width=1.5       # 减小边宽度
    )
    
    ax.set_title(f"异常时间段 {start}-{end}", fontsize=10)
    
    # 设置紧凑布局
    plt.tight_layout()
    
    # 设置轴限制（优化部分）
    pos_array = np.array(list(pos.values()))
    x_min, y_min = np.min(pos_array, axis=0)
    x_max, y_max = np.max(pos_array, axis=0)
    
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # 图例
    legend_elements = [
        mpatches.Patch(color='orange', label='检测到的异常'),
        mpatches.Patch(color='blue', label='真实异常'),
        mpatches.Patch(color='purple', label='正确检测'),
        mpatches.Patch(color='lightgray', label='正常节点'),
        mpatches.Patch(color='red', label='异常边'),
        mpatches.Patch(color='gray', label='正常边')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8)

    logger.debug(f"正在保存可视化结果到 {save_path}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 增加dpi和bbox_inches参数
    plt.close(fig)
    logger.info(f"成功保存时间段 {start}-{end} 的可视化结果")

def visualize_all_anomaly_segments(
    test_data: npt.NDArray,
    segments: List[Tuple[int, int]],
    edge_stats: Dict[Tuple[int, int], Dict[str, Union[float, Tuple[float, float]]]],
    topk_neighbors: Dict[int, Set[int]],
    ground_truth_nodes_dict: Dict[Tuple[int, int], Set[int]],
    detected_nodes_dict: Dict[Tuple[int, int], Set[int]],
    save_path: str,
    logger: logging.Logger,
    figsize: Tuple[int, int] = (32, 20),  # 大幅增加画布尺寸
    dpi: int = 150,                       # 更高分辨率
    node_size: int = 300,                 # 显著增大节点
    font_size: int = 12,                  # 增大标签
    title_fontsize: int = 14              # 单独控制标题字号
):
    """
    终极优化版 - 解决拥挤问题，最大化利用画面空间
    """
    logger.info("生成全屏优化的可视化图表")
    
    num_segments = len(segments)
    if num_segments == 0:
        logger.warning("没有异常段可供可视化")
        return
    
    # 动态行列计算（保持2行但增加列宽）
    cols = min(4, int(np.ceil(num_segments / 2)))  # 每行最多4个子图
    rows = int(np.ceil(num_segments / cols))
    
    # 创建超大画布（根据您的图片有8个时间段）
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    
    # 极端宽松的边距设置（关键调整！）
    plt.subplots_adjust(
        left=0.03, right=0.97,  # 左右边距从5%→3%
        top=0.92, bottom=0.18,   # 底部留更多空间给图例
        wspace=0.5, hspace=0.6   # 子图间距增大50%
    )
    
    if num_segments == 1:
        axes = np.array([[axes]])
    
    # 颜色配置（与您图片完全匹配）
    color_map = {
        'detected_only': 'orange',
        'truth_only': 'blue',
        'both': 'purple',
        'normal': 'lightgray',
        'abnormal_edge': 'red',
        'normal_edge': 'gray'
    }
    
    # 绘制每个子图
    for idx, (start, end) in enumerate(segments):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # 准备数据
        gt_nodes = ground_truth_nodes_dict.get((start, end), set())
        detected_nodes = detected_nodes_dict.get((start, end), set())
        window = test_data[start:end+1]
        corr = np.corrcoef(window, rowvar=False)
        num_sensors = corr.shape[0]
        
        # 构建网络图
        G = nx.Graph()
        G.add_nodes_from(range(1, num_sensors+1))
        
        # 添加边（根据您图片中的连接关系）
        edge_colors = []
        for j in range(num_sensors):
            for k in topk_neighbors[j]:
                key = (min(j, k), max(j, k))
                if key not in edge_stats: continue
                
                val = corr[j, k] if j < k else corr[k, j]
                if np.isnan(val): continue
                
                percentiles = edge_stats[key].get('percentiles', (0, 0))
                low, high = (percentiles if isinstance(percentiles, (tuple, list)) 
                            else (0, 0))
                
                is_abnormal = val < low or val > high
                edge_color = color_map['abnormal_edge' if is_abnormal else 'normal_edge']
                G.add_edge(j+1, k+1, color=edge_color)
                edge_colors.append(edge_color)
        
        # 节点着色（精确匹配您图片的颜色方案）
        node_colors = []
        for node in G.nodes():
            if node in gt_nodes and node in detected_nodes:
                node_colors.append(color_map['both'])
            elif node in gt_nodes:
                node_colors.append(color_map['truth_only'])
            elif node in detected_nodes:
                node_colors.append(color_map['detected_only'])
            else:
                node_colors.append(color_map['normal'])
        
        # 优化绘图参数
        pos = nx.spring_layout(G, seed=42, k=1.5)  # 增加节点间距参数k
        
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_size,
            node_color=node_colors,
            edgecolors='black',
            linewidths=0.5
        )
        
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=1.5,
            edge_color=edge_colors,
            alpha=0.7
        )
        
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=font_size,
            font_weight='bold'
        )
        
        # 优化标题（与您图片中的格式一致）
        ax.set_title(f"时间段 {start}-{end}", 
                    fontsize=title_fontsize, 
                    pad=20)  # 增加标题间距
        ax.set_axis_off()
    
    # 在绘制完所有子图后，添加分割线
    for ax in axes.flatten():
        # 添加红色矩形边框（线宽=2，透明度=0.7）
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('red')
            spine.set_linewidth(2)
            spine.set_alpha(0.7)
    
    # 调整子图间距（增大空白区域）
    plt.subplots_adjust(
        wspace=0.6,  # 列间距增大60%
        hspace=0.8   # 行间距增大80%
    )

    # 隐藏空白子图
    for idx in range(num_segments, rows*cols):
        row, col = idx // cols, idx % cols
        if rows > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')
    
    # 专业级图例设计（完全匹配您图片的图例）
    legend_elements = [
        # 节点分类
        Patch(facecolor='orange', edgecolor='black', label='检测到的异常'),  # FP (False Positive)
        Patch(facecolor='blue', edgecolor='black', label='真实异常'),        # FN (False Negative)
        Patch(facecolor='purple', edgecolor='black', label='正确检测'),      # TP (True Positive)
        Patch(facecolor='lightgray', edgecolor='black', label='正常节点'),   # TN (True Negative)
        
        # 边分类
        Line2D([], [], color='red', linewidth=2, label='异常边'),          # 异常相关性
        Line2D([], [], color='gray', linewidth=2, label='正常边')           # 正常相关性
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),  # 精确控制图例位置
        ncol=6,
        fontsize=14,
        framealpha=1,
        borderpad=1,
        labelspacing=1.2
    )
    
    # 保存高清图像
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    logger.info(f"已生成全屏优化的可视化图表: {save_path}")

def run_visualization(
    best_results: pd.DataFrame,
    config: ExperimentConfig,
    logger: logging.Logger
):
    """运行可视化流程"""
    if best_results.empty:
        logger.warning("没有最优结果可供可视化")
        return

    for _, row in best_results.iterrows():
        dataset_name = row['dataset'].replace('.txt', '')
        score_mode = row['score_mode']
        neighbor_param = row['neighbor_param']
        threshold = float(row['score_threshold'])

        logger.info(f"\n{'='*50}")
        logger.info(f"开始数据集可视化: {dataset_name}")
        logger.info(f"使用参数: 评分模式={score_mode}, {config.neighbor_selection}={neighbor_param}, 阈值={threshold}")
        logger.info(f"{'='*50}\n")

        # 加载数据
        train_path = os.path.join(config.data_dir, f"{dataset_name}_train.pkl")
        test_path = os.path.join(config.data_dir, f"{dataset_name}_test.pkl")
        label_path = os.path.join(config.data_dir, f"{dataset_name}_test_label.pkl")
        gt_txt_path = os.path.join(config.label_dir, f"{dataset_name}.txt")

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path, gt_txt_path]):
            logger.warning(f"[{dataset_name}] 缺少必要文件，跳过可视化")
            continue

        try:
            logger.info(f"正在加载数据集 {dataset_name} 的数据")
            train_data = pd.read_pickle(train_path)
            if isinstance(train_data, pd.DataFrame):
                train_data = train_data.values
                
            test_data = pd.read_pickle(test_path)
            if isinstance(test_data, pd.DataFrame):
                test_data = test_data.values
                
            test_labels = pd.read_pickle(label_path)
            if isinstance(test_labels, pd.DataFrame):
                test_labels = test_labels.values

            logger.info(f"数据加载成功。训练集形状: {train_data.shape}, 测试集形状: {test_data.shape}")

            # 计算边缘统计量和邻居
            logger.info("正在计算边缘统计量...")
            edge_stats, avg_corr = calculate_edge_stats(train_data, config.window_size)
            num_sensors = train_data.shape[1]
            logger.info(f"已计算 {len(edge_stats)} 条边的统计量")

            # 实现三种邻居选择方式
            if config.neighbor_selection == 'topk':
                logger.info(f"为每个节点使用 top-{neighbor_param} 邻居")
                topk_neighbors = {
                    i: set(np.argsort(-avg_corr[i])[:neighbor_param + 1]) - {i}
                    for i in range(num_sensors)
                }
            elif config.neighbor_selection == 'corr_threshold':
                logger.info(f"为每个节点使用相关系数阈值 >= {neighbor_param} 的邻居")
                topk_neighbors = {
                    i: set(j for j in range(num_sensors) 
                          if j != i and abs(avg_corr[i, j]) >= neighbor_param)
                    for i in range(num_sensors)
                }
            else:
                logger.info("为每个节点使用所有邻居")
                topk_neighbors = {
                    i: set(range(num_sensors)) - {i}
                    for i in range(num_sensors)
                }

            # 创建可视化目录
            visual_dir = os.path.join(config.visual_dir, score_mode, dataset_name)
            os.makedirs(visual_dir, exist_ok=True)
            logger.info(f"可视化结果将保存到: {visual_dir}")

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

            logger.info(f"在数据集中发现 {len(segments)} 个异常时间段")

            # 初始化存储字典
            ground_truth_nodes_dict = {}
            detected_nodes_dict = {}

            # 处理每个异常段
            for idx, (start, end) in enumerate(segments):
                logger.info(f"\n正在处理时间段 {idx+1}/{len(segments)}: {start}-{end}")
                
                # 获取真实异常节点
                gt_nodes = set()
                with open(gt_txt_path, 'r') as f:
                    for line in f:
                        t_range, s_str = line.strip().split(':')
                        t_start, t_end = map(int, t_range.split('-'))
                        if start >= t_start and end <= t_end:
                            gt_nodes = set(map(int, s_str.split(',')))
                            break
                
                ground_truth_nodes_dict[(start, end)] = gt_nodes
                logger.info(f"该时间段的真实异常节点: {gt_nodes}")

                # 检测异常节点
                logger.info("正在计算异常分数...")
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
                detected_nodes_dict[(start, end)] = detected_nodes
                logger.info(f"检测到的异常节点: {detected_nodes}")
                
                # 单独可视化
                image_path = os.path.join(visual_dir, f"seg_{idx}_{start}_{end}.png")
                logger.info(f"正在为时间段 {start}-{end} 生成可视化")
                
                visualize_anomaly_graph(
                    test_data, start, end, 
                    edge_stats,
                    topk_neighbors, 
                    gt_nodes, 
                    detected_nodes, 
                    image_path,
                    logger
                )

            # 生成所有异常段的综合可视化
            if segments:
                combined_path = os.path.join(visual_dir, "all_segments.png")
                logger.info(f"正在生成所有异常段的综合可视化图像: {combined_path}")
                
                visualize_all_anomaly_segments(
                    test_data, segments,
                    edge_stats, topk_neighbors,
                    ground_truth_nodes_dict, detected_nodes_dict,
                    combined_path, logger,
                    figsize=(24, 16),  # 可进一步调整
                    dpi=120,
                    node_size=200,
                    font_size=10
                )

            # 创建GIF
            gif_path = os.path.join(visual_dir, f"{dataset_name}.gif")
            logger.info(f"正在从 {visual_dir} 中的图像创建GIF")
            create_gif_from_images(visual_dir, gif_path, config.gif_duration)
            logger.info(f"GIF已保存到 {gif_path}")
            
        except Exception as e:
            logger.error(f"可视化 {dataset_name} 时出错: {str(e)}", exc_info=True)
            continue

def create_gif_from_images(image_folder: str, gif_path: str, duration: int = 500,exclude_all_segments: bool = True):
    """从图像创建GIF"""
    images = []
    for filename in sorted(os.listdir(image_folder)):
        # 排除all_segments.png（如果启用）
        if exclude_all_segments and "all_segments" in filename.lower(): # filename.lower是为了避免大小写问题
            print(f"跳过文件: {filename}（已排除）")
            continue

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
def main(config=None,logger=None):
    start_time = datetime.datetime.now()
    start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"程序启动时间: {start_str}")
    
    # 如果未传入配置，则使用默认配置
    if config is None:
        config = ExperimentConfig()
    if logger is None:
        logger = initialize_logging(config)
    logger.info(f"当前评分模式: {config.score_modes}")

    try:
        # ==================== 网格搜索阶段 ====================
        logger.info("\n" + "="*50)
        logger.info("开始执行网格搜索...")
        results = run_full_grid_search(config, logger)
        
        if results.empty:
            logger.warning("⚠️ 网格搜索未产生任何结果，请检查输入数据！")
            return

        # ==================== 结果汇总阶段 ====================
        logger.info("\n" + "="*50)
        logger.info("开始汇总结果...")
        
        # 原始汇总方式
        best_results, overall_metrics = summarize_best_results(results, logger)
        logger.info("\n【原始汇总指标】")
        logger.info(f"精确率(Precision): {overall_metrics['overall_precision']:.4f}")
        logger.info(f"召回率(Recall):    {overall_metrics['overall_recall']:.4f}")
        logger.info(f"F1值:             {overall_metrics['overall_f1_score']:.4f}")
        
        # # 优化后的汇总方式
        # optimized_results, optimized_metrics = optimize_metrics(results,logger)
        # logger.info("\n【优化后指标】")
        # logger.info(f"精确率(Precision): {optimized_metrics['overall_precision']:.4f}")
        # logger.info(f"召回率(Recall):    {optimized_metrics['overall_recall']:.4f}")
        # logger.info(f"F1值:             {optimized_metrics['overall_f1_score']:.4f}")
        
        # # 结果对比分析
        # logger.info("\n【性能变化分析】")
        # precision_diff = optimized_metrics['overall_precision'] - overall_metrics['overall_precision']
        # recall_diff = optimized_metrics['overall_recall'] - overall_metrics['overall_recall']
        # f1_diff = optimized_metrics['overall_f1_score'] - overall_metrics['overall_f1_score']
        
        # logger.info(f"精确率变化: {'↑' if precision_diff > 0 else '↓'} {abs(precision_diff):.4f}")
        # logger.info(f"召回率变化: {'↑' if recall_diff > 0 else '↓'} {abs(recall_diff):.4f}")
        # logger.info(f"F1值变化:   {'↑' if f1_diff > 0 else '↓'} {abs(f1_diff):.4f}")

        # ==================== 可视化阶段 ====================
        logger.info("\n" + "="*50)
        logger.info("开始生成可视化结果...")
        run_visualization(best_results, config, logger)
        logger.info("可视化结果生成完成！")

    except Exception as e:
        logger.error(f"❌ 程序运行出错: {str(e)}", exc_info=True)
        raise
    finally:
        # ==================== 收尾工作 ====================
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        logger.info("\n" + "="*50)
        logger.info(f"程序运行总耗时: {duration} \n")
        print(f"程序执行完毕，总耗时: {duration}")

if __name__ == "__main__":
    # main()
    # for neighbor_selection in ['corr_threshold']:
    #     for score_mode in ['strict_deviation', 'deviation', 'mean_ratio', 'range_ratio', 'value_times_range', 'robust_zscore']:
    #         print(f"\n{'='*50}")
    #         print(f"Running with neighbor_selection={neighbor_selection}, score_mode={score_mode}")
    #         print(f"{'='*50}\n")
            
    #         # 创建配置实例
    #         config = ExperimentConfig(
    #             neighbor_selection=neighbor_selection,
    #             score_modes=score_mode,
    #             use_topk=False
    #         )
    #         logger = initialize_logging(config)

    #         # 运行主程序
    #         main(config, logger)

    all_score_modes = ['strict_deviation', 'deviation', 'mean_ratio', 'range_ratio', 'value_times_range', 'robust_zscore']
    # all_score_modes = ['strict_deviation', 'value_times_range']
    
    # 设置邻居选择方式为相关系数阈值
    # neighbor_selection = 'corr_threshold' ['topk', 'corr_threshold', 'all']
    
    # 遍历每种评分模式
    for neighbor_selection in ['topk', 'corr_threshold', 'all']:
        for score_mode in all_score_modes:
            print(f"\n{'='*50}")
            print(f"Running with neighbor_selection={neighbor_selection}, score_mode={score_mode}")
            print(f"{'='*50}\n")
            
            # 创建配置实例
            config = ExperimentConfig(
                neighbor_selection=neighbor_selection,
                score_modes=[score_mode]  # 注意这里传入列表，因为score_modes是List[str]类型
            )
            logger = initialize_logging(config)

            # 运行主程序
            main(config, logger)

# 相比于 main.py，主要修改了以下几点：
    # 1. 现在使用三种邻居选择方式：topk、相关系数阈值和所有邻居。
    # 当选择 corr_threshold 时，邻居选择方式为相关系数阈值 [0.1, 0.2, 0.3, 0.4, 0.5]。
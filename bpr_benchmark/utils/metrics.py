"""
评估指标计算模块

职责：
1. 计算基本评估指标 (MAE, RMSE, MAPE, R²)
2. 计算分层评估指标（按V/C比分组）
3. 生成评估报告

对应报告 Section 4.5 和 5.2
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    MAE = (1/n) * Σ|y_true - y_pred|
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        MAE值
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根误差 (Root Mean Square Error)
    
    RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        RMSE值
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    计算平均绝对百分比误差 (Mean Absolute Percentage Error)
    
    MAPE = (100/n) * Σ|y_true - y_pred| / y_true
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        epsilon: 避免除零的小常数
        
    Returns:
        MAPE值 (百分比)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 避免除零
    mask = np.abs(y_true) > epsilon
    
    if not np.any(mask):
        return np.nan
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算决定系数 (R-squared)
    
    R² = 1 - (SS_res / SS_tot)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        R²值
    """
    return r2_score(y_true, y_pred)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算所有基本评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        包含所有指标的字典
    """
    
    metrics = {
        'MAE': calculate_mae(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred)
    }
    
    return metrics


def calculate_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vcr: np.ndarray,
    vcr_bins: List[Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    计算分层评估指标（按V/C比分组）
    
    对应报告中的分层分析方法
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        vcr: V/C比值
        vcr_bins: V/C比分组区间，可以是：
                  - 列表of列表/元组: [(0, 0.3), (0.3, 0.7), ...]
                  - 边界列表: [0, 0.3, 0.7, 1.0, 999] (将被转换为区间)
        
    Returns:
        包含各层指标的DataFrame
    """
    
    if vcr_bins is None:
        vcr_bins = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0), (1.0, 999)]
    elif isinstance(vcr_bins, list) and len(vcr_bins) > 0:
        # 检查是边界列表还是区间列表
        if not isinstance(vcr_bins[0], (tuple, list)):
            # 边界列表 [0, 0.3, 0.7, ...] -> 转换为区间列表
            vcr_bins = [(vcr_bins[i], vcr_bins[i+1]) for i in range(len(vcr_bins)-1)]
    
    results = []
    
    for lower, upper in vcr_bins:
        # 筛选该层的数据
        mask = (vcr >= lower) & (vcr < upper)
        
        if not np.any(mask):
            continue
        
        y_true_layer = y_true[mask]
        y_pred_layer = y_pred[mask]
        
        # 计算该层的指标
        metrics = calculate_all_metrics(y_true_layer, y_pred_layer)
        
        # 添加层信息
        if upper >= 999:
            layer_name = f"V/C ≥ {lower}"
        else:
            layer_name = f"{lower} ≤ V/C < {upper}"
        
        metrics['Layer'] = layer_name
        metrics['Count'] = int(np.sum(mask))
        metrics['VCR_Mean'] = float(np.mean(vcr[mask]))
        
        results.append(metrics)
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 重新排列列顺序
    cols = ['Layer', 'Count', 'VCR_Mean', 'MAE', 'RMSE', 'MAPE', 'R2']
    df_results = df_results[cols]
    
    return df_results


def calculate_metrics_by_time_period(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_peak: np.ndarray
) -> pd.DataFrame:
    """
    按时段计算评估指标（高峰 vs 非高峰）
    
    对应 M1 (动态参数) 的评估
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        is_peak: 是否高峰时段标志
        
    Returns:
        包含各时段指标的DataFrame
    """
    
    results = []
    
    for period_name, period_mask in [('高峰时段', is_peak == 1), ('非高峰时段', is_peak == 0)]:
        if not np.any(period_mask):
            continue
        
        y_true_period = y_true[period_mask]
        y_pred_period = y_pred[period_mask]
        
        metrics = calculate_all_metrics(y_true_period, y_pred_period)
        metrics['Period'] = period_name
        metrics['Count'] = int(np.sum(period_mask))
        
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    
    # 重新排列列顺序
    cols = ['Period', 'Count', 'MAE', 'RMSE', 'MAPE', 'R2']
    df_results = df_results[cols]
    
    return df_results


def print_metrics_summary(metrics: Dict[str, float], model_name: str = "Model"):
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
        model_name: 模型名称
    """
    
    print(f"\n{model_name} 评估结果:")
    print("-" * 50)
    print(f"  MAE:   {metrics['MAE']:.4f} 秒")
    print(f"  RMSE:  {metrics['RMSE']:.4f} 秒")
    print(f"  MAPE:  {metrics['MAPE']:.2f} %")
    print(f"  R²:    {metrics['R2']:.4f}")
    print("-" * 50)


def create_metrics_comparison_table(
    results_dict: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    创建模型对比表
    
    Args:
        results_dict: 字典，键为模型名称，值为指标字典
        
    Returns:
        对比表DataFrame
    """
    
    df = pd.DataFrame(results_dict).T
    
    # 按MAE排序
    df = df.sort_values('MAE')
    
    # 重置索引，使模型名称成为一列
    df = df.reset_index()
    df = df.rename(columns={'index': 'Model'})
    
    return df


def calculate_improvement(baseline_metrics: Dict[str, float], 
                         model_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    计算相对于基线模型的改进百分比
    
    Args:
        baseline_metrics: 基线模型的指标
        model_metrics: 当前模型的指标
        
    Returns:
        改进百分比字典
    """
    
    improvements = {}
    
    for metric in ['MAE', 'RMSE', 'MAPE']:
        baseline_val = baseline_metrics[metric]
        model_val = model_metrics[metric]
        
        # 对于误差指标，负值表示改进
        improvement = ((baseline_val - model_val) / baseline_val) * 100
        improvements[f'{metric}_improvement'] = improvement
    
    # R²的改进计算不同
    improvements['R2_improvement'] = model_metrics['R2'] - baseline_metrics['R2']
    
    return improvements


def generate_evaluation_report(
    results_dict: Dict[str, Dict],
    output_path: str = None
) -> str:
    """
    生成完整的评估报告
    
    Args:
        results_dict: 包含所有模型结果的字典
        output_path: 输出文件路径（可选）
        
    Returns:
        报告文本
    """
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BPR 基准测试评估报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. 总体对比表
    report_lines.append("1. 模型性能对比")
    report_lines.append("-" * 80)
    
    # 只使用成功的结果
    success_results = {
        name: res['overall'] 
        for name, res in results_dict.items() 
        if 'overall' in res
    }
    
    if not success_results:
        report_lines.append("没有成功的模型结果")
        report_lines.append("")
        return "\n".join(report_lines)
    
    comparison_df = create_metrics_comparison_table(success_results)
    report_lines.append(comparison_df.to_string(index=False))
    report_lines.append("")
    
    # 2. 最佳模型
    best_model = comparison_df.iloc[0]['Model']
    best_mae = comparison_df.iloc[0]['MAE']
    report_lines.append(f"最佳模型: {best_model} (MAE = {best_mae:.4f} 秒)")
    report_lines.append("")
    
    # 3. 相对于基线的改进
    if 'ClassicalBPR' in results_dict:
        report_lines.append("2. 相对于经典BPR的改进")
        report_lines.append("-" * 80)
        
        baseline = results_dict['ClassicalBPR']['overall']
        
        for model_name, model_res in results_dict.items():
            if model_name == 'ClassicalBPR':
                continue
            
            improvements = calculate_improvement(baseline, model_res['overall'])
            
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"  MAE 改进:  {improvements['MAE_improvement']:+.2f}%")
            report_lines.append(f"  RMSE 改进: {improvements['RMSE_improvement']:+.2f}%")
            report_lines.append(f"  MAPE 改进: {improvements['MAPE_improvement']:+.2f}%")
            report_lines.append(f"  R² 变化:   {improvements['R2_improvement']:+.4f}")
        
        report_lines.append("")
    
    # 4. 分层分析（如果有）
    if any('stratified' in res for res in results_dict.values()):
        report_lines.append("3. 分层分析（按V/C比）")
        report_lines.append("-" * 80)
        
        for model_name, model_res in results_dict.items():
            if 'stratified' in model_res:
                report_lines.append(f"\n{model_name}:")
                report_lines.append(model_res['stratified'].to_string(index=False))
                report_lines.append("")
    
    report_lines.append("=" * 80)
    
    # 生成报告文本
    report_text = "\n".join(report_lines)
    
    # 保存到文件（如果指定了路径）
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n评估报告已保存到: {output_path}")
    
    return report_text


if __name__ == "__main__":
    """测试指标计算"""
    
    # 生成测试数据
    np.random.seed(42)
    n = 1000
    
    y_true = np.random.uniform(100, 300, n)
    y_pred = y_true + np.random.normal(0, 20, n)
    vcr = np.random.uniform(0, 1.5, n)
    is_peak = np.random.choice([0, 1], n)
    
    # 测试基本指标
    print("测试基本指标计算:")
    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics_summary(metrics, "测试模型")
    
    # 测试分层指标
    print("\n测试分层指标计算:")
    stratified = calculate_stratified_metrics(y_true, y_pred, vcr)
    print(stratified)
    
    # 测试时段指标
    print("\n测试时段指标计算:")
    by_period = calculate_metrics_by_time_period(y_true, y_pred, is_peak)
    print(by_period)


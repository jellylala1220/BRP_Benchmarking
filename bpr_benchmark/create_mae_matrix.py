"""
创建完整的MAE矩阵表（模型 × 方法）
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))

def create_mae_matrix_from_results(results: dict) -> pd.DataFrame:
    """
    从结果字典创建MAE矩阵
    
    Args:
        results: 格式为 {'M0_nls': {...}, 'M1_classical': {...}, ...}
        
    Returns:
        MAE矩阵DataFrame，行=模型，列=方法
    """
    
    # 收集所有模型和方法
    all_models = set()
    all_methods = set()
    
    for key, res in results.items():
        if not res.get('success', False):
            continue
        parts = key.rsplit('_', 1)  # 从右侧分割一次
        if len(parts) == 2:
            model, method = parts
            all_models.add(model)
            all_methods.add(method)
    
    # 排序
    all_models = sorted(all_models)
    all_methods = sorted(all_methods)
    
    # 创建矩阵
    matrix_data = []
    
    for model in all_models:
        row = {'Model': model}
        for method in all_methods:
            key = f"{model}_{method}"
            if key in results and results[key].get('success', False):
                mae = results[key]['overall']['MAE']
                row[method] = round(mae, 2)
            else:
                row[method] = np.nan
        matrix_data.append(row)
    
    df = pd.DataFrame(matrix_data)
    
    # 设置Model列为索引
    df = df.set_index('Model')
    
    return df


def create_all_metrics_matrices(results: dict) -> dict:
    """
    创建所有指标的矩阵（MAE, RMSE, MAPE, R²）
    
    Returns:
        包含4个矩阵的字典
    """
    
    # 收集所有模型和方法
    all_models = set()
    all_methods = set()
    
    for key, res in results.items():
        if not res.get('success', False):
            continue
        parts = key.rsplit('_', 1)
        if len(parts) == 2:
            model, method = parts
            all_models.add(model)
            all_methods.add(method)
    
    all_models = sorted(all_models)
    all_methods = sorted(all_methods)
    
    # 创建4个指标的矩阵
    metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
    matrices = {}
    
    for metric in metrics:
        matrix_data = []
        
        for model in all_models:
            row = {'Model': model}
            for method in all_methods:
                key = f"{model}_{method}"
                if key in results and results[key].get('success', False):
                    value = results[key]['overall'][metric]
                    if metric == 'MAPE':
                        row[method] = f"{value:.2f}%"
                    elif metric == 'R2':
                        row[method] = f"{value:.4f}"
                    else:
                        row[method] = round(value, 2)
                else:
                    row[method] = '-'
            matrix_data.append(row)
        
        df = pd.DataFrame(matrix_data)
        df = df.set_index('Model')
        matrices[metric] = df
    
    return matrices


def print_mae_matrix(df: pd.DataFrame):
    """
    打印格式化的MAE矩阵
    """
    print("\n" + "="*80)
    print("MAE矩阵（秒）- 模型 × 方法")
    print("="*80)
    print(df.to_string())
    print("="*80)
    print("\n✓ 值越小越好")
    
    # 找出每一列的最佳值
    print("\n每种方法的最佳模型:")
    for col in df.columns:
        valid_values = df[col].dropna()
        if len(valid_values) > 0:
            best_model = valid_values.idxmin()
            best_value = valid_values.min()
            print(f"  {col:12s}: {best_model:4s} (MAE={best_value:.2f}秒)")
    
    # 找出每一行的最佳方法
    print("\n每个模型的最佳方法:")
    for idx in df.index:
        valid_values = df.loc[idx].dropna()
        if len(valid_values) > 0:
            best_method = valid_values.idxmin()
            best_value = valid_values.min()
            print(f"  {idx:4s}: {best_method:12s} (MAE={best_value:.2f}秒)")


if __name__ == "__main__":
    # 这个脚本通常会被run_benchmark.py调用
    # 但也可以独立运行来分析已有结果
    
    print("这个脚本用于创建MAE矩阵")
    print("请运行 run_benchmark.py 来生成完整结果")


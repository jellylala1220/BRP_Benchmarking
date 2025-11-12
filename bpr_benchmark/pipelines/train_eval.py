"""
训练和评估流程

职责：
- 读取FinalData
- 按时间切分训练/测试集
- 循环训练所有模型×方法组合
- 生成"行=模型、列=方法"的MAE表
- 输出RMSE/MAPE/R²表和分层表
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.registry import get_available_models, get_available_estimators, get_compatible_methods, create_model
from utils.metrics import calculate_all_metrics, calculate_stratified_metrics


def split_by_time(df: pd.DataFrame, train_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间分割训练/测试集
    
    Args:
        df: FinalData DataFrame
        train_end: 训练集结束日期 (如 "2024-09-20")
        
    Returns:
        (df_train, df_test)
    """
    
    df_train = df[df['datetime'] <= train_end].copy()
    df_test = df[df['datetime'] > train_end].copy()
    
    print(f"\n数据分割:")
    print(f"  训练集: {len(df_train)} 条 (至 {train_end})")
    print(f"  测试集: {len(df_test)} 条 (从 {train_end} 之后)")
    
    return df_train, df_test


def train_and_evaluate_single(
    model_name: str,
    method: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    config: Dict = None
) -> Dict:
    """
    训练和评估单个模型×方法组合
    
    Args:
        model_name: 模型名称 (如 'M0')
        method: 估计方法 (如 'nls')
        df_train: 训练数据
        df_test: 测试数据
        config: 配置字典
        
    Returns:
        结果字典，包含metrics和predictions
    """
    
    try:
        # 创建模型
        model = create_model(model_name, config)
        
        # 训练
        model.fit(df_train, method=method)
        
        # 预测
        y_pred = model.predict(df_test)
        y_true = df_test['fused_tt_15min'].values
        
        # 评估
        metrics = calculate_all_metrics(y_true, y_pred)
        
        return {
            'success': True,
            'metrics': metrics,
            'predictions': y_pred,
            'model_info': model.info()
        }
        
    except Exception as e:
        print(f"    ✗ 失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def run_benchmark(
    df: pd.DataFrame,
    models_to_run: List[str] = None,
    methods_to_run: List[str] = None,
    train_end: str = None,
    config: Dict = None,
    output_dir: str = "outputs/benchmark"
) -> Dict:
    """
    运行完整的基准测试
    
    Args:
        df: FinalData DataFrame
        models_to_run: 要测试的模型列表（如果为None则测试所有）
        methods_to_run: 要测试的方法列表（如果为None则测试所有）
        train_end: 训练集结束日期
        config: 配置字典
        output_dir: 输出目录
        
    Returns:
        完整的结果字典
    """
    
    print("\n" + "="*60)
    print("BPR基准测试")
    print("="*60)
    
    # 确定要测试的模型和方法
    if models_to_run is None:
        models_to_run = get_available_models()
    
    if methods_to_run is None:
        methods_to_run = get_available_estimators()
    
    print(f"\n测试配置:")
    print(f"  模型: {models_to_run}")
    print(f"  方法: {methods_to_run}")
    
    # 分割数据
    if train_end is None:
        # 默认使用80%作为训练集
        train_end = df['datetime'].quantile(0.8)
    
    df_train, df_test = split_by_time(df, train_end)
    
    # 过滤有效数据
    if 'is_valid' in df_train.columns:
        df_train = df_train[df_train['is_valid'] == 1]
        df_test = df_test[df_test['is_valid'] == 1]
        print(f"  过滤后: 训练集 {len(df_train)} 条, 测试集 {len(df_test)} 条")
    
    # 存储所有结果
    all_results = {}
    mae_matrix = {}
    rmse_matrix = {}
    mape_matrix = {}
    r2_matrix = {}
    
    # 循环测试所有组合
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"{'='*60}")
        
        # 获取该模型兼容的方法
        compatible_methods = get_compatible_methods(model_name)
        
        mae_matrix[model_name] = {}
        rmse_matrix[model_name] = {}
        mape_matrix[model_name] = {}
        r2_matrix[model_name] = {}
        
        for method in methods_to_run:
            # 检查兼容性
            if method not in compatible_methods:
                print(f"  {method}: 跳过（不兼容）")
                mae_matrix[model_name][method] = np.nan
                rmse_matrix[model_name][method] = np.nan
                mape_matrix[model_name][method] = np.nan
                r2_matrix[model_name][method] = np.nan
                continue
            
            print(f"\n  方法: {method}")
            
            # 训练和评估
            result = train_and_evaluate_single(
                model_name, method, df_train, df_test, config
            )
            
            # 保存结果
            key = f"{model_name}_{method}"
            all_results[key] = result
            
            if result['success']:
                metrics = result['metrics']
                mae_matrix[model_name][method] = metrics['MAE']
                rmse_matrix[model_name][method] = metrics['RMSE']
                mape_matrix[model_name][method] = metrics['MAPE']
                r2_matrix[model_name][method] = metrics['R2']
                
                print(f"    ✓ MAE: {metrics['MAE']:.4f} 秒")
                print(f"      RMSE: {metrics['RMSE']:.4f} 秒")
                print(f"      MAPE: {metrics['MAPE']:.2f} %")
                print(f"      R²: {metrics['R2']:.4f}")
            else:
                mae_matrix[model_name][method] = np.nan
                rmse_matrix[model_name][method] = np.nan
                mape_matrix[model_name][method] = np.nan
                r2_matrix[model_name][method] = np.nan
    
    # 创建对比表
    mae_df = pd.DataFrame(mae_matrix).T
    rmse_df = pd.DataFrame(rmse_matrix).T
    mape_df = pd.DataFrame(mape_matrix).T
    r2_df = pd.DataFrame(r2_matrix).T
    
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mae_df.to_csv(output_path / "MAE_matrix.csv")
    rmse_df.to_csv(output_path / "RMSE_matrix.csv")
    mape_df.to_csv(output_path / "MAPE_matrix.csv")
    r2_df.to_csv(output_path / "R2_matrix.csv")
    
    print(f"\n{'='*60}")
    print("基准测试完成！")
    print(f"{'='*60}")
    print(f"\n结果已保存到: {output_path}")
    
    # 打印MAE对比表
    print(f"\nMAE对比表 (秒):")
    print(mae_df.to_string())
    
    # 找出最佳组合
    mae_flat = mae_df.stack()
    best_idx = mae_flat.idxmin()
    best_mae = mae_flat.min()
    print(f"\n最佳组合: {best_idx[0]} × {best_idx[1]} (MAE = {best_mae:.4f} 秒)")
    
    return {
        'mae_matrix': mae_df,
        'rmse_matrix': rmse_df,
        'mape_matrix': mape_df,
        'r2_matrix': r2_df,
        'all_results': all_results
    }


if __name__ == "__main__":
    """测试训练评估流程"""
    
    print("这是train_eval.py的测试模式")
    print("实际使用时，应该从外部调用run_benchmark()函数")
    print("\n示例:")
    print("""
from pipelines.train_eval import run_benchmark
from utils.data import build_finaldata

# 构建FinalData
df = build_finaldata(
    link_id=115030402,
    precleaned_path="data/Precleaned_M67_...xlsx",
    capacity=6649,
    link_length_m=2713.8037
)

# 运行基准测试
results = run_benchmark(
    df=df,
    models_to_run=['M0'],
    methods_to_run=['classical', 'loglinear', 'nls'],
    train_end="2024-09-20"
)
    """)


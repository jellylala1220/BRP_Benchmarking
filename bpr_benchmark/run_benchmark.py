"""
BPR 基准测试主程序 - 总指挥

这是项目的"启动按钮"

职责：
1. 加载配置文件
2. 调用数据预处理模块
3. 循环遍历所有路段和模型
4. 训练和评估每个模型
5. 生成对比报告和可视化
6. 导出结果

使用方法：
    python run_benchmark.py
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from utils.data import load_and_preprocess, split_data
from utils.metrics import (
    calculate_all_metrics,
    calculate_stratified_metrics,
    calculate_metrics_by_time_period,
    create_metrics_comparison_table,
    generate_evaluation_report,
    print_metrics_summary
)
from models.base import create_model


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_single_model(model_name: str, config: dict, df_train: pd.DataFrame, 
                     df_test: pd.DataFrame, t_0: float, capacity: float) -> dict:
    """
    运行单个模型的训练和评估
    
    Args:
        model_name: 模型名称
        config: 配置字典
        df_train: 训练数据
        df_test: 测试数据
        t_0: 自由流行程时间
        capacity: 路段容量
        
    Returns:
        包含评估结果的字典
    """
    
    print(f"\n{'='*60}")
    print(f"运行模型: {model_name}")
    print(f"{'='*60}")
    
    try:
        # 创建模型实例
        model = create_model(model_name, config, t_0, capacity)
        
        # 训练模型
        print("训练中...")
        model.fit(df_train)
        
        # 获取模型参数
        params = model.get_params()
        print(f"模型参数: {params}")
        
        # 预测
        print("预测中...")
        y_pred = model.predict(df_test)
        y_true = df_test['t_ground_truth'].values
        
        # 评估 - 总体指标
        print("评估中...")
        overall_metrics = calculate_all_metrics(y_true, y_pred)
        print_metrics_summary(overall_metrics, model_name)
        
        # 评估 - 分层指标（按V/C比）
        vcr = df_test['V_C_Ratio'].values
        vcr_bins = config['metrics']['stratified']['by_vcr']
        stratified_metrics = calculate_stratified_metrics(y_true, y_pred, vcr, vcr_bins)
        
        print("\n分层评估（按V/C比）:")
        print(stratified_metrics.to_string(index=False))
        
        # 评估 - 按时段
        if 'is_peak' in df_test.columns:
            is_peak = df_test['is_peak'].values
            period_metrics = calculate_metrics_by_time_period(y_true, y_pred, is_peak)
            
            print("\n按时段评估:")
            print(period_metrics.to_string(index=False))
        else:
            period_metrics = None
        
        # 返回结果
        result = {
            'model_name': model_name,
            'overall': overall_metrics,
            'stratified': stratified_metrics,
            'by_period': period_metrics,
            'params': params,
            'predictions': y_pred,
            'success': True
        }
        
        print(f"\n✓ {model_name} 完成！")
        
        return result
        
    except Exception as e:
        print(f"\n✗ {model_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'model_name': model_name,
            'success': False,
            'error': str(e)
        }


def save_results(results: dict, output_dir: Path, road_name: str):
    """
    保存结果到文件
    
    Args:
        results: 所有模型的结果字典
        output_dir: 输出目录
        road_name: 路段名称
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存总体对比表
    comparison_data = {
        name: res['overall'] 
        for name, res in results.items() 
        if res['success']
    }
    
    if comparison_data:
        comparison_df = create_metrics_comparison_table(comparison_data)
        comparison_file = output_dir / f"{road_name}_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
        print(f"\n保存对比表到: {comparison_file}")
    
    # 2. 保存详细报告
    report_text = generate_evaluation_report(results, output_path=None)
    report_file = output_dir / f"{road_name}_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"保存评估报告到: {report_file}")
    
    # 3. 保存每个模型的分层结果
    for model_name, res in results.items():
        if not res['success']:
            continue
        
        if 'stratified' in res and res['stratified'] is not None:
            stratified_file = output_dir / f"{road_name}_{model_name}_stratified.csv"
            res['stratified'].to_csv(stratified_file, index=False, encoding='utf-8-sig')
    
    print(f"\n所有结果已保存到: {output_dir}")


def create_visualizations(results: dict, df_test: pd.DataFrame, 
                         output_dir: Path, road_name: str):
    """
    创建可视化图表
    
    Args:
        results: 所有模型的结果字典
        df_test: 测试数据
        output_dir: 输出目录
        road_name: 路段名称
    """
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        y_true = df_test['t_ground_truth'].values
        vcr = df_test['V_C_Ratio'].values
        
        # 1. 预测 vs 真实值散点图
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()
        
        for idx, (model_name, res) in enumerate(results.items()):
            if not res['success'] or idx >= 9:
                continue
            
            ax = axes[idx]
            y_pred = res['predictions']
            
            ax.scatter(y_true, y_pred, alpha=0.3, s=10)
            ax.plot([y_true.min(), y_true.max()], 
                   [y_true.min(), y_true.max()], 
                   'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('True Travel Time (s)', fontsize=10)
            ax.set_ylabel('Predicted Travel Time (s)', fontsize=10)
            ax.set_title(f"{model_name}\nMAE={res['overall']['MAE']:.2f}s", fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(results), 9):
            axes[idx].axis('off')
        
        plt.tight_layout()
        scatter_file = output_dir / f"{road_name}_predictions_scatter.png"
        plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存散点图到: {scatter_file}")
        
        # 2. 模型对比柱状图
        comparison_data = {
            name: res['overall'] 
            for name, res in results.items() 
            if res['success']
        }
        
        if comparison_data:
            comparison_df = create_metrics_comparison_table(comparison_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
            for idx, metric in enumerate(metrics):
                ax = axes[idx // 2, idx % 2]
                
                data = comparison_df.sort_values(metric if metric != 'R2' else metric, 
                                                ascending=(metric != 'R2'))
                
                ax.barh(data['Model'], data[metric])
                ax.set_xlabel(metric, fontsize=11)
                ax.set_title(f'Model Comparison - {metric}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                # 添加数值标签
                for i, v in enumerate(data[metric]):
                    ax.text(v, i, f' {v:.2f}', va='center', fontsize=9)
            
            plt.tight_layout()
            comparison_file = output_dir / f"{road_name}_comparison_bars.png"
            plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"保存对比柱状图到: {comparison_file}")
        
        # 3. V/C比 vs 误差分析
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()
        
        for idx, (model_name, res) in enumerate(results.items()):
            if not res['success'] or idx >= 9:
                continue
            
            ax = axes[idx]
            y_pred = res['predictions']
            errors = y_pred - y_true
            
            ax.scatter(vcr, errors, alpha=0.3, s=10)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('V/C Ratio', fontsize=10)
            ax.set_ylabel('Prediction Error (s)', fontsize=10)
            ax.set_title(f"{model_name} - Error Analysis", fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(results), 9):
            axes[idx].axis('off')
        
        plt.tight_layout()
        error_file = output_dir / f"{road_name}_error_analysis.png"
        plt.savefig(error_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存误差分析图到: {error_file}")
        
    except ImportError:
        print("\n警告：未安装 matplotlib，跳过可视化")
    except Exception as e:
        print(f"\n警告：可视化失败 - {e}")


def main():
    """主函数"""
    
    print("=" * 80)
    print("BPR 基准测试框架")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 加载配置
    print("步骤 1: 加载配置文件...")
    config = load_config("configs/default.yaml")
    print(f"  ✓ 配置加载完成")
    print(f"  - 测试路段: {config['roads_to_test']}")
    print(f"  - 测试模型: {config['models_to_run']}")
    
    # 2. 加载和预处理数据
    print("\n步骤 2: 加载和预处理数据...")
    all_data = load_and_preprocess("configs/default.yaml")
    
    # 3. 循环遍历每条路段
    all_results = {}
    
    for road_name in config['roads_to_test']:
        print("\n" + "=" * 80)
        print(f"处理路段: {road_name}")
        print("=" * 80)
        
        # 获取该路段的数据
        df_final = all_data[road_name]
        
        # 分割训练/测试集
        print("\n步骤 3: 分割训练/测试集...")
        df_train, df_test = split_data(df_final, config)
        
        # 获取路段参数
        road_info = config['roads'][road_name]
        capacity = road_info['capacity_vph']
        t_0 = df_train['t_0'].iloc[0]
        
        print(f"\n路段参数:")
        print(f"  - 容量: {capacity} vph")
        print(f"  - 自由流行程时间: {t_0:.2f} 秒")
        
        # 4. 循环遍历每个模型
        print("\n步骤 4: 训练和评估模型...")
        
        road_results = {}
        
        for model_name in config['models_to_run']:
            result = run_single_model(
                model_name=model_name,
                config=config,
                df_train=df_train,
                df_test=df_test,
                t_0=t_0,
                capacity=capacity
            )
            
            road_results[model_name] = result
        
        # 5. 保存结果
        print("\n步骤 5: 保存结果...")
        output_dir = Path(config['output']['dir']) / road_name
        
        if config['output']['save_summary']:
            save_results(road_results, output_dir, road_name)
        
        # 6. 创建可视化
        if config['output']['save_plots']:
            print("\n步骤 6: 创建可视化...")
            create_visualizations(road_results, df_test, output_dir, road_name)
        
        # 7. 保存预测结果（可选）
        if config['output']['save_predictions']:
            print("\n步骤 7: 保存预测结果...")
            predictions_df = df_test[['timestamp', 't_ground_truth', 'V_C_Ratio']].copy()
            
            for model_name, res in road_results.items():
                if res['success']:
                    predictions_df[f'{model_name}_pred'] = res['predictions']
            
            pred_file = output_dir / f"{road_name}_predictions.csv"
            predictions_df.to_csv(pred_file, index=False, encoding='utf-8-sig')
            print(f"保存预测结果到: {pred_file}")
        
        all_results[road_name] = road_results
    
    # 8. 生成总结报告
    print("\n" + "=" * 80)
    print("基准测试完成！")
    print("=" * 80)
    
    for road_name, road_results in all_results.items():
        print(f"\n{road_name} 最佳模型:")
        
        successful_results = {
            name: res['overall'] 
            for name, res in road_results.items() 
            if res['success']
        }
        
        if successful_results:
            best_model = min(successful_results.items(), key=lambda x: x[1]['MAE'])
            print(f"  模型: {best_model[0]}")
            print(f"  MAE: {best_model[1]['MAE']:.4f} 秒")
            print(f"  RMSE: {best_model[1]['RMSE']:.4f} 秒")
            print(f"  MAPE: {best_model[1]['MAPE']:.2f} %")
            print(f"  R²: {best_model[1]['R2']:.4f}")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n所有结果已保存到 outputs/ 目录")


if __name__ == "__main__":
    main()


"""
使用示例脚本

展示如何使用BPR基准测试框架的各个组件

这个脚本演示了：
1. 如何加载和预处理数据
2. 如何创建和训练单个模型
3. 如何评估模型性能
4. 如何比较多个模型

使用方法：
    python example_usage.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from utils.data import load_config, create_final_dataset, split_data
from utils.metrics import calculate_all_metrics, print_metrics_summary
from models.base import create_model


def example_1_load_data():
    """示例1: 加载和预处理数据"""
    
    print("\n" + "=" * 60)
    print("示例1: 加载和预处理数据")
    print("=" * 60)
    
    # 加载配置
    config = load_config("configs/default.yaml")
    
    # 获取路段信息
    road_name = config['roads_to_test'][0]
    road_info = config['roads'][road_name]
    
    print(f"\n路段: {road_name}")
    print(f"  LinkID: {road_info['link_id']}")
    print(f"  长度: {road_info['length_km']} km")
    print(f"  容量: {road_info['capacity_vph']} vph")
    
    # 加载数据
    data_file = Path(__file__).parent / config['data']['precleaned_file']
    
    if not data_file.exists():
        print(f"\n数据文件不存在: {data_file}")
        print("请确保数据文件路径正确")
        return None, None, None
    
    print(f"\n正在加载数据...")
    df_final = create_final_dataset(
        link_id=road_info['link_id'],
        precleaned_filepath=str(data_file),
        config=config
    )
    
    print(f"\n数据预览:")
    print(df_final.head())
    
    print(f"\n数据统计:")
    print(df_final[['t_ground_truth', 'V_C_Ratio', 'v_avg_kmh']].describe())
    
    # 分割数据
    df_train, df_test = split_data(df_final, config)
    
    return config, df_train, df_test


def example_2_single_model(config, df_train, df_test):
    """示例2: 训练和评估单个模型"""
    
    if config is None:
        print("\n跳过示例2（数据未加载）")
        return
    
    print("\n" + "=" * 60)
    print("示例2: 训练和评估单个模型")
    print("=" * 60)
    
    # 获取参数
    t_0 = df_train['t_0'].iloc[0]
    capacity = config['roads'][config['roads_to_test'][0]]['capacity_vph']
    
    # 创建模型
    model_name = 'NLS_BPR'
    print(f"\n创建模型: {model_name}")
    model = create_model(model_name, config, t_0, capacity)
    
    # 训练
    print("训练中...")
    model.fit(df_train)
    
    # 获取参数
    params = model.get_params()
    print(f"\n拟合的参数:")
    for key, value in params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 预测
    print("\n预测中...")
    y_pred = model.predict(df_test)
    y_true = df_test['t_ground_truth'].values
    
    # 评估
    print("\n评估中...")
    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics_summary(metrics, model_name)
    
    return model, y_pred


def example_3_compare_models(config, df_train, df_test):
    """示例3: 比较多个模型"""
    
    if config is None:
        print("\n跳过示例3（数据未加载）")
        return
    
    print("\n" + "=" * 60)
    print("示例3: 比较多个模型")
    print("=" * 60)
    
    # 获取参数
    t_0 = df_train['t_0'].iloc[0]
    capacity = config['roads'][config['roads_to_test'][0]]['capacity_vph']
    
    # 要比较的模型
    model_names = ['ClassicalBPR', 'NLS_BPR', 'LogLinearBPR', 'RandomForest']
    
    results = {}
    
    for model_name in model_names:
        print(f"\n处理 {model_name}...")
        
        try:
            # 创建和训练模型
            model = create_model(model_name, config, t_0, capacity)
            model.fit(df_train)
            
            # 预测和评估
            y_pred = model.predict(df_test)
            y_true = df_test['t_ground_truth'].values
            metrics = calculate_all_metrics(y_true, y_pred)
            
            results[model_name] = metrics
            
            print(f"  MAE: {metrics['MAE']:.2f} 秒")
            
        except Exception as e:
            print(f"  失败: {e}")
    
    # 创建对比表
    print("\n" + "=" * 60)
    print("模型对比")
    print("=" * 60)
    
    from utils.metrics import create_metrics_comparison_table
    
    comparison_df = create_metrics_comparison_table(results)
    print("\n" + comparison_df.to_string(index=False))
    
    # 找出最佳模型
    best_model = comparison_df.iloc[0]
    print(f"\n最佳模型: {best_model['Model']}")
    print(f"  MAE: {best_model['MAE']:.2f} 秒")
    print(f"  RMSE: {best_model['RMSE']:.2f} 秒")
    print(f"  MAPE: {best_model['MAPE']:.2f} %")
    print(f"  R²: {best_model['R2']:.4f}")


def example_4_custom_model():
    """示例4: 创建自定义模型"""
    
    print("\n" + "=" * 60)
    print("示例4: 创建自定义模型")
    print("=" * 60)
    
    print("\n自定义模型示例代码:")
    print("""
from models.base import BPRModel
import numpy as np

class MyCustomBPR(BPRModel):
    '''自定义BPR模型'''
    
    def fit(self, df_train):
        # 实现您的参数估计方法
        # 例如：使用遗传算法、粒子群优化等
        
        # 这里用简单的方法演示
        self.alpha = 0.20
        self.beta = 3.5
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test):
        vcr = df_test['V_C_Ratio'].values
        t_0 = df_test['t_0'].values
        
        return self.bpr_function(vcr, t_0, self.alpha, self.beta)

# 使用自定义模型
model = MyCustomBPR(config, t_0=100, capacity=6000)
model.fit(df_train)
y_pred = model.predict(df_test)
    """)
    
    print("\n要添加自定义模型:")
    print("1. 在 models/ 目录创建新文件")
    print("2. 继承 BaseModel 或 BPRModel/MLModel")
    print("3. 实现 fit() 和 predict() 方法")
    print("4. 在 models/base.py 的 create_model() 中注册")


def example_5_feature_analysis(config, df_train, df_test):
    """示例5: 特征重要性分析"""
    
    if config is None:
        print("\n跳过示例5（数据未加载）")
        return
    
    print("\n" + "=" * 60)
    print("示例5: 特征重要性分析")
    print("=" * 60)
    
    # 获取参数
    t_0 = df_train['t_0'].iloc[0]
    capacity = config['roads'][config['roads_to_test'][0]]['capacity_vph']
    
    # 使用随机森林（提供特征重要性）
    print("\n使用随机森林分析特征重要性...")
    
    model = create_model('RandomForest', config, t_0, capacity)
    model.fit(df_train)
    
    # 获取特征重要性
    if hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance()
        
        print("\n特征重要性排序:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        for feature, score in sorted_importance:
            print(f"  {feature:20s}: {score:.4f}")
    else:
        print("该模型不支持特征重要性分析")


def main():
    """主函数"""
    
    print("\n" + "*" * 60)
    print("BPR 基准测试框架 - 使用示例")
    print("*" * 60)
    
    # 示例1: 加载数据
    config, df_train, df_test = example_1_load_data()
    
    if config is not None:
        # 示例2: 单个模型
        example_2_single_model(config, df_train, df_test)
        
        # 示例3: 比较模型
        example_3_compare_models(config, df_train, df_test)
        
        # 示例5: 特征分析
        example_5_feature_analysis(config, df_train, df_test)
    
    # 示例4: 自定义模型（不需要数据）
    example_4_custom_model()
    
    print("\n" + "*" * 60)
    print("示例完成！")
    print("*" * 60)
    
    print("\n更多信息:")
    print("  - 完整文档: README.md")
    print("  - 快速开始: QUICKSTART.md")
    print("  - 项目结构: PROJECT_STRUCTURE.md")
    print("  - 运行基准测试: python run_benchmark.py")
    print("\n")


if __name__ == "__main__":
    main()


"""
BPR框架2.0 - 完整使用示例

展示如何使用新架构的所有功能
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

print("="*60)
print("BPR框架2.0 - 使用示例")
print("="*60)

# ========== 示例1: 构建FinalData ==========
print("\n示例1: 构建FinalData")
print("-"*60)

from utils.data import build_finaldata, finaldata_qc_report

# 注意：这里使用模拟数据，实际使用时替换为真实路径
print("构建FinalData（模拟数据）...")

# 生成模拟数据用于演示
np.random.seed(42)
n = 1000

v_over_c = np.random.uniform(0.1, 1.2, n)
t0 = 100
alpha = 0.20
beta = 3.5

t_true = t0 * (1 + alpha * np.power(v_over_c, beta))
t_true += np.random.normal(0, 5, n)

datetime_range = pd.date_range('2024-09-01', periods=n, freq='15min')

df = pd.DataFrame({
    'datetime': datetime_range,
    'LinkUID': 115030402,
    'flow_veh_hr': v_over_c * 6649,
    'capacity': 6649,
    'link_length_m': 2713.8037,
    'fused_tt_15min': t_true,
    't0_ff': t0,
    'v_over_c': v_over_c,
    'count_len_cat1': 100,
    'count_len_cat2': 50,
    'count_len_cat3': 20,
    'count_len_cat4': 10,
    'share_len_cat1': 0.5,
    'share_len_cat2': 0.3,
    'share_len_cat3': 0.15,
    'share_len_cat4': 0.05,
    'hgv_share': 0.2,
    'hour': datetime_range.hour,
    'weekday': datetime_range.dayofweek,
    'daytype': 'weekday',
    'is_valid': 1,
    'flag_tt_outlier': 0,
    'fused_tt_15min_winsor': t_true
})

print(f"✓ FinalData创建完成: {df.shape}")
print(f"  时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")

# 生成QC报告
qc_report = finaldata_qc_report(df)
print("\nQC报告:")
print(qc_report.head(10).to_string(index=False))

# ========== 示例2: 使用单个模型 ==========
print("\n\n示例2: 使用单个模型")
print("-"*60)

from models.m0_bpr_new import M0_BPR

# 分割数据
split_idx = int(0.8 * len(df))
df_train = df.iloc[:split_idx]
df_test = df.iloc[split_idx:]

print(f"数据分割: 训练集{len(df_train)}条, 测试集{len(df_test)}条")

# 创建和训练模型
print("\n训练M0_BPR模型...")
model = M0_BPR()
model.fit(df_train, method='nls')

# 预测
y_pred = model.predict(df_test)
y_true = df_test['fused_tt_15min'].values

# 评估
from utils.metrics import calculate_all_metrics

metrics = calculate_all_metrics(y_true, y_pred)
print(f"\n评估结果:")
print(f"  MAE: {metrics['MAE']:.2f} 秒")
print(f"  RMSE: {metrics['RMSE']:.2f} 秒")
print(f"  MAPE: {metrics['MAPE']:.2f} %")
print(f"  R²: {metrics['R2']:.4f}")

# ========== 示例3: 对比多种方法 ==========
print("\n\n示例3: 对比多种方法")
print("-"*60)

methods = ['classical', 'loglinear', 'nls']
results = {}

for method in methods:
    model = M0_BPR()
    model.fit(df_train, method=method)
    y_pred = model.predict(df_test)
    metrics = calculate_all_metrics(y_true, y_pred)
    results[method] = metrics['MAE']
    print(f"{method:12s}: MAE = {metrics['MAE']:.4f} 秒")

best_method = min(results, key=results.get)
print(f"\n最佳方法: {best_method} (MAE = {results[best_method]:.4f})")

# ========== 示例4: 使用动态参数模型 ==========
print("\n\n示例4: 使用动态参数模型")
print("-"*60)

from models.m1_dp_bpr import M1_DP_BPR

# 添加时段信息
df_train['is_peak'] = ((df_train['hour'] >= 7) & (df_train['hour'] < 9) |
                        (df_train['hour'] >= 15) & (df_train['hour'] < 18)).astype(int)
df_test['is_peak'] = ((df_test['hour'] >= 7) & (df_test['hour'] < 9) |
                       (df_test['hour'] >= 15) & (df_test['hour'] < 18)).astype(int)

model_m1 = M1_DP_BPR()
model_m1.fit(df_train, method='nls')

y_pred_m1 = model_m1.predict(df_test)
metrics_m1 = calculate_all_metrics(y_true, y_pred_m1)

print(f"\nM1_DP_BPR评估:")
print(f"  MAE: {metrics_m1['MAE']:.2f} 秒")
print(f"  相比M0改进: {metrics['MAE'] - metrics_m1['MAE']:.2f} 秒")

# ========== 示例5: 完整基准测试 ==========
print("\n\n示例5: 完整基准测试")
print("-"*60)

from pipelines.train_eval import run_benchmark

print("运行基准测试...")
results = run_benchmark(
    df=df,
    models_to_run=['M0', 'M1'],
    methods_to_run=['classical', 'loglinear', 'nls'],
    train_end=df['datetime'].iloc[split_idx],
    output_dir="outputs/example_benchmark"
)

print("\n✓ 基准测试完成！")
print("\nMAE对比表:")
print(results['mae_matrix'])

# ========== 示例6: 使用注册表 ==========
print("\n\n示例6: 使用注册表")
print("-"*60)

from pipelines.registry import print_registry, get_available_models, get_available_estimators

print("\n可用模型:", get_available_models())
print("可用估计器:", get_available_estimators())

# ========== 总结 ==========
print("\n\n" + "="*60)
print("示例完成！")
print("="*60)

print("\n您已学会：")
print("  ✓ 构建FinalData")
print("  ✓ 使用单个模型")
print("  ✓ 对比多种方法")
print("  ✓ 使用动态参数模型")
print("  ✓ 运行完整基准测试")
print("  ✓ 使用注册表")

print("\n更多信息请参考:")
print("  - README.md: 完整文档")
print("  - PROJECT_COMPLETE.md: 项目完成报告")
print("  - QUICKSTART.md: 快速开始")

print("\n🎉 BPR框架2.0已准备就绪！")


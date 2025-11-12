"""
快速测试脚本 - 测试单个模型

这是最简单的运行方式，适合第一次使用
"""

import sys
from pathlib import Path
import pandas as pd

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from utils.data import build_finaldata
from models.m0_bpr_new import M0_BPR
from utils.metrics import calculate_all_metrics

print("="*60)
print("BPR框架 - 快速测试")
print("="*60)

# ========== 步骤1: 构建FinalData ==========
print("\n[步骤1] 构建FinalData")
print("-"*60)

try:
    df = build_finaldata(
        link_id=115030402,
        precleaned_path="../Data/Precleaned_M67_Traffic_Data_September_2024.xlsx",
        snapshot_csv_path="../Data/M67 westbound between J4 and J3 mainCarriageway 115030402.csv",
        capacity=6649,
        link_length_m=2713.8037,
        month_start="2024-09-01",
        month_end="2024-09-30",
        t0_strategy="min5pct"
    )
    
    print(f"\n✓ FinalData构建成功！")
    print(f"  数据形状: {df.shape}")
    print(f"  时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
    print(f"  平均流量: {df['flow_veh_hr'].mean():.0f} veh/hr")
    print(f"  平均V/C: {df['v_over_c'].mean():.3f}")
    print(f"  平均行程时间: {df['fused_tt_15min'].mean():.2f} 秒")
    print(f"  自由流时间: {df['t0_ff'].iloc[0]:.2f} 秒")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== 步骤2: 分割训练/测试集 ==========
print("\n[步骤2] 分割训练/测试集")
print("-"*60)

train_end = "2024-09-20"
df_train = df[df['datetime'] <= train_end].copy()
df_test = df[df['datetime'] > train_end].copy()

print(f"  训练集: {len(df_train)} 条 (至 {train_end})")
print(f"  测试集: {len(df_test)} 条 (从 {train_end} 之后)")

if len(df_train) == 0 or len(df_test) == 0:
    print("\n❌ 错误: 训练集或测试集为空！")
    print("   请检查时间范围设置")
    sys.exit(1)

# ========== 步骤3: 训练模型 ==========
print("\n[步骤3] 训练M0_BPR模型")
print("-"*60)

try:
    model = M0_BPR()
    model.fit(df_train, method='nls')
    print("\n✓ 模型训练完成")
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== 步骤4: 预测 ==========
print("\n[步骤4] 预测")
print("-"*60)

try:
    y_pred = model.predict(df_test)
    y_true = df_test['fused_tt_15min'].values
    
    print(f"  预测了 {len(y_pred)} 条记录")
    print(f"  预测范围: [{y_pred.min():.2f}, {y_pred.max():.2f}] 秒")
    print(f"  真实范围: [{y_true.min():.2f}, {y_true.max():.2f}] 秒")
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== 步骤5: 评估 ==========
print("\n[步骤5] 评估结果")
print("-"*60)

try:
    metrics = calculate_all_metrics(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print("评估指标")
    print(f"{'='*60}")
    print(f"  MAE:  {metrics['MAE']:.2f} 秒")
    print(f"  RMSE: {metrics['RMSE']:.2f} 秒")
    print(f"  MAPE: {metrics['MAPE']:.2f} %")
    print(f"  R²:   {metrics['R2']:.4f}")
    print(f"{'='*60}")
    
    # 模型信息
    model_info = model.info()
    print(f"\n模型信息:")
    print(f"  模型: {model_info.get('model', 'N/A')}")
    print(f"  方法: {model_info.get('method', 'N/A')}")
    if 'estimator_info' in model_info:
        params = model_info['estimator_info'].get('params', {})
        print(f"  α = {params.get('alpha', 'N/A'):.4f}")
        print(f"  β = {params.get('beta', 'N/A'):.4f}")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== 完成 ==========
print("\n" + "="*60)
print("✓ 测试完成！")
print("="*60)
print("\n下一步:")
print("  1. 查看 outputs/ 目录中的结果")
print("  2. 尝试运行完整基准测试: python run_benchmark.py")
print("  3. 查看文档: RUN_GUIDE.md")


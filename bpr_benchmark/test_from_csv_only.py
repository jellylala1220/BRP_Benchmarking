"""
从CSV文件直接构建FinalData（不依赖Precleaned数据）

适用于：Precleaned数据中没有该LinkID的情况
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))

from utils.metrics import calculate_all_metrics
from models.m0_bpr_new import M0_BPR

print("="*60)
print("从CSV文件构建FinalData")
print("="*60)

# 读取CSV文件
csv_path = "../Data/M67 westbound between J4 and J3 mainCarriageway 115030402.csv"
print(f"\n读取CSV文件: {csv_path}")

df_csv = pd.read_csv(csv_path)

# 创建时间戳
df_csv['datetime'] = pd.to_datetime(
    df_csv['Local Date'].astype(str).str.strip() + ' ' + df_csv[' Local Time'].astype(str).str.strip(),
    format='%Y-%m-%d %H:%M:%S',
    errors='coerce'
)

df_csv = df_csv[df_csv['datetime'].notna()].copy()
print(f"✓ 读取 {len(df_csv)} 条记录")

# 聚合到15分钟窗口
df_csv['datetime_15min'] = df_csv['datetime'].dt.floor('15min')

# 获取Fused Travel Time列（处理空格）
fused_tt_col = None
for col in df_csv.columns:
    if 'Fused Travel Time' in col:
        fused_tt_col = col
        break

if not fused_tt_col:
    raise ValueError("未找到Fused Travel Time列")

# 聚合
df_agg = df_csv.groupby('datetime_15min').agg({
    fused_tt_col: 'mean',
    ' Total Traffic Flow': 'sum',  # 总流量
    ' Fused Average Speed': 'mean'  # 平均速度
}).reset_index()

df_agg.columns = ['datetime', 'fused_tt_15min', 'total_flow', 'avg_speed']

# 添加路段信息
link_length_m = 2713.8037
link_length_km = link_length_m / 1000
capacity = 6649

df_agg['LinkUID'] = 115030402
df_agg['capacity'] = capacity
df_agg['link_length_m'] = link_length_m

# 计算小时流量（CSV中的Total Traffic Flow已经是15分钟流量）
df_agg['flow_veh_hr'] = df_agg['total_flow'] * 4
df_agg['v_over_c'] = df_agg['flow_veh_hr'] / capacity

# 计算T0（从fused_tt_15min取最低5%）
t0_candidates = df_agg['fused_tt_15min'].nsmallest(int(len(df_agg) * 0.05))
t0_ff = t0_candidates.mean()
df_agg['t0_ff'] = t0_ff

# 添加时间特征
df_agg['hour'] = df_agg['datetime'].dt.hour
df_agg['weekday'] = df_agg['datetime'].dt.dayofweek
df_agg['daytype'] = df_agg['weekday'].apply(lambda x: 'weekday' if x < 5 else 'weekend')
df_agg['is_peak'] = ((df_agg['hour'] >= 7) & (df_agg['hour'] < 9) |
                     (df_agg['hour'] >= 15) & (df_agg['hour'] < 18)).astype(int)

# 添加默认值
df_agg['hgv_share'] = 0.15  # 默认值
df_agg['is_raining'] = 0
df_agg['temperature'] = np.nan
df_agg['is_valid'] = 1

print(f"\n✓ FinalData构建完成: {df_agg.shape}")
print(f"  时间范围: {df_agg['datetime'].min()} 至 {df_agg['datetime'].max()}")
print(f"  平均行程时间: {df_agg['fused_tt_15min'].mean():.2f} 秒")
print(f"  自由流时间: {t0_ff:.2f} 秒")

# 分割训练/测试集
train_end = "2024-09-20"
df_train = df_agg[df_agg['datetime'] <= train_end].copy()
df_test = df_agg[df_agg['datetime'] > train_end].copy()

print(f"\n数据分割:")
print(f"  训练集: {len(df_train)} 条")
print(f"  测试集: {len(df_test)} 条")

# 训练模型
print("\n" + "="*60)
print("训练M0_BPR模型")
print("="*60)

model = M0_BPR()
model.fit(df_train, method='nls')

# 预测
y_pred = model.predict(df_test)
y_true = df_test['fused_tt_15min'].values

# 评估
metrics = calculate_all_metrics(y_true, y_pred)

print("\n" + "="*60)
print("评估结果")
print("="*60)
print(f"  MAE:  {metrics['MAE']:.2f} 秒")
print(f"  RMSE: {metrics['RMSE']:.2f} 秒")
print(f"  MAPE: {metrics['MAPE']:.2f} %")
print(f"  R²:   {metrics['R2']:.4f}")
print("="*60)

print("\n✓ 测试完成！")


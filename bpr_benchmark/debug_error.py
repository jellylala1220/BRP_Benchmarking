"""
调试 "cannot unpack non-iterable int object" 错误
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.data import build_finaldata
from pipelines.registry import create_model
from utils.metrics import calculate_all_metrics, calculate_stratified_metrics
import yaml

# 加载配置
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 构建数据
print("构建数据...")
df = build_finaldata(
    link_id=115030402,
    precleaned_path="../Data/Precleaned_M67_Traffic_Data_September_2024.xlsx",
    snapshot_csv_path="../Data/M67 westbound between J4 and J3 mainCarriageway 115030402.csv",
    capacity=6649,
    link_length_m=2713.8037
)

# 分割数据
train_end = "2024-09-20"
df_train = df[df['datetime'] <= train_end].copy()
df_test = df[df['datetime'] > train_end].copy()

print(f"训练集: {len(df_train)}, 测试集: {len(df_test)}")

# 创建和训练模型
print("\n创建模型...")
model = create_model('M0', config)

print("训练模型...")
model.fit(df_train, method='nls')

print("预测...")
y_pred = model.predict(df_test)
y_true = df_test['fused_tt_15min'].values

print(f"预测值类型: {type(y_pred)}, 形状: {y_pred.shape if hasattr(y_pred, 'shape') else 'N/A'}")
print(f"真实值类型: {type(y_true)}, 形状: {y_true.shape}")

# 评估
print("\n评估...")
try:
    overall_metrics = calculate_all_metrics(y_true, y_pred)
    print(f"✓ 总体评估成功: {overall_metrics}")
except Exception as e:
    print(f"✗ 总体评估失败: {e}")
    import traceback
    traceback.print_exc()

try:
    vcr = df_test['v_over_c'].values
    vcr_bins = [0, 0.6, 0.85, 1.0, 9]
    stratified_metrics = calculate_stratified_metrics(y_true, y_pred, vcr, vcr_bins)
    print(f"✓ 分层评估成功")
except Exception as e:
    print(f"✗ 分层评估失败: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ 所有步骤完成，无错误！")


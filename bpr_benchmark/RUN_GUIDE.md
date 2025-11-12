# 🚀 BPR框架运行指南

**一步一步教您如何运行代码**

---

## 📋 前置检查

### 步骤0：检查环境

```bash
# 1. 确认您在项目根目录
cd /Users/lvlei/PycharmProjects/BPR

# 2. 检查Python版本（需要Python 3.7+）
python --version

# 3. 安装依赖（如果还没安装）
pip install -r bpr_benchmark/requirements.txt
```

---

## 🎯 方式1：快速测试（推荐新手）

### 步骤1：构建FinalData

```bash
cd bpr_benchmark

# 使用CLI工具构建FinalData
python -m pipelines.build_finaldata \
    --link 115030402 \
    --preclean "../Data/Precleaned_M67_Traffic_Data_September_2024.xlsx" \
    --snapshot "../Data/M67 westbound between J4 and J3 mainCarriageway 115030402.csv" \
    --capacity 6649 \
    --length 2713.8037 \
    --start 2024-09-01 \
    --end 2024-09-30 \
    --output outputs/finaldata/
```

**预期输出**：
- `outputs/finaldata/finaldata_115030402.parquet` - FinalData文件
- `outputs/finaldata/qc_report_115030402.csv` - 质量报告

### 步骤2：运行单个模型测试

创建一个测试脚本 `test_single_model.py`：

```python
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from utils.data import build_finaldata
from models.m0_bpr_new import M0_BPR
from utils.metrics import calculate_all_metrics

# 1. 构建FinalData
print("="*60)
print("步骤1: 构建FinalData")
print("="*60)

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

print(f"\n✓ FinalData构建完成: {df.shape}")
print(f"  时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")

# 2. 分割训练/测试集
print("\n" + "="*60)
print("步骤2: 分割训练/测试集")
print("="*60)

train_end = "2024-09-20"
df_train = df[df['datetime'] <= train_end].copy()
df_test = df[df['datetime'] > train_end].copy()

print(f"  训练集: {len(df_train)} 条 (至 {train_end})")
print(f"  测试集: {len(df_test)} 条 (从 {train_end} 之后)")

# 3. 训练模型
print("\n" + "="*60)
print("步骤3: 训练M0_BPR模型")
print("="*60)

model = M0_BPR()
model.fit(df_train, method='nls')

# 4. 预测
print("\n" + "="*60)
print("步骤4: 预测")
print("="*60)

y_pred = model.predict(df_test)
y_true = df_test['fused_tt_15min'].values

# 5. 评估
print("\n" + "="*60)
print("步骤5: 评估结果")
print("="*60)

metrics = calculate_all_metrics(y_true, y_pred)

print(f"\n评估指标:")
print(f"  MAE:  {metrics['MAE']:.2f} 秒")
print(f"  RMSE: {metrics['RMSE']:.2f} 秒")
print(f"  MAPE: {metrics['MAPE']:.2f} %")
print(f"  R²:   {metrics['R2']:.4f}")

print("\n✓ 测试完成！")
```

运行：
```bash
python test_single_model.py
```

---

## 🎯 方式2：运行完整基准测试

### 步骤1：更新配置文件

编辑 `configs/default.yaml`，确保路径正确：

```yaml
data:
  precleaned_file: "../Data/Precleaned_M67_Traffic_Data_September_2024.xlsx"
  
roads:
  M67_115030402:
    link_id: 115030402
    link_name: "M67 westbound between J4 and J3"
    length_km: 2.7138037
    capacity_vph: 6649
    link_length_m: 2713.8037  # 添加这个字段
    snapshot_csv: "../Data/M67 westbound between J4 and J3 mainCarriageway 115030402.csv"  # 添加CSV路径

# 更新模型列表（使用新架构）
models: [M0, M1, M2, M3, M4, M5, M6]
methods: [classical, loglinear, nls, svr, tree, rf, gbdt, nn, bayes]

# 添加builder配置
builder:
  t0_strategy: min5pct
  winsor: [0.01, 0.99]
  vc_bins: [0, 0.6, 0.85, 1.0, 9]

# 添加train配置
train:
  split:
    train_end: "2024-09-20"
  filters:
    use_winsor_tt: false
    require_is_valid: true
```

### 步骤2：运行基准测试

```bash
cd bpr_benchmark
python run_benchmark.py
```

**预期输出**：
- 训练和评估所有模型×方法组合
- 生成MAE/RMSE/MAPE/R²对比表
- 保存结果到 `outputs/` 目录

---

## 🎯 方式3：使用pipelines（推荐）

### 步骤1：构建FinalData

```bash
python -m pipelines.build_finaldata \
    --link 115030402 \
    --preclean "../Data/Precleaned_M67_Traffic_Data_September_2024.xlsx" \
    --snapshot "../Data/M67 westbound between J4 and J3 mainCarriageway 115030402.csv" \
    --capacity 6649 \
    --length 2713.8037 \
    --start 2024-09-01 \
    --end 2024-09-30 \
    --output outputs/finaldata/
```

### 步骤2：运行训练和评估

创建 `run_quick_test.py`：

```python
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from utils.data import build_finaldata
from pipelines.train_eval import run_benchmark

# 1. 构建FinalData
print("="*60)
print("构建FinalData")
print("="*60)

df = build_finaldata(
    link_id=115030402,
    precleaned_path="../Data/Precleaned_M67_Traffic_Data_September_2024.xlsx",
    snapshot_csv_path="../Data/M67 westbound between J4 and J3 mainCarriageway 115030402.csv",
    capacity=6649,
    link_length_m=2713.8037,
    month_start="2024-09-01",
    month_end="2024-09-30"
)

# 2. 运行基准测试
print("\n" + "="*60)
print("运行基准测试")
print("="*60)

results = run_benchmark(
    df=df,
    models_to_run=['M0', 'M1'],  # 先测试两个模型
    methods_to_run=['classical', 'loglinear', 'nls'],  # 测试3种方法
    train_end="2024-09-20",
    output_dir="outputs/quick_test"
)

# 3. 查看结果
print("\n" + "="*60)
print("结果摘要")
print("="*60)
print(results['mae_matrix'])
```

运行：
```bash
python run_quick_test.py
```

---

## 🔍 故障排查

### 问题1：找不到数据文件

**错误**：`FileNotFoundError: ../Data/Precleaned_...xlsx`

**解决**：
```bash
# 检查文件是否存在
ls -lh ../Data/Precleaned_M67_Traffic_Data_September_2024.xlsx

# 如果路径不对，使用绝对路径
python -m pipelines.build_finaldata \
    --preclean "/Users/lvlei/PycharmProjects/BPR/Data/Precleaned_M67_Traffic_Data_September_2024.xlsx" \
    ...
```

### 问题2：缺少依赖包

**错误**：`ModuleNotFoundError: No module named 'pandas'`

**解决**：
```bash
pip install pandas numpy scipy scikit-learn openpyxl pyyaml
# 或
pip install -r requirements.txt
```

### 问题3：CSV匹配失败

**警告**：`匹配率较低`

**解决**：
- 检查CSV文件的时间格式
- 确保CSV文件与Precleaned数据的时间范围一致
- 如果匹配率<50%，代码会自动回退到计算值

---

## 📊 预期输出示例

### FinalData构建输出

```
============================================================
构建FinalData: LinkID=115030402
============================================================

[1/8] 加载数据...
  找到 2880 条记录

[2/8] 时间筛选...
  筛选后剩余 2880 条记录

[3/8] 获取路段参数...
  容量: 6649 veh/hr
  长度: 2713.8037 m (2.714 km)

[4/8] 计算流量...
  平均流量: 3456 veh/hr
  平均V/C: 0.520
  平均HGV份额: 0.150

[5/8] 计算速度和行程时间...
  从CSV文件读取Fused Travel Time: ../Data/...
  CSV数据聚合：2878 条秒级记录 → 2880 个15分钟窗口
  ✓ 成功匹配 2880/2880 条记录 (100.0%)
  平均速度: 92.50 km/h
  平均行程时间: 105.67 秒

[6/8] 计算自由流行程时间 (策略: min5pct)...
  使用最低5%的fused_tt_15min均值
  自由流行程时间 t0: 105.23 秒
  最低5%范围: [105.13, 105.45] 秒

[7/8] Winsorize异常值处理...
  异常值数量: 58 (2.01%)
  Winsor边界: [95.50, 125.30] 秒

[8/8] 提取时间特征...
  有效记录: 2850 / 2880 (98.96%)

============================================================
✓ FinalData构建完成！
  形状: (2880, 23)
  时间范围: 2024-09-01 00:00:00 至 2024-09-30 23:45:00
============================================================
```

### 模型训练输出

```
============================================================
M0_BPR 训练
  方法: nls
============================================================

拟合参数:
  α = 0.2015
  β = 3.4567
  t0 = 105.23 秒

✓ M0_BPR训练完成
```

### 评估结果输出

```
评估指标:
  MAE:  8.45 秒
  RMSE: 12.34 秒
  MAPE: 7.89 %
  R²:   0.9234
```

---

## 🎯 推荐流程（完整版）

### 第一次运行（验证环境）

```bash
# 1. 进入项目目录
cd /Users/lvlei/PycharmProjects/BPR/bpr_benchmark

# 2. 运行示例脚本（最简单）
python example_usage_new.py
```

### 第二次运行（测试单个模型）

```bash
# 创建test_single_model.py（见上面的代码）
python test_single_model.py
```

### 第三次运行（完整基准测试）

```bash
# 1. 构建FinalData
python -m pipelines.build_finaldata \
    --link 115030402 \
    --preclean "../Data/Precleaned_M67_Traffic_Data_September_2024.xlsx" \
    --snapshot "../Data/M67 westbound between J4 and J3 mainCarriageway 115030402.csv" \
    --capacity 6649 --length 2713.8037 \
    --start 2024-09-01 --end 2024-09-30

# 2. 运行完整基准测试
python run_benchmark.py
```

---

## ✅ 检查清单

运行前确认：
- [ ] Python 3.7+ 已安装
- [ ] 依赖包已安装（`pip install -r requirements.txt`）
- [ ] 数据文件路径正确
- [ ] CSV文件路径正确（如果使用）
- [ ] 输出目录可写（`outputs/`）

运行后检查：
- [ ] FinalData文件已生成
- [ ] QC报告已生成
- [ ] 模型训练无错误
- [ ] 评估结果已保存
- [ ] MAE/RMSE值合理

---

## 🆘 需要帮助？

如果遇到问题：
1. 检查错误信息
2. 查看 `outputs/` 目录中的日志
3. 参考 `GROUND_TRUTH_EXPLANATION.md` 和 `T0_CALCULATION_GUIDE.md`
4. 运行 `python example_usage_new.py` 查看示例

---

**祝您运行顺利！** 🚀


# 模型输入输出详解

## 📊 核心概念：模型的两个阶段

所有模型都有两个独立的阶段：

### 1️⃣ **训练阶段（Training）**：学习参数
### 2️⃣ **预测阶段（Prediction）**：输出预测值

---

## 🔄 完整流程图

```
训练数据 → fit() → 模型参数 (保存在内存)
                    ↓
测试数据 → predict() → 预测的旅行时间（秒）
```

---

## 📥 输入（Input）

### 所有模型的输入格式：**FinalData DataFrame**

**必需列**：
- `fused_tt_15min`：真实旅行时间（秒）【训练时使用】
- `v_over_c`：V/C比
- `t0_ff`：自由流行程时间（秒）
- `flow_veh_hr`：小时流量（veh/hr）
- `capacity`：容量（veh/hr）

**可选列**（部分模型使用）：
- `hgv_share`：HGV份额
- `is_peak`：是否高峰时段
- `is_raining`：是否降雨
- `temperature`：温度
- `hour`, `weekday`, `daytype`：时间特征

---

## 📤 输出（Output）

### 1️⃣ 训练阶段 `fit()` 的输出

**返回值**：`self`（模型对象本身，支持链式调用）

**副作用**：在模型内部保存参数

#### BPR模型（M0-M4）保存的参数：
```python
{
    'alpha': 0.8244,        # BPR参数α
    'beta': 1.0,            # BPR参数β
    't0': 94.02,            # 自由流时间（秒）
    'covariance': [[...]]   # 参数协方差（NLS方法）
}
```

#### ML模型（M2/M4的tree/rf/gbdt，M5）保存的参数：
```python
{
    't0': 94.02,                    # 自由流时间
    'mode': 'direct',               # 模式：direct或residual
    'feature_importance': {...}     # 特征重要性（树模型）
}
```

**查看方式**：
```python
model.fit(df_train, method='nls')
info = model.info()  # 获取所有参数
print(info)
```

---

### 2️⃣ 预测阶段 `predict()` 的输出

**返回值**：`numpy.ndarray`（一维数组）

**内容**：预测的旅行时间（秒）

**示例**：
```python
predictions = model.predict(df_test)
# array([108.61, 109.40, 108.89, ...])  # 单位：秒
```

---

## 📊 不同模型的输出对比

### BPR模型（M0, M1, M3）

| 阶段 | 输出 | 说明 |
|------|------|------|
| **训练** | `{'alpha': α, 'beta': β, 't0': t0}` | 拟合BPR公式参数 |
| **预测** | `[94.03, 94.84, 108.61, ...]` | 行程时间（秒） |

**预测公式**：
```
T_pred = t0 * (1 + α * (V/C)^β)
```

---

### ML直接预测模型（M2, M4）

| 阶段 | 输出 | 说明 |
|------|------|------|
| **训练** | `{'t0': t0, 'mode': 'direct'}` | 训练ML模型学习 t/t0 比值 |
| **预测** | `[101.82, 102.75, 104.13, ...]` | 行程时间（秒） |

**预测公式**：
```
ratio = ML_model(V/C, hour, is_peak, ...)
T_pred = ratio * t0
```

---

### ML混合模型（M5）

| 阶段 | 输出 | 说明 |
|------|------|------|
| **训练** | `{'base_bpr': {...}, 'ml_model': {...}}` | 两阶段：BPR基础 + ML残差 |
| **预测** | `[102.34, 103.53, 104.45, ...]` | 行程时间（秒） |

**预测公式**：
```
T_bpr = t0 * (1 + α * (V/C)^β)     # 第一阶段：BPR基础
residual = ML_model(features)       # 第二阶段：ML残差
T_pred = T_bpr + residual           # 最终预测
```

---

## 💾 保存到文件的输出

### 1. 预测结果文件：`M67_115030402_predictions.csv`

**内容**：所有模型的预测值（秒）

| 列名 | 说明 |
|------|------|
| `datetime` | 时间戳 |
| `fused_tt_15min` | 真实值（Ground Truth，秒） |
| `v_over_c` | V/C比 |
| `M0_classical_pred` | M0模型（classical方法）的预测（秒） |
| `M0_loglinear_pred` | M0模型（loglinear方法）的预测（秒） |
| `M0_nls_pred` | M0模型（NLS方法）的预测（秒） |
| `M1_classical_pred` | M1模型（classical方法）的预测（秒） |
| ... | ... |

**示例行**：
```
2024-09-25 09:30:00, 106.07, 0.188, 94.03, 94.83, 108.61, ...
```
含义：
- 真实旅行时间：106.07秒
- V/C比：0.188
- M0_classical预测：94.03秒
- M0_loglinear预测：94.83秒
- M0_nls预测：108.61秒（最接近真实值！）

---

### 2. 对比表文件：`M67_115030402_comparison.csv`

**内容**：每个模型的评估指标

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| M5_svr | 14.92 | 33.16 | 9.05 | -0.12 |
| M0_nls | 19.28 | 35.15 | 13.14 | -0.26 |
| ... | ... | ... | ... | ... |

**说明**：
- MAE, RMSE：单位为秒
- MAPE：单位为百分比
- R²：拟合优度

---

### 3. 评估报告文件：`M67_115030402_report.txt`

**内容**：详细的评估结果

```
================================================================================
BPR 基准测试评估报告
================================================================================

1. 模型性能对比
--------------------------------------------------------------------------------
       Model       MAE       RMSE      MAPE         R2
      M5_svr 14.922778  33.157719  9.053235  -0.123886
      M0_nls 19.278374  35.147311 13.142513  -0.262808
      ...

3. 分层分析（按V/C比）
--------------------------------------------------------------------------------
M0_nls:
        Layer  Count  VCR_Mean       MAE      RMSE      MAPE        R2
0 ≤ V/C < 0.6    634  0.116854 19.278374 35.147311 13.142513 -0.262808
...
```

---

## 🎯 总结：输出的核心要点

### ✅ 训练阶段输出
- **BPR模型**：输出 `α, β` 参数
- **ML模型**：不输出明确参数（保存在sklearn模型内部）
- **M5混合**：输出 BPR参数 + ML模型

### ✅ 预测阶段输出
- **所有模型**：统一输出**预测的旅行时间（秒）**
- **不是**：不输出 V/C 比、速度、或其他中间变量
- **格式**：numpy数组，长度 = 测试集样本数

### ✅ 保存到文件
- **预测CSV**：每个模型的预测值（秒）
- **对比CSV**：每个模型的MAE/RMSE/MAPE/R²
- **报告TXT**：详细分析和分层评估

---

## 📝 代码示例

### 完整流程

```python
# 1. 准备数据
from utils.data import build_finaldata
df = build_finaldata(link_id=115030402, ...)

# 2. 分割数据
train_mask = df['datetime'] < '2024-09-25'
df_train = df[train_mask]
df_test = df[~train_mask]

# 3. 训练模型
from models.m0_bpr_new import M0_BPR
model = M0_BPR()
model.fit(df_train, method='nls')

# 4. 查看参数（训练阶段输出）
info = model.info()
print(info)
# {'model': 'M0_BPR', 'method': 'nls', 
#  'estimator_info': {'params': {'alpha': 0.82, 'beta': 1.0, 't0': 94.02}}}

# 5. 预测（预测阶段输出）
y_pred = model.predict(df_test)
print(y_pred)
# array([108.61, 109.40, 108.89, ...])  # 单位：秒

# 6. 评估
from utils.metrics import calculate_metrics
y_true = df_test['fused_tt_15min'].values
mae = calculate_metrics(y_true, y_pred)['mae']
print(f"MAE: {mae:.2f} 秒")
# MAE: 19.28 秒
```

---

## 🔍 常见问题

### Q1: 模型输出的是什么单位？
**A**: **秒（seconds）**。所有模型预测的都是旅行时间（秒）。

### Q2: 如何获取BPR参数α和β？
**A**: 调用 `model.info()` 或查看训练时的终端输出。

### Q3: ML模型有没有参数输出？
**A**: ML模型（如随机森林）的参数保存在sklearn模型内部（如树结构），可通过 `estimator.get_feature_importance()` 查看特征重要性。

### Q4: 预测结果保存在哪里？
**A**: `outputs/M67_115030402/M67_115030402_predictions.csv`

### Q5: 如何对比不同模型？
**A**: 查看 `M67_115030402_comparison.csv`（对比MAE）或 `M67_115030402_report.txt`（详细报告）。

---

## 🎊 结论

| 阶段 | 输入 | 输出 | 用途 |
|------|------|------|------|
| **训练** | `df_train` (FinalData) | 模型参数（内存） | 学习数据模式 |
| **预测** | `df_test` (FinalData) | **行程时间（秒）** | 评估模型性能 |

**最重要的输出**：`predict()` 返回的**预测旅行时间（秒）**，用于与真实值对比，计算MAE/RMSE等指标。


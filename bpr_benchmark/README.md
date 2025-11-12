# BPR 基准测试框架

这是一个完整的BPR（Bureau of Public Roads）函数基准测试框架，用于评估和比较不同的行程时间估计方法。

## 项目概述

本项目实现了9种不同的行程时间估计方法，涵盖：
- **传统BPR方法**：经典BPR、非线性最小二乘法、对数线性回归
- **机器学习方法**：SVR、决策树、随机森林、梯度提升、神经网络
- **可靠性方法**：贝叶斯回归

## 项目结构

```
bpr_benchmark/
├── configs/
│   └── default.yaml          # 配置文件（实验的"控制面板"）
├── utils/
│   ├── data.py              # 数据预处理模块（"Final Data"工厂）
│   └── metrics.py           # 评估指标计算模块
├── models/
│   ├── base.py              # 模型基类
│   ├── m0_bpr.py            # BPR参数估计器
│   ├── m5_ml.py             # 机器学习模型
│   └── m6_reliability.py    # 贝叶斯可靠性模型
├── outputs/                 # 输出结果目录
├── run_benchmark.py         # 主程序（总指挥）
├── requirements.txt         # 依赖包列表
└── README.md               # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
cd bpr_benchmark
pip install -r requirements.txt
```

### 2. 配置实验

编辑 `configs/default.yaml` 文件，设置：
- 数据文件路径
- 要测试的路段
- 要运行的模型
- 评估指标配置

### 3. 运行基准测试

```bash
python run_benchmark.py
```

### 4. 查看结果

结果将保存在 `outputs/` 目录下，包括：
- `*_comparison.csv` - 模型对比表
- `*_report.txt` - 详细评估报告
- `*_predictions.csv` - 预测结果
- `*.png` - 可视化图表

## 核心模块说明

### 1. 数据预处理 (`utils/data.py`)

**核心功能**：`create_final_dataset()`

这是整个框架的数据基础，实现了：
- 加载原始Precleaned数据
- 计算小时流量率：`V = 4Q`（15分钟流量 × 4）
- 计算加权平均速度
- 计算Ground Truth行程时间：`T = 3.6L/v`
- 计算自由流行程时间 `t_0`
- 计算V/C比
- 计算HGV份额（重型货车比例）
- 提取时段特征（高峰/非高峰）
- 合并天气数据（可选）

**数据分割**：使用"分块时间分割法"
- 训练集：第1-3周
- 测试集：第4周

### 2. 评估指标 (`utils/metrics.py`)

实现了以下评估指标：
- **MAE** (Mean Absolute Error) - 平均绝对误差
- **RMSE** (Root Mean Square Error) - 均方根误差
- **MAPE** (Mean Absolute Percentage Error) - 平均绝对百分比误差
- **R²** (R-squared) - 决定系数

支持分层评估：
- 按V/C比分组（自由流、中等拥堵、高拥堵、超容量）
- 按时段分组（高峰 vs 非高峰）

### 3. 模型库 (`models/`)

#### 3.1 BPR参数估计器 (`m0_bpr.py`)

**ClassicalBPR**
- 使用固定参数：α=0.15, β=4.0
- 无需训练
- 作为基线模型

**NLS_BPR**
- 使用非线性最小二乘法（`scipy.optimize.curve_fit`）
- 拟合α和β参数
- 最常用的BPR参数估计方法

**LogLinearBPR**
- 对BPR公式进行对数转换
- 使用线性回归
- 计算速度快，无需初始猜测

#### 3.2 机器学习模型 (`m5_ml.py`)

所有ML模型都使用完整特征集：
- V/C比（核心特征）
- HGV份额（M3：多类别）
- 时段特征（M1：动态参数）
- 天气特征（M4：外部因素）

**SVRModel** - 支持向量回归
- 使用RBF核
- 适合非线性关系

**DecisionTreeModel** - 决策树
- 可解释性强
- 可以捕捉交互效应

**RandomForestModel** - 随机森林
- 集成多个决策树
- 抗过拟合能力强

**GradientBoostingModel** - 梯度提升
- 逐步修正错误
- 通常表现最好

**NeuralNetworkModel** - 神经网络
- 多层感知机
- 学习复杂非线性模式

#### 3.3 可靠性模型 (`m6_reliability.py`)

**BayesianBPR** - 贝叶斯回归
- 提供预测不确定性估计
- 返回置信区间
- 适合可靠性分析

**QuantileBPR** - 分位数回归（可选）
- 估计特定分位数（如90%）
- 用于可靠性行程时间

### 4. 主程序 (`run_benchmark.py`)

执行流程：
1. 加载配置文件
2. 调用数据预处理
3. 循环遍历路段
4. 循环遍历模型
5. 训练和评估
6. 保存结果和可视化

## 配置文件说明

`configs/default.yaml` 是实验的控制面板，包含：

### 数据配置
```yaml
data:
  precleaned_file: "../Data/Precleaned_M67_Traffic_Data_September_2024.xlsx"
  weather_file: null
```

### 路段配置
```yaml
roads:
  M67_115030402:
    link_id: 115030402
    length_km: 2.7138037
    capacity_vph: 6649
```

### 模型配置
```yaml
models_to_run:
  - ClassicalBPR
  - NLS_BPR
  - LogLinearBPR
  - SVR
  - DecisionTree
  - RandomForest
  - GradientBoosting
  - NeuralNetwork
  - BayesianBPR
```

### 特征配置
```yaml
features:
  base:
    - V_C_Ratio
    - t_0
  m1_dynamic:      # M1: 动态参数
    - is_peak
    - hour
  m3_multiclass:   # M3: 多类别
    - p_H
  m4_external:     # M4: 外部因素
    - is_raining
```

## 核心概念

### BPR公式

标准BPR公式：

```
T = t_0 * (1 + α * (V/C)^β)
```

其中：
- `T` = 行程时间（秒）
- `t_0` = 自由流行程时间（秒）
- `V` = 流量（vehicles/hour）
- `C` = 容量（vehicles/hour）
- `α`, `β` = 待估计参数

### Ground Truth计算

本框架使用以下公式计算真实行程时间：

```
T = 3.6 * L / v
```

其中：
- `L` = 路段长度（km）
- `v` = 平均速度（km/h）
- `T` = 行程时间（秒）

这确保了流量、速度和行程时间在15分钟聚合窗口内的一致性。

### M1-M6 模型概念

- **M1 (动态参数)**：考虑时段变化（高峰/非高峰）
- **M2 (基本图VDF)**：基于交通基本图
- **M3 (多类别)**：考虑HGV（重型货车）影响
- **M4 (外部因素)**：考虑天气影响
- **M5 (ML混合)**：机器学习方法
- **M6 (可靠性)**：不确定性估计

## 输出结果

### 1. 对比表 (`*_comparison.csv`)

| Model | MAE | RMSE | MAPE | R2 |
|-------|-----|------|------|----|
| GradientBoosting | 12.34 | 18.56 | 8.45 | 0.92 |
| RandomForest | 13.21 | 19.34 | 9.12 | 0.91 |
| ... | ... | ... | ... | ... |

### 2. 评估报告 (`*_report.txt`)

包含：
- 模型性能对比
- 相对于基线的改进百分比
- 分层分析结果

### 3. 预测结果 (`*_predictions.csv`)

包含每个模型的预测值，用于进一步分析。

### 4. 可视化图表

- 预测 vs 真实值散点图
- 模型对比柱状图
- V/C比 vs 误差分析图

## 扩展和自定义

### 添加新模型

1. 在 `models/` 目录下创建新的模型类
2. 继承 `BaseModel` 或 `BPRModel`/`MLModel`
3. 实现 `fit()` 和 `predict()` 方法
4. 在 `models/base.py` 的 `create_model()` 函数中注册
5. 在配置文件中添加模型名称

### 添加新特征

1. 在 `utils/data.py` 的 `create_final_dataset()` 中计算新特征
2. 在配置文件中定义特征
3. 在 `models/base.py` 的 `prepare_features()` 中添加特征

### 添加新路段

1. 在配置文件的 `roads` 部分添加路段信息
2. 在 `roads_to_test` 中添加路段名称

## 常见问题

### Q1: 如何只运行部分模型？

编辑 `configs/default.yaml`，在 `models_to_run` 中只保留需要的模型。

### Q2: 如何调整模型超参数？

在 `configs/default.yaml` 的 `model_params` 部分修改。

### Q3: 数据文件路径错误怎么办？

确保 `configs/default.yaml` 中的 `precleaned_file` 路径正确，相对于 `bpr_benchmark/` 目录。

### Q4: 如何使用自己的数据？

确保数据包含以下列：
- LinkUID（路段ID）
- 流量列（FlowLane*Category*Value）
- 速度列（AverageSpeedLane*）
- 时间戳列（MeasurementStartUTC）

然后在配置文件中设置正确的路段参数（容量、长度）。

## 技术要求

- Python 3.8+
- pandas 1.5+
- numpy 1.23+
- scikit-learn 1.2+
- scipy 1.9+

## 引用

如果您使用本框架进行研究，请引用相关论文。

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请联系项目维护者。

---

**祝您使用愉快！**


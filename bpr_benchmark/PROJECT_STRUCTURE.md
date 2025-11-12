# BPR基准测试框架 - 项目结构详解

## 总体架构

```
用户 → run_benchmark.py (总指挥)
         ↓
         ├─→ configs/default.yaml (配置)
         ├─→ utils/data.py (数据预处理)
         ├─→ models/* (模型库)
         └─→ utils/metrics.py (评估)
              ↓
         outputs/ (结果输出)
```

## 详细文件说明

### 1. 配置层 (`configs/`)

```
configs/
└── default.yaml          # 实验配置文件（控制面板）
```

**职责**：
- 定义数据源路径
- 定义路段参数（容量、长度）
- 定义要测试的模型列表
- 定义特征集合
- 定义评估指标
- 定义模型超参数

**关键配置项**：
```yaml
data:                      # 数据源
  precleaned_file: "..."
  
roads:                     # 路段定义
  M67_115030402:
    link_id: 115030402
    length_km: 2.7138037
    capacity_vph: 6649
    
models_to_run:             # 模型列表
  - ClassicalBPR
  - NLS_BPR
  - ...
  
features:                  # 特征定义
  base: [V_C_Ratio, t_0]
  m1_dynamic: [is_peak, hour]
  m3_multiclass: [p_H]
  m4_external: [is_raining]
```

### 2. 数据层 (`utils/data.py`)

**核心函数**：

#### `create_final_dataset(link_id, precleaned_filepath, config)`
这是整个框架的数据基础！

**输入**：
- `link_id`: 路段ID（如115030402）
- `precleaned_filepath`: 数据文件路径
- `config`: 配置字典

**处理流程**：
```
原始数据 (Precleaned Excel)
  ↓
1. 筛选指定路段
  ↓
2. 计算V (小时流量): V = 4Q
  ↓
3. 计算v (平均速度): 加权平均
  ↓
4. 计算y (Ground Truth): T = 3.6L/v
  ↓
5. 计算t_0 (自由流时间): 低流量时的中位数速度
  ↓
6. 计算X1 (V/C比): V/C
  ↓
7. 计算X2 (HGV份额): p_H = HGV_Count / Total_Count
  ↓
8. 计算X3 (时段): is_peak, is_weekday
  ↓
9. 合并X4 (天气): is_raining, temperature
  ↓
Final Dataset (标准化数据)
```

**输出列**：
```python
[
    'timestamp',           # 时间戳
    'week', 'hour',       # 时间特征
    't_ground_truth',     # y (目标变量)
    't_0',                # 自由流时间
    'V_hourly_rate',      # 流量
    'v_avg_kmh',          # 速度
    'V_C_Ratio',          # X1: 核心特征
    'p_H',                # X2: HGV份额
    'is_peak',            # X3: 时段
    'is_weekday',         # X3: 工作日
    'is_raining',         # X4: 天气
    'temperature'         # X4: 温度
]
```

#### `split_data(df, config)`
分割训练/测试集

**方法**：分块时间分割法
- 训练集：第1-3周
- 测试集：第4周

#### `load_and_preprocess(config_path)`
主入口函数，处理所有路段

### 3. 模型层 (`models/`)

#### 3.1 基类 (`base.py`)

```python
BaseModel                  # 所有模型的抽象基类
  ├── fit(df_train)       # 训练接口
  └── predict(df_test)    # 预测接口

BPRModel(BaseModel)        # BPR模型基类
  ├── bpr_function()      # BPR公式实现
  └── get_params()        # 获取α, β参数

MLModel(BaseModel)         # 机器学习模型基类
  ├── prepare_features()  # 特征准备
  └── get_params()        # 获取模型参数

HybridModel(BaseModel)     # 混合模型基类
  ├── base_bpr            # BPR基础模型
  └── ml_model            # ML修正模型
```

**模型工厂**：
```python
create_model(model_name, config, t_0, capacity)
# 根据名称创建模型实例
```

#### 3.2 BPR参数估计器 (`m0_bpr.py`)

```python
ClassicalBPR(BPRModel)
  # 固定参数：α=0.15, β=4.0
  # 无需训练
  
NLS_BPR(BPRModel)
  # 非线性最小二乘法
  # 使用scipy.optimize.curve_fit
  # 拟合α和β
  
LogLinearBPR(BPRModel)
  # 对数线性回归
  # 转换为线性问题
  # 使用sklearn.LinearRegression
```

#### 3.3 机器学习模型 (`m5_ml.py`)

```python
SVRModel(MLModel)
  # 支持向量回归
  # 需要特征标准化
  
DecisionTreeModel(MLModel)
  # 决策树
  # 可解释性强
  
RandomForestModel(MLModel)
  # 随机森林
  # 集成学习
  # 提供特征重要性
  
GradientBoostingModel(MLModel)
  # 梯度提升
  # 通常表现最好
  # 提供特征重要性
  
NeuralNetworkModel(MLModel)
  # 神经网络
  # 多层感知机
  # 需要特征标准化
  
HybridBPR_ML(MLModel)
  # 混合模型（可选）
  # BPR + ML残差修正
```

#### 3.4 可靠性模型 (`m6_reliability.py`)

```python
BayesianBPR(BPRModel)
  # 贝叶斯回归
  # 提供不确定性估计
  # 返回置信区间
  
QuantileBPR(BPRModel)
  # 分位数回归（可选）
  # 估计特定分位数
  # 用于可靠性分析
```

### 4. 评估层 (`utils/metrics.py`)

**核心函数**：

```python
calculate_all_metrics(y_true, y_pred)
  # 计算MAE, RMSE, MAPE, R²
  
calculate_stratified_metrics(y_true, y_pred, vcr, vcr_bins)
  # 按V/C比分层评估
  # 分组：[0-0.3], [0.3-0.7], [0.7-1.0], [1.0+]
  
calculate_metrics_by_time_period(y_true, y_pred, is_peak)
  # 按时段评估
  # 分组：高峰 vs 非高峰
  
create_metrics_comparison_table(results_dict)
  # 创建模型对比表
  
generate_evaluation_report(results_dict, output_path)
  # 生成完整评估报告
```

### 5. 主程序 (`run_benchmark.py`)

**执行流程**：

```python
main():
    # 1. 加载配置
    config = load_config()
    
    # 2. 加载和预处理数据
    all_data = load_and_preprocess(config)
    
    # 3. 对每条路段
    for road_name in config['roads_to_test']:
        df_final = all_data[road_name]
        
        # 4. 分割数据
        df_train, df_test = split_data(df_final, config)
        
        # 5. 对每个模型
        for model_name in config['models_to_run']:
            # 5.1 创建模型
            model = create_model(model_name, ...)
            
            # 5.2 训练
            model.fit(df_train)
            
            # 5.3 预测
            y_pred = model.predict(df_test)
            
            # 5.4 评估
            metrics = calculate_all_metrics(y_true, y_pred)
            
            # 5.5 保存结果
            results[model_name] = metrics
        
        # 6. 保存结果
        save_results(results, output_dir)
        
        # 7. 创建可视化
        create_visualizations(results, df_test, output_dir)
```

### 6. 输出层 (`outputs/`)

**目录结构**：
```
outputs/
└── M67_115030402/                    # 路段名称
    ├── M67_115030402_comparison.csv  # 模型对比表
    ├── M67_115030402_report.txt      # 详细评估报告
    ├── M67_115030402_predictions.csv # 预测结果
    ├── M67_115030402_*_stratified.csv # 各模型分层结果
    ├── M67_115030402_predictions_scatter.png  # 散点图
    ├── M67_115030402_comparison_bars.png      # 对比柱状图
    └── M67_115030402_error_analysis.png       # 误差分析图
```

## 数据流图

```
Excel数据
   ↓
[utils/data.py]
   ├─ create_final_dataset()
   │    ├─ 计算V (流量)
   │    ├─ 计算v (速度)
   │    ├─ 计算T (Ground Truth)
   │    ├─ 计算t_0 (自由流时间)
   │    ├─ 计算V/C比
   │    ├─ 计算HGV份额
   │    └─ 提取时段特征
   │
   └─ split_data()
        ├─ df_train (第1-3周)
        └─ df_test (第4周)
             ↓
[models/*]
   ├─ model.fit(df_train)
   └─ y_pred = model.predict(df_test)
             ↓
[utils/metrics.py]
   ├─ calculate_all_metrics()
   ├─ calculate_stratified_metrics()
   └─ generate_evaluation_report()
             ↓
[outputs/]
   ├─ CSV文件
   └─ PNG图表
```

## 核心公式

### BPR公式
```
T = t_0 * (1 + α * (V/C)^β)
```

### Ground Truth计算
```
T = 3.6 * L / v
```
其中：
- L: 路段长度 (km)
- v: 平均速度 (km/h)
- T: 行程时间 (秒)

### 流量计算
```
V = 4 * Q
```
其中：
- Q: 15分钟流量
- V: 小时流量率

### HGV份额计算
```
p_H = HGV_Count / Total_Count
```

## 扩展点

### 1. 添加新模型
位置：`models/` 目录
步骤：
1. 创建新类，继承 `BaseModel`
2. 实现 `fit()` 和 `predict()`
3. 在 `base.py` 中注册

### 2. 添加新特征
位置：`utils/data.py` 的 `create_final_dataset()`
步骤：
1. 在数据处理流程中计算新特征
2. 添加到返回的DataFrame中
3. 在 `base.py` 的 `prepare_features()` 中使用

### 3. 添加新评估指标
位置：`utils/metrics.py`
步骤：
1. 实现新的指标计算函数
2. 在 `calculate_all_metrics()` 中调用
3. 在报告生成中展示

### 4. 添加新路段
位置：`configs/default.yaml`
步骤：
1. 在 `roads` 部分添加路段信息
2. 在 `roads_to_test` 中添加路段名称

## 团队协作建议

### 角色分工

1. **数据工程师**：负责 `utils/data.py`
   - 确保数据质量
   - 添加新特征
   - 处理缺失值

2. **模型工程师**：负责 `models/`
   - 实现新模型
   - 调优超参数
   - 提高预测精度

3. **评估工程师**：负责 `utils/metrics.py`
   - 添加新指标
   - 生成报告
   - 创建可视化

4. **系统工程师**：负责 `run_benchmark.py`
   - 优化流程
   - 并行化
   - 错误处理

### 开发流程

1. **开发新功能**：在对应模块中实现
2. **单元测试**：每个模块的 `if __name__ == "__main__"` 部分
3. **集成测试**：运行 `run_benchmark.py`
4. **代码审查**：确保符合框架规范
5. **文档更新**：更新 README 和注释

## 性能优化建议

1. **数据层**：
   - 使用 `pandas` 的向量化操作
   - 避免循环
   - 使用 `numba` 加速关键函数

2. **模型层**：
   - 使用 `n_jobs=-1` 并行训练
   - 缓存中间结果
   - 使用 GPU（如果可用）

3. **评估层**：
   - 批量计算指标
   - 避免重复计算

4. **主程序**：
   - 使用 `multiprocessing` 并行处理多个路段
   - 使用 `tqdm` 显示进度

## 常见问题排查

### 问题1：数据加载失败
- 检查文件路径
- 检查Excel格式
- 检查列名是否正确

### 问题2：模型训练失败
- 检查数据是否有缺失值
- 检查参数边界是否合理
- 查看错误堆栈

### 问题3：评估结果异常
- 检查Ground Truth计算是否正确
- 检查V/C比是否合理
- 检查是否有异常值

### 问题4：内存不足
- 减少数据量（采样）
- 分批处理
- 使用更高效的数据结构

---

**这个框架是模块化、可扩展的，欢迎贡献！**


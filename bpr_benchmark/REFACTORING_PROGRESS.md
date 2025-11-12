# BPR框架重构进度

## 🎯 重构目标

将现有框架升级为更加**工程化、模块化、可扩展**的架构：
- **解耦**：模型形态（M1-M6）与参数估计方法（九法）分离
- **标准化**：统一的FinalData接口
- **工程化**：pipelines层实现一键评测

## ✅ 已完成（P0部分）

### P0-1: utils/data.py 工程化函数 ✅
- [x] `build_finaldata()` - 统一的FinalData构建接口
  - 支持任意LinkID
  - 标准化列名（flow_veh_hr, capacity, fused_tt_15min, t0_ff, v_over_c等）
  - Winsorize异常值处理
  - 多种t0估计策略（min5pct, low_volume）
- [x] `finaldata_qc_report()` - 质量控制报告
  - 基本统计、流量统计、行程时间统计
  - HGV统计、时间覆盖
  - 自动标记WARNING/PASS状态

### P0-2: estimators/ 目录和基类 ✅
- [x] `estimators/__init__.py`
- [x] `estimators/base_estimator.py`
  - `BaseEstimator` - 所有估计器的抽象基类
  - `BPREstimator` - BPR类估计器基类
  - `MLEstimator` - 机器学习估计器基类
  - `create_estimator()` - 估计器工厂函数

## 🚧 进行中（P0剩余）

### P0-3: 实现3个BPR估计器
- [ ] `estimators/bpr_classical.py` - 经典BPR（α=0.15, β=4.0）
- [ ] `estimators/bpr_loglinear.py` - 对数线性回归
- [ ] `estimators/bpr_nls.py` - 非线性最小二乘法

### P0-4: 重构models/m0_bpr.py
- [ ] 改为"BPR主体 + 可插拔估计器"
- [ ] 从estimators取α、β、t0
- [ ] 统一fit(df, *, method, config)接口

### P0-5: 创建pipelines/目录
- [ ] `pipelines/build_finaldata.py` - CLI工具
- [ ] `pipelines/train_eval.py` - 训练+评测，输出MAE表
- [ ] `pipelines/registry.py` - 模型/估计器注册表

### P0-6: 更新configs/default.yaml
- [ ] 添加builder配置（t0_strategy, winsor, vc_bins）
- [ ] 添加train配置（split, filters）
- [ ] 添加methods和models列表

## 📋 待完成（P1-P3）

### P1: 核心模型扩展
- [ ] `models/m1_dp_bpr.py` - 动态参数（分时段）
- [ ] `estimators/{svr,tree,rf,gbdt,nn}.py` - ML估计器
- [ ] `models/m5_ml_hbpr.py` - BPR+残差混合模型

### P2: 高级模型
- [ ] `models/m2_fd_vdf.py` - 基本图VDF
- [ ] `models/m3_mc_bpr.py` - 多类别（HGV等效流量）
- [ ] `models/m4_ef_bpr.py` - 外部因素

### P3: 可靠性和测试
- [ ] `models/m6_sc_bpr.py` - 可靠性模型重构
- [ ] 更新test_framework.py
- [ ] 更新example_usage.py

## 🎨 新架构特点

### 1. 标准化数据流
```
原始数据 → build_finaldata() → FinalData (标准列名)
                                    ↓
                            estimators/ (九法)
                                    ↓
                            models/ (M0-M6)
                                    ↓
                            pipelines/train_eval
                                    ↓
                            MAE表 (行=模型, 列=方法)
```

### 2. 解耦设计
```
模型形态（M1-M6）      估计方法（九法）
     M0_BPR      ×    {classical, loglinear, nls}
     M1_DP_BPR   ×    {classical, loglinear, nls}
     M5_ML_HBPR  ×    {svr, tree, rf, gbdt, nn}
     ...
```

### 3. 统一接口
所有模型：
```python
model.fit(df_train, *, method='nls', config=...)
y_pred = model.predict(df_test)
info = model.info()  # 返回 {t0, alpha, beta, ...}
```

所有估计器：
```python
estimator.fit(df, *, t0=100)
y_pred = estimator.predict(df)
info = estimator.info()
```

## 📊 预期输出

### MAE对比表（行=模型，列=方法）
|  | classical | loglinear | nls | svr | tree | rf | gbdt | bayes | nn |
|---|---|---|---|---|---|---|---|---|---|
| M0_BPR | 20.5 | 18.3 | 17.8 | - | - | - | - | - | - |
| M1_DP_BPR | 19.2 | 17.1 | 16.5 | - | - | - | - | - | - |
| M2_FD_VDF | - | - | 15.8 | - | - | - | - | - | - |
| M3_MC_BPR | 18.7 | 16.9 | 16.2 | - | - | - | - | - | - |
| M4_EF_BPR | 18.3 | 16.5 | 15.9 | - | - | - | - | - | - |
| M5_ML_HBPR | - | - | - | 14.2 | 15.1 | 13.5 | 12.8 | 14.8 | 13.9 |
| M6_SC_BPR | - | - | - | - | - | - | - | 15.2 | - |

## 🔄 迁移策略

### 保留的文件（最小改动）
- `utils/data.py` - ✅ 已添加新函数，保留原有功能
- `utils/metrics.py` - 保留，后续补充by_vc_bins和可靠性指标
- `models/base.py` - 保留，后续添加统一入口

### 新增的文件
- `estimators/` - ✅ 已创建基类
- `pipelines/` - 待创建
- `models/m1_dp_bpr.py` ~ `m4_ef_bpr.py` - 待创建

### 重构的文件
- `models/m0_bpr.py` - 改为使用estimators
- `models/m5_ml.py` → `models/m5_ml_hbpr.py` - 改为BPR+残差
- `models/m6_reliability.py` → `models/m6_sc_bpr.py` - 统一接口

## 📝 下一步行动

1. **立即完成P0-3**: 实现3个BPR估计器（30分钟）
2. **完成P0-4**: 重构M0_BPR使用estimators（20分钟）
3. **完成P0-5**: 创建pipelines/（40分钟）
4. **完成P0-6**: 更新配置文件（10分钟）
5. **测试P0**: 运行M0×3法，验证MAE表输出（10分钟）

**预计P0完成时间**: 2小时

---

**当前状态**: 🟢 进展顺利  
**完成度**: P0 40% | 总体 15%  
**最后更新**: 2024-11-12


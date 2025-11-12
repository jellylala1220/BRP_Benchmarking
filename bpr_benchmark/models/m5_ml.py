"""
M5: 机器学习模型

实现五种机器学习方法：
1. SVR - 支持向量回归
2. DecisionTree - 决策树
3. RandomForest - 随机森林
4. GradientBoosting - 梯度提升
5. NeuralNetwork - 神经网络

这些模型将使用所有可用特征（V/C比、HGV份额、时段、天气等）
来击败简单的BPR模型

对应报告中的 M5 (ML Hybrid BPR)
"""

import numpy as np
import pandas as pd
from typing import Dict

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .base import MLModel


class SVRModel(MLModel):
    """
    支持向量回归模型
    
    使用RBF核的SVR
    适合处理非线性关系
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        super().__init__(config, t_0, capacity)
        
        # 从配置中获取超参数
        params = config.get('model_params', {}).get('SVR', {})
        
        self.model = SVR(
            kernel=params.get('kernel', 'rbf'),
            C=params.get('C', 1.0),
            epsilon=params.get('epsilon', 0.1)
        )
        
        # SVR需要特征标准化
        self.scaler = StandardScaler()
    
    def fit(self, df_train: pd.DataFrame) -> 'SVRModel':
        """训练SVR模型"""
        
        # 准备特征
        X_train = self.prepare_features(df_train, feature_set='full')
        y_train = df_train['t_ground_truth'].values
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 训练模型
        self.model.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 fit()")
        
        X_test = self.prepare_features(df_test, feature_set='full')
        X_test_scaled = self.scaler.transform(X_test)
        
        return self.model.predict(X_test_scaled)


class DecisionTreeModel(MLModel):
    """
    决策树回归模型
    
    简单但可解释性强的模型
    可以捕捉非线性关系和交互效应
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        super().__init__(config, t_0, capacity)
        
        # 从配置中获取超参数
        params = config.get('model_params', {}).get('DecisionTree', {})
        
        self.model = DecisionTreeRegressor(
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 20),
            min_samples_leaf=params.get('min_samples_leaf', 10),
            random_state=params.get('random_state', 42)
        )
    
    def fit(self, df_train: pd.DataFrame) -> 'DecisionTreeModel':
        """训练决策树模型"""
        
        X_train = self.prepare_features(df_train, feature_set='full')
        y_train = df_train['t_ground_truth'].values
        
        self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 fit()")
        
        X_test = self.prepare_features(df_test, feature_set='full')
        return self.model.predict(X_test)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.is_fitted:
            return {}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return importance


class RandomForestModel(MLModel):
    """
    随机森林回归模型
    
    集成多个决策树
    通常比单个决策树更稳定、准确
    对异常值和过拟合有较好的抵抗力
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        super().__init__(config, t_0, capacity)
        
        # 从配置中获取超参数
        params = config.get('model_params', {}).get('RandomForest', {})
        
        self.model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 20),
            min_samples_leaf=params.get('min_samples_leaf', 10),
            random_state=params.get('random_state', 42),
            n_jobs=-1  # 使用所有CPU核心
        )
    
    def fit(self, df_train: pd.DataFrame) -> 'RandomForestModel':
        """训练随机森林模型"""
        
        X_train = self.prepare_features(df_train, feature_set='full')
        y_train = df_train['t_ground_truth'].values
        
        self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 fit()")
        
        X_test = self.prepare_features(df_test, feature_set='full')
        return self.model.predict(X_test)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.is_fitted:
            return {}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return importance


class GradientBoostingModel(MLModel):
    """
    梯度提升回归模型
    
    逐步构建模型，每次添加一个新的树来修正之前的错误
    通常是表现最好的传统机器学习方法之一
    
    对应报告中的 M5 (ML Hybrid BPR) 的主要实现
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        super().__init__(config, t_0, capacity)
        
        # 从配置中获取超参数
        params = config.get('model_params', {}).get('GradientBoosting', {})
        
        self.model = GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 5),
            min_samples_split=params.get('min_samples_split', 20),
            min_samples_leaf=params.get('min_samples_leaf', 10),
            subsample=params.get('subsample', 0.8),
            random_state=params.get('random_state', 42)
        )
    
    def fit(self, df_train: pd.DataFrame) -> 'GradientBoostingModel':
        """训练梯度提升模型"""
        
        X_train = self.prepare_features(df_train, feature_set='full')
        y_train = df_train['t_ground_truth'].values
        
        self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 fit()")
        
        X_test = self.prepare_features(df_test, feature_set='full')
        return self.model.predict(X_test)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.is_fitted:
            return {}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return importance


class NeuralNetworkModel(MLModel):
    """
    神经网络回归模型
    
    使用多层感知机(MLP)
    可以学习复杂的非线性模式
    
    对应报告中的深度学习方法
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        super().__init__(config, t_0, capacity)
        
        # 从配置中获取超参数
        params = config.get('model_params', {}).get('NeuralNetwork', {})
        
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(params.get('hidden_layer_sizes', [64, 32, 16])),
            activation=params.get('activation', 'relu'),
            solver=params.get('solver', 'adam'),
            alpha=params.get('alpha', 0.0001),
            learning_rate=params.get('learning_rate', 'adaptive'),
            max_iter=params.get('max_iter', 1000),
            random_state=params.get('random_state', 42),
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # 神经网络需要特征标准化
        self.scaler = StandardScaler()
    
    def fit(self, df_train: pd.DataFrame) -> 'NeuralNetworkModel':
        """训练神经网络模型"""
        
        X_train = self.prepare_features(df_train, feature_set='full')
        y_train = df_train['t_ground_truth'].values
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 训练模型
        self.model.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 fit()")
        
        X_test = self.prepare_features(df_test, feature_set='full')
        X_test_scaled = self.scaler.transform(X_test)
        
        return self.model.predict(X_test_scaled)


# ========== 混合模型（可选实现）==========

class HybridBPR_ML(MLModel):
    """
    混合BPR-ML模型（可选）
    
    实现"残差修正法"（对应报告 Eq. 4.13）：
    1. 先用NLS_BPR拟合基础模型
    2. 用ML模型学习残差
    3. 最终预测 = BPR预测 * (1 + ML残差预测)
    
    这是报告中M5的另一种实现方式
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float, ml_model_class=GradientBoostingRegressor):
        super().__init__(config, t_0, capacity)
        
        # 导入NLS_BPR
        from .m0_bpr import NLS_BPR
        
        self.base_bpr = NLS_BPR(config, t_0, capacity)
        self.ml_model = ml_model_class(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    def fit(self, df_train: pd.DataFrame) -> 'HybridBPR_ML':
        """训练混合模型"""
        
        # 步骤1: 训练基础BPR模型
        self.base_bpr.fit(df_train)
        
        # 步骤2: 计算BPR预测和残差
        T_bpr_pred = self.base_bpr.predict(df_train)
        T_true = df_train['t_ground_truth'].values
        
        # 残差因子: r = (T_true / T_bpr) - 1
        residual_factor = (T_true / T_bpr_pred) - 1
        
        # 步骤3: 用ML模型学习残差
        X_train = self.prepare_features(df_train, feature_set='full')
        self.ml_model.fit(X_train, residual_factor)
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 fit()")
        
        # BPR基础预测
        T_bpr_pred = self.base_bpr.predict(df_test)
        
        # ML残差预测
        X_test = self.prepare_features(df_test, feature_set='full')
        residual_pred = self.ml_model.predict(X_test)
        
        # 最终预测
        T_final = T_bpr_pred * (1 + residual_pred)
        
        return T_final


if __name__ == "__main__":
    """测试机器学习模型"""
    
    # 生成模拟数据
    np.random.seed(42)
    n = 1000
    
    # 生成特征
    vcr = np.random.uniform(0.1, 1.2, n)
    p_h = np.random.uniform(0, 0.3, n)
    is_peak = np.random.choice([0, 1], n)
    hour = np.random.randint(0, 24, n)
    
    # 生成目标（复杂的非线性关系）
    t_0 = 100
    t_true = t_0 * (1 + 0.2 * np.power(vcr, 3.5))
    t_true *= (1 + 0.1 * p_h)  # HGV效应
    t_true *= (1 + 0.05 * is_peak)  # 高峰效应
    t_true += np.random.normal(0, 5, n)  # 噪声
    
    # 创建DataFrame
    df = pd.DataFrame({
        'V_C_Ratio': vcr,
        't_ground_truth': t_true,
        't_0': t_0,
        'p_H': p_h,
        'is_peak': is_peak,
        'is_weekday': 1,
        'hour': hour,
        'is_raining': 0,
        'temperature': 20
    })
    
    # 分割数据
    split_idx = int(0.8 * n)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    config = {
        'model_params': {
            'RandomForest': {'n_estimators': 50, 'max_depth': 10, 'random_state': 42},
            'GradientBoosting': {'n_estimators': 50, 'learning_rate': 0.1, 'random_state': 42}
        }
    }
    
    # 测试模型
    models = [
        ('RandomForest', RandomForestModel(config, t_0, 6000)),
        ('GradientBoosting', GradientBoostingModel(config, t_0, 6000)),
        ('NeuralNetwork', NeuralNetworkModel(config, t_0, 6000))
    ]
    
    for name, model in models:
        # 训练
        model.fit(df_train)
        
        # 预测
        y_pred = model.predict(df_test)
        y_true = df_test['t_ground_truth'].values
        
        # 评估
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        print(f"{name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        # 特征重要性（如果有）
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            print(f"  特征重要性: {importance}")
        
        print()


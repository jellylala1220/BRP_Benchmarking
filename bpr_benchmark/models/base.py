"""
模型基类

定义所有BPR模型必须遵循的接口
确保所有模型都有统一的 fit() 和 predict() 方法
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any


class BaseModel(ABC):
    """
    所有BPR模型的抽象基类
    
    所有模型必须实现：
    1. fit(df_train) - 训练模型
    2. predict(df_test) - 预测
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        """
        初始化模型
        
        Args:
            config: 配置字典
            t_0: 自由流行程时间（秒）
            capacity: 路段容量（vehicles/hour）
        """
        self.config = config
        self.t_0 = t_0
        self.capacity = capacity
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, df_train: pd.DataFrame) -> 'BaseModel':
        """
        训练模型
        
        Args:
            df_train: 训练数据，必须包含以下列：
                - t_ground_truth: 真实行程时间（目标变量）
                - V_C_Ratio: 流量容量比
                - t_0: 自由流行程时间
                - 其他特征（根据模型需要）
                
        Returns:
            self (支持链式调用)
        """
        pass
    
    @abstractmethod
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        预测行程时间
        
        Args:
            df_test: 测试数据，必须包含与训练时相同的特征列
            
        Returns:
            预测的行程时间数组（秒）
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取模型参数
        
        Returns:
            参数字典
        """
        return {}
    
    def __repr__(self) -> str:
        """模型的字符串表示"""
        return f"{self.__class__.__name__}(t_0={self.t_0:.2f}, capacity={self.capacity})"


class BPRModel(BaseModel):
    """
    BPR模型的基类
    
    提供BPR公式的通用实现：
    T = t_0 * (1 + α * (V/C)^β)
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        super().__init__(config, t_0, capacity)
        self.alpha = None
        self.beta = None
    
    def bpr_function(self, vcr: np.ndarray, t_0: float = None, 
                     alpha: float = None, beta: float = None) -> np.ndarray:
        """
        BPR函数
        
        T = t_0 * (1 + α * (V/C)^β)
        
        Args:
            vcr: V/C比
            t_0: 自由流行程时间（如果为None，使用self.t_0）
            alpha: α参数（如果为None，使用self.alpha）
            beta: β参数（如果为None，使用self.beta）
            
        Returns:
            预测的行程时间
        """
        if t_0 is None:
            t_0 = self.t_0
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        
        if alpha is None or beta is None:
            raise ValueError("必须先拟合模型或提供 alpha 和 beta 参数")
        
        # BPR公式
        vcr = np.array(vcr)
        vcr = np.maximum(vcr, 0)  # 确保非负
        
        travel_time = t_0 * (1 + alpha * np.power(vcr, beta))
        
        return travel_time
    
    def get_params(self) -> Dict[str, Any]:
        """获取BPR参数"""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            't_0': self.t_0,
            'capacity': self.capacity
        }
    
    def __repr__(self) -> str:
        if self.alpha is not None and self.beta is not None:
            return f"{self.__class__.__name__}(α={self.alpha:.4f}, β={self.beta:.4f})"
        else:
            return f"{self.__class__.__name__}(未拟合)"


class MLModel(BaseModel):
    """
    机器学习模型的基类
    
    提供特征准备的通用方法
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        super().__init__(config, t_0, capacity)
        self.feature_names = None
        self.model = None
    
    def prepare_features(self, df: pd.DataFrame, 
                        feature_set: str = 'full') -> np.ndarray:
        """
        准备特征矩阵
        
        Args:
            df: 数据DataFrame
            feature_set: 特征集合
                - 'base': 只有V_C_Ratio
                - 'full': 所有可用特征
                
        Returns:
            特征矩阵
        """
        
        if feature_set == 'base':
            # 基础特征：只有V/C比
            features = ['V_C_Ratio']
        
        elif feature_set == 'full':
            # 完整特征集：包含所有M1-M4的特征
            features = ['V_C_Ratio']
            
            # 添加可用的特征
            optional_features = [
                't_0',           # 自由流时间
                'p_H',           # HGV份额 (M3)
                'is_peak',       # 高峰标志 (M1)
                'is_weekday',    # 工作日标志 (M1)
                'hour',          # 小时 (M1)
                'is_raining',    # 天气 (M4)
                'temperature'    # 温度 (M4)
            ]
            
            for feat in optional_features:
                if feat in df.columns:
                    features.append(feat)
        
        else:
            raise ValueError(f"不支持的特征集: {feature_set}")
        
        # 保存特征名称
        self.feature_names = features
        
        # 提取特征矩阵
        X = df[features].values
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)
        
        return X
    
    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        params = {
            'feature_names': self.feature_names,
            't_0': self.t_0,
            'capacity': self.capacity
        }
        
        # 如果模型有参数，也返回
        if hasattr(self.model, 'get_params'):
            params['model_params'] = self.model.get_params()
        
        return params


class HybridModel(BaseModel):
    """
    混合模型的基类
    
    结合BPR和机器学习的混合方法
    对应 M5 (ML Hybrid BPR)
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        super().__init__(config, t_0, capacity)
        self.base_bpr = None
        self.ml_model = None
    
    def get_params(self) -> Dict[str, Any]:
        """获取混合模型参数"""
        params = {
            't_0': self.t_0,
            'capacity': self.capacity
        }
        
        if self.base_bpr is not None:
            params['bpr_params'] = self.base_bpr.get_params()
        
        if self.ml_model is not None and hasattr(self.ml_model, 'get_params'):
            params['ml_params'] = self.ml_model.get_params()
        
        return params


def create_model(model_name: str, config: Dict, t_0: float, capacity: float) -> BaseModel:
    """
    模型工厂函数
    
    根据模型名称创建相应的模型实例
    
    Args:
        model_name: 模型名称
        config: 配置字典
        t_0: 自由流行程时间
        capacity: 路段容量
        
    Returns:
        模型实例
    """
    
    # 延迟导入以避免循环依赖
    from .m0_bpr import ClassicalBPR, NLS_BPR, LogLinearBPR
    from .m5_ml import SVRModel, DecisionTreeModel, RandomForestModel, GradientBoostingModel, NeuralNetworkModel
    from .m6_reliability import BayesianBPR
    
    model_map = {
        'ClassicalBPR': ClassicalBPR,
        'NLS_BPR': NLS_BPR,
        'LogLinearBPR': LogLinearBPR,
        'SVR': SVRModel,
        'DecisionTree': DecisionTreeModel,
        'RandomForest': RandomForestModel,
        'GradientBoosting': GradientBoostingModel,
        'NeuralNetwork': NeuralNetworkModel,
        'BayesianBPR': BayesianBPR
    }
    
    if model_name not in model_map:
        raise ValueError(f"未知的模型: {model_name}. 可用模型: {list(model_map.keys())}")
    
    model_class = model_map[model_name]
    return model_class(config, t_0, capacity)


if __name__ == "__main__":
    """测试基类"""
    
    # 测试BPR函数
    from .m0_bpr import ClassicalBPR
    
    config = {}
    model = ClassicalBPR(config, t_0=100, capacity=6000)
    
    # 测试预测
    vcr = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    predictions = model.bpr_function(vcr, t_0=100, alpha=0.15, beta=4.0)
    
    print("V/C比:", vcr)
    print("预测行程时间:", predictions)
    print("模型:", model)


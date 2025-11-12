"""
估计器基类

职责：
定义所有参数估计方法的统一接口
将"模型形态（M1-M6）"与"参数估计方法（九法）"解耦

所有估计器都遵循统一的接口：
- fit(df, *, t0): 在FinalData上拟合参数
- predict(df): 预测行程时间
- info(): 返回估计的参数信息
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class BaseEstimator(ABC):
    """
    所有估计器的抽象基类
    
    统一接口确保所有估计方法可以无缝替换
    """
    
    def __init__(self):
        """初始化估计器"""
        self.is_fitted = False
        self.params = {}
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, *, t0: float = None) -> 'BaseEstimator':
        """
        拟合估计器
        
        Args:
            df: FinalData DataFrame，必须包含以下列：
                - flow_veh_hr: 小时流量（veh/hr）
                - capacity: 容量（veh/hr）
                - fused_tt_15min: 真实行程时间（秒）
                - t0_ff: 自由流行程时间（秒）（可选，如果提供t0参数则使用参数值）
            t0: 自由流行程时间（秒），如果为None则从df中获取
            
        Returns:
            self（支持链式调用）
        """
        pass
    
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        预测行程时间
        
        Args:
            df: FinalData DataFrame，必须包含与fit时相同的特征列
            
        Returns:
            预测的行程时间数组（秒）
        """
        pass
    
    def info(self) -> Dict[str, Any]:
        """
        获取估计器信息
        
        Returns:
            包含估计参数和元信息的字典
        """
        return {
            'estimator': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'params': self.params.copy()
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        if self.is_fitted:
            params_str = ', '.join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                  for k, v in self.params.items())
            return f"{self.__class__.__name__}({params_str})"
        else:
            return f"{self.__class__.__name__}(未拟合)"


class BPREstimator(BaseEstimator):
    """
    BPR估计器的基类
    
    提供BPR公式的通用实现
    T = t0 * (1 + α * (V/C)^β)
    """
    
    def __init__(self):
        super().__init__()
        self.alpha = None
        self.beta = None
        self.t0 = None
    
    def bpr_function(self, v_over_c: np.ndarray, t0: float = None, 
                     alpha: float = None, beta: float = None) -> np.ndarray:
        """
        BPR函数
        
        T = t0 * (1 + α * (V/C)^β)
        
        Args:
            v_over_c: V/C比数组
            t0: 自由流行程时间（秒）
            alpha: α参数
            beta: β参数
            
        Returns:
            预测的行程时间数组（秒）
        """
        if t0 is None:
            t0 = self.t0
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        
        if t0 is None or alpha is None or beta is None:
            raise ValueError("必须先拟合估计器或提供所有参数")
        
        # 确保V/C比非负
        v_over_c = np.maximum(v_over_c, 0)
        
        # BPR公式
        travel_time = t0 * (1 + alpha * np.power(v_over_c, beta))
        
        return travel_time
    
    def info(self) -> Dict[str, Any]:
        """获取BPR估计器信息"""
        info_dict = super().info()
        info_dict['params'].update({
            'alpha': self.alpha,
            'beta': self.beta,
            't0': self.t0
        })
        return info_dict


class MLEstimator(BaseEstimator):
    """
    机器学习估计器的基类
    
    提供两种模式：
    1. 直接学习 t/t0（用于M0直接替代BPR）
    2. 学习残差 Δ = t - t_bpr（用于M5混合模型）
    """
    
    def __init__(self, mode: str = 'direct'):
        """
        初始化ML估计器
        
        Args:
            mode: 模式
                - 'direct': 直接学习 t/t0
                - 'residual': 学习残差 Δ
        """
        super().__init__()
        self.mode = mode
        self.model = None
        self.feature_names = None
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        准备特征矩阵
        
        Args:
            df: FinalData DataFrame
            
        Returns:
            特征矩阵
        """
        # 基础特征
        features = ['v_over_c']
        
        # 可选特征
        optional_features = [
            'hgv_share',      # HGV份额
            'hour',           # 小时
            'weekday',        # 星期几
        ]
        
        for feat in optional_features:
            if feat in df.columns:
                features.append(feat)
        
        self.feature_names = features
        
        # 提取特征矩阵
        X = df[features].values
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)
        
        return X
    
    def info(self) -> Dict[str, Any]:
        """获取ML估计器信息"""
        info_dict = super().info()
        info_dict['params'].update({
            'mode': self.mode,
            'feature_names': self.feature_names
        })
        return info_dict


def create_estimator(method: str, **kwargs) -> BaseEstimator:
    """
    估计器工厂函数
    
    根据方法名称创建相应的估计器实例
    
    Args:
        method: 估计方法名称
            - 'classical': 经典BPR（α=0.15, β=4.0）
            - 'loglinear': 对数线性回归
            - 'nls': 非线性最小二乘法
            - 'svr': 支持向量回归
            - 'tree': 决策树
            - 'rf': 随机森林
            - 'gbdt': 梯度提升决策树
            - 'bayes': 贝叶斯回归
            - 'nn': 神经网络
        **kwargs: 传递给估计器的额外参数
        
    Returns:
        估计器实例
    """
    
    # 延迟导入以避免循环依赖
    from .bpr_classical import BPRClassical
    from .bpr_loglinear import BPRLogLinear
    from .bpr_nls import BPRNLS
    from .ml_svr import SVREstimator
    from .ml_tree import TreeEstimator
    from .ml_rf import RFEstimator
    from .ml_gbdt import GBDTEstimator
    from .ml_nn import NNEstimator
    
    estimator_map = {
        'classical': BPRClassical,
        'loglinear': BPRLogLinear,
        'nls': BPRNLS,
        'svr': SVREstimator,
        'tree': TreeEstimator,
        'rf': RFEstimator,
        'gbdt': GBDTEstimator,
        'nn': NNEstimator,
    }
    
    if method not in estimator_map:
        raise ValueError(f"未知的估计方法: {method}. 可用方法: {list(estimator_map.keys())}")
    
    estimator_class = estimator_map[method]
    return estimator_class(**kwargs)


if __name__ == "__main__":
    """测试基类"""
    
    print("估计器基类定义完成")
    print("\n可用的估计器接口:")
    print("  - fit(df, *, t0=None)")
    print("  - predict(df)")
    print("  - info()")
    print("\n估计器类型:")
    print("  - BPREstimator: BPR类估计器基类")
    print("  - MLEstimator: 机器学习估计器基类")


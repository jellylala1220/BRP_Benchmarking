"""
SVR估计器 - 支持向量回归

使用RBF核的SVR进行行程时间估计
支持两种模式：
1. direct: 直接学习 t/t0
2. residual: 学习残差（用于M5混合模型）
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from .base_estimator import MLEstimator


class SVREstimator(MLEstimator):
    """SVR估计器"""
    
    def __init__(self, mode='direct', **kwargs):
        super().__init__(mode)
        
        # SVR参数
        self.model = SVR(
            kernel=kwargs.get('kernel', 'rbf'),
            C=kwargs.get('C', 1.0),
            epsilon=kwargs.get('epsilon', 0.1),
            gamma=kwargs.get('gamma', 'scale')
        )
        
        self.scaler = StandardScaler()
        self.t0 = None
    
    def fit(self, df: pd.DataFrame, *, t0: float = None) -> 'SVREstimator':
        if t0 is None:
            self.t0 = df['t0_ff'].iloc[0] if 't0_ff' in df.columns else 100
        else:
            self.t0 = t0
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        if self.mode == 'direct':
            y = df['fused_tt_15min'].values / self.t0
        else:  # residual
            y = df['fused_tt_15min'].values
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.params = {'t0': self.t0, 'mode': self.mode}
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("估计器未拟合")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        if self.mode == 'direct':
            return y_pred * self.t0
        return y_pred


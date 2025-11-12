"""神经网络估计器"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from .base_estimator import MLEstimator


class NNEstimator(MLEstimator):
    """神经网络估计器"""
    
    def __init__(self, mode='direct', **kwargs):
        super().__init__(mode)
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(kwargs.get('hidden_layer_sizes', [64, 32, 16])),
            activation=kwargs.get('activation', 'relu'),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42),
            early_stopping=True
        )
        self.scaler = StandardScaler()
        self.t0 = None
    
    def fit(self, df: pd.DataFrame, *, t0: float = None) -> 'NNEstimator':
        if t0 is None:
            self.t0 = df['t0_ff'].iloc[0] if 't0_ff' in df.columns else 100
        else:
            self.t0 = t0
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        y = df['fused_tt_15min'].values / self.t0 if self.mode == 'direct' else df['fused_tt_15min'].values
        
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
        return y_pred * self.t0 if self.mode == 'direct' else y_pred


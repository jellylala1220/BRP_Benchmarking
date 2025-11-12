"""随机森林估计器"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .base_estimator import MLEstimator


class RFEstimator(MLEstimator):
    """随机森林估计器"""
    
    def __init__(self, mode='direct', **kwargs):
        super().__init__(mode)
        self.model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            min_samples_split=kwargs.get('min_samples_split', 20),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1
        )
        self.t0 = None
    
    def fit(self, df: pd.DataFrame, *, t0: float = None) -> 'RFEstimator':
        if t0 is None:
            self.t0 = df['t0_ff'].iloc[0] if 't0_ff' in df.columns else 100
        else:
            self.t0 = t0
        
        X = self.prepare_features(df)
        y = df['fused_tt_15min'].values / self.t0 if self.mode == 'direct' else df['fused_tt_15min'].values
        
        self.model.fit(X, y)
        self.is_fitted = True
        self.params = {'t0': self.t0, 'mode': self.mode}
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("估计器未拟合")
        
        X = self.prepare_features(df)
        y_pred = self.model.predict(X)
        return y_pred * self.t0 if self.mode == 'direct' else y_pred
    
    def get_feature_importance(self):
        if self.is_fitted:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}


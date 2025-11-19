import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from .base import BaseVDF

class E1_SVR(BaseVDF):
    """
    E1: SVR-BPR Hybrid (Pure SVR here as per requirement 'Unstructured VDF')
    Features: VOC, VOC^2, HGV, TOD, Daytype, Weather
    """
    def __init__(self):
        super().__init__("E1_SVR")
        self.model = SVR(kernel='rbf', C=10.0, epsilon=0.1)
        self.scaler = StandardScaler()
        
    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        # x_t = [VOC, VOC^2, HGV, tod_sin, tod_cos, is_weekend, Rain, HeavyRain, LowVis]
        voc = df['VOC_t'].values
        hgv = df['HGV_share'].values
        tod_sin = df['tod_sin'].values
        tod_cos = df['tod_cos'].values
        is_weekend = df['daytype'].isin([5, 6]).astype(int).values
        rain = df['Rain_t'].values
        
        X = np.column_stack([
            voc, np.power(voc, 2), hgv, 
            tod_sin, tod_cos, is_weekend, rain
        ])
        return X
        
    def fit(self, df_train: pd.DataFrame) -> 'E1_SVR':
        X = self._get_features(df_train)
        y = df_train['T_obs'].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X = self._get_features(df_test)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class E2_RF(BaseVDF):
    """
    E2: Random Forest
    """
    def __init__(self):
        super().__init__("E2_RF")
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        # Same features as E1
        voc = df['VOC_t'].values
        hgv = df['HGV_share'].values
        tod_sin = df['tod_sin'].values
        tod_cos = df['tod_cos'].values
        is_weekend = df['daytype'].isin([5, 6]).astype(int).values
        rain = df['Rain_t'].values
        
        X = np.column_stack([
            voc, np.power(voc, 2), hgv, 
            tod_sin, tod_cos, is_weekend, rain
        ])
        return X
        
    def fit(self, df_train: pd.DataFrame) -> 'E2_RF':
        X = self._get_features(df_train)
        y = df_train['T_obs'].values
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X = self._get_features(df_test)
        return self.model.predict(X)

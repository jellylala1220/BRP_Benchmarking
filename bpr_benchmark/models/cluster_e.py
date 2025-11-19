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
        self.model = SVR(kernel='rbf', C=100, epsilon=0.1)
        self.scaler = StandardScaler()
        # Unified features
        self.features = ['VOC_t', 'HGV_share', 'tod_sin', 'tod_cos', 'is_weekend', 'Rain_t', 'LowVis_t']
        
    def _prepare_X(self, df: pd.DataFrame, fit_scaler=False):
        # Ensure features exist
        if 'is_weekend' not in df.columns and 'daytype' in df.columns:
            df['is_weekend'] = df['daytype'].isin([5, 6]).astype(int)
            
        # Add VOC^2
        df['VOC_sq'] = df['VOC_t'] ** 2
        
        # Full feature list for X
        # Note: VOC_sq is added dynamically
        cols = self.features + ['VOC_sq']
        
        X = df[cols].values
        
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
            
        return X

    def fit(self, df_train: pd.DataFrame):
        X = self._prepare_X(df_train, fit_scaler=True)
        y = df_train['T_obs'].values
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X = self._prepare_X(df_test, fit_scaler=False)
        return self.model.predict(X)


class E2_RF(BaseVDF):
    """
    E2: Random Forest
    Uses same unified features.
    """
    def __init__(self):
        super().__init__("E2_RF")
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        # Unified features
        self.features = ['VOC_t', 'HGV_share', 'tod_sin', 'tod_cos', 'is_weekend', 'Rain_t', 'LowVis_t']
        
    def _prepare_X(self, df: pd.DataFrame):
        # Ensure features exist
        if 'is_weekend' not in df.columns and 'daytype' in df.columns:
            df['is_weekend'] = df['daytype'].isin([5, 6]).astype(int)
            
        # Add VOC^2
        df['VOC_sq'] = df['VOC_t'] ** 2
        
        cols = self.features + ['VOC_sq']
        return df[cols].values

    def fit(self, df_train: pd.DataFrame):
        X = self._prepare_X(df_train)
        y = df_train['T_obs'].values
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X = self._prepare_X(df_test)
        return self.model.predict(X)

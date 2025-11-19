import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from .base import BaseVDF

class E1_SVR(BaseVDF):
    """
    E1: SVR Residual Model (BPR Enhancer)
    
    Per requirementsNEW.md Section 6:
    - Predicts residual ratio r_t = (T_obs / T_A1) - 1
    - Final output: T_E = T_A1 * (1 + r_hat)
    - Preserves BPR monotonicity via A1 baseline
    
    Features: VOC, VOC^2, HGV_share, tod_sin, tod_cos, is_weekend, Rain, LowVis
    """
    def __init__(self):
        super().__init__("E1_SVR")
        self.model = SVR(kernel='rbf', C=100, epsilon=0.1)
        self.scaler = StandardScaler()
        self.features = ['VOC_t', 'HGV_share', 'tod_sin', 'tod_cos', 'is_weekend', 'Rain_t', 'LowVis_t']
        self.A1_baseline = None  # Will store A1 predictions for residual calculation
        
    def _prepare_X(self, df: pd.DataFrame, fit_scaler=False):
        # Ensure features exist
        if 'is_weekend' not in df.columns and 'daytype' in df.columns:
            df['is_weekend'] = df['daytype'].isin([5, 6]).astype(int)
            
        # Add VOC^2
        df_work = df.copy()
        df_work['VOC_sq'] = df_work['VOC_t'] ** 2
        
        cols = self.features + ['VOC_sq']
        X = df_work[cols].values
        
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
            
        return X

    def fit(self, df_train: pd.DataFrame, T_A1_train: np.ndarray):
        """
        Fit SVR on residual ratios.
        
        Args:
            df_train: Training dataframe
            T_A1_train: A1 baseline predictions on training set
        """
        X = self._prepare_X(df_train, fit_scaler=True)
        
        # Calculate residual ratio: r_t = (T_obs / T_A1) - 1
        T_obs = df_train['T_obs'].values
        r_train = (T_obs / T_A1_train) - 1
        
        self.model.fit(X, r_train)
        self.is_fitted = True
        return self

    def predict(self, df_test: pd.DataFrame, T_A1_test: np.ndarray) -> np.ndarray:
        """
        Predict travel time via residual enhancement.
        
        Args:
            df_test: Test dataframe
            T_A1_test: A1 baseline predictions on test set
            
        Returns:
            T_E = T_A1 * (1 + r_hat)
        """
        X = self._prepare_X(df_test, fit_scaler=False)
        r_hat = self.model.predict(X)
        
        # Final prediction: T_E = T_A1 * (1 + r_hat)
        return T_A1_test * (1 + r_hat)


class E2_RF(BaseVDF):
    """
    E2: Random Forest Residual Model (BPR Enhancer)
    
    Per requirementsNEW.md Section 6:
    - Predicts residual ratio r_t = (T_obs / T_A1) - 1
    - Final output: T_E = T_A1 * (1 + r_hat)
    - Preserves BPR monotonicity via A1 baseline
    
    Features: VOC, VOC^2, HGV_share, tod_sin, tod_cos, is_weekend, Rain, LowVis
    """
    def __init__(self):
        super().__init__("E2_RF")
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.features = ['VOC_t', 'HGV_share', 'tod_sin', 'tod_cos', 'is_weekend', 'Rain_t', 'LowVis_t']
        self.A1_baseline = None
        
    def _prepare_X(self, df: pd.DataFrame):
        # Ensure features exist
        if 'is_weekend' not in df.columns and 'daytype' in df.columns:
            df['is_weekend'] = df['daytype'].isin([5, 6]).astype(int)
            
        df_work = df.copy()
        df_work['VOC_sq'] = df_work['VOC_t'] ** 2
        
        cols = self.features + ['VOC_sq']
        return df_work[cols].values

    def fit(self, df_train: pd.DataFrame, T_A1_train: np.ndarray):
        """
        Fit Random Forest on residual ratios.
        
        Args:
            df_train: Training dataframe
            T_A1_train: A1 baseline predictions on training set
        """
        X = self._prepare_X(df_train)
        
        # Calculate residual ratio: r_t = (T_obs / T_A1) - 1
        T_obs = df_train['T_obs'].values
        r_train = (T_obs / T_A1_train) - 1
        
        self.model.fit(X, r_train)
        self.is_fitted = True
        return self

    def predict(self, df_test: pd.DataFrame, T_A1_test: np.ndarray) -> np.ndarray:
        """
        Predict travel time via residual enhancement.
        
        Args:
            df_test: Test dataframe
            T_A1_test: A1 baseline predictions on test set
            
        Returns:
            T_E = T_A1 * (1 + r_hat)
        """
        X = self._prepare_X(df_test)
        r_hat = self.model.predict(X)
        
        # Final prediction: T_E = T_A1 * (1 + r_hat)
        return T_A1_test * (1 + r_hat)

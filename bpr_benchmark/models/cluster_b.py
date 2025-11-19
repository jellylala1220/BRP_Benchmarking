import numpy as np
import pandas as pd
from .base import BaseVDF
from ..utils.estimation import fit_nls

class B1_DP_BPR(BaseVDF):
    """
    B1: DP-BPR (Dynamic Parameters)
    T = t0 * (1 + alpha(x) * (V/C)^beta(x))
    log(alpha) = eta0 + eta * x
    log(beta) = gamma0 + gamma * x
    
    Features x: tod_sin, tod_cos, daytype dummies, weather dummies.
    """
    def __init__(self):
        super().__init__("B1_DP_BPR")
        self.params = {}
        
    def fit(self, df_train: pd.DataFrame) -> 'B1_DP_BPR':
        t0 = df_train['t_0'].values
        voc = df_train['VOC_t'].values
        y_true = df_train['T_obs'].values
        
        # Prepare features X
        # x columns: tod_sin, tod_cos, Rain_t, HeavyRain_t, LowVis_t
        # And Daytype dummies (simplified: is_weekend)
        df_train['is_weekend'] = df_train['daytype'].isin([5, 6]).astype(int)
        
        feature_cols = ['tod_sin', 'tod_cos', 'is_weekend', 'Rain_t']
        X = df_train[feature_cols].values
        n_features = X.shape[1]
        
        # Parameters: eta0 (1), eta (n), gamma0 (1), gamma (n)
        # Total params = 2 * (1 + n)
        
        def dp_bpr_func(data_inputs, *params):
            # Unpack inputs
            # data_inputs is flattened, we need to reshape or pass tuple
            # curve_fit passes X as first arg. If X is (N, M), it works.
            # But here we need both VOC and X.
            # So we stack VOC and X: data_inputs = [VOC, X_col1, X_col2...]
            
            voc_in = data_inputs[:, 0]
            X_in = data_inputs[:, 1:]
            
            n_feat = X_in.shape[1]
            
            # Parse params
            eta0 = params[0]
            eta = np.array(params[1:1+n_feat])
            gamma0 = params[1+n_feat]
            gamma = np.array(params[2+n_feat:])
            
            # Calculate alpha and beta
            # log_alpha = eta0 + X @ eta
            log_alpha = eta0 + np.dot(X_in, eta)
            alpha = np.exp(log_alpha)
            
            # log_beta = gamma0 + X @ gamma
            log_beta = gamma0 + np.dot(X_in, gamma)
            beta = np.exp(log_beta)
            
            # BPR
            return t0 * (1 + alpha * np.power(voc_in, beta))
            
        # Prepare combined input
        X_combined = np.column_stack([voc, X])
        
        # Initial guess
        # eta0 corresponds to log(0.15) ~ -1.9
        # gamma0 corresponds to log(4.0) ~ 1.4
        # others 0
        p0 = [-1.9] + [0]*n_features + [1.4] + [0]*n_features
        
        # Bounds? Unconstrained for log-params usually, but let's keep them reasonable
        # to prevent overflow.
        
        popt, _ = fit_nls(
            dp_bpr_func,
            X_combined,
            y_true,
            p0=p0,
            bounds=(-np.inf, np.inf) # Let optimizer handle it, or add loose bounds
        )
        
        # Debug popt
        # print(f"DEBUG B1 fit: popt shape: {popt.shape}")
        # print(f"DEBUG B1 fit: popt: {popt}")
        # print(f"DEBUG B1 fit: n_features: {n_features}")
        
        self.params = {
            'eta0': popt[0],
            'eta': popt[1:1+n_features],
            'gamma0': popt[1+n_features],
            'gamma': popt[2+n_features:], # Fix slicing index
            'features': feature_cols
        }
        
        # Debug params
        # print(f"DEBUG B1 fit: eta shape: {self.params['eta'].shape}")
        # print(f"DEBUG B1 fit: gamma shape: {self.params['gamma'].shape}")
        
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        t0 = df_test['t_0'].values
        voc = df_test['VOC_t'].values
        
        # Features
        df_test['is_weekend'] = df_test['daytype'].isin([5, 6]).astype(int)
        X = df_test[self.params['features']].values
        
        eta0 = self.params['eta0']
        eta = self.params['eta']
        gamma0 = self.params['gamma0']
        gamma = self.params['gamma']
        
        # Ensure dot product results in 1D array
        # Debug shapes
        # print(f"DEBUG B1: X shape: {X.shape}")
        # print(f"DEBUG B1: eta shape: {np.shape(eta)}")
        # print(f"DEBUG B1: gamma shape: {np.shape(gamma)}")
        
        # Force eta/gamma to be flat arrays
        eta = np.array(eta).flatten()
        gamma = np.array(gamma).flatten()
        
        alpha = np.exp(eta0 + np.dot(X, eta))
        beta = np.exp(gamma0 + np.dot(X, gamma))
        
        # print(f"DEBUG B1: alpha shape: {alpha.shape}")
        # print(f"DEBUG B1: beta shape: {beta.shape}")
        
        return t0 * (1 + alpha * np.power(voc, beta))


class B2_Rolling_DVDF(BaseVDF):
    """
    B2: Rolling-Horizon DVDF
    Uses Load Factor chi_t instead of VOC.
    chi_t = VOC if not oversaturated
          = D_h(t) / m_h(t) if oversaturated
          
    Oversaturated defined as VOC > 0.9 (example threshold).
    Rolling window h = 30 min (2 steps).
    """
    def __init__(self):
        super().__init__("B2_Rolling")
        self.alpha = 0.15
        self.beta = 4.0
        self.window = 2 # 30 mins = 2 * 15min
        self.threshold = 0.9
        
    def _calculate_chi(self, df: pd.DataFrame) -> np.ndarray:
        voc = df['VOC_t'].values
        q = df['q_t'].values
        C = df['C'].values
        
        # Use numpy for rolling to avoid index alignment issues
        # q is numpy array, C is numpy array
        q_series = pd.Series(q)
        c_series = pd.Series(C)
        
        d_avg = q_series.rolling(window=self.window, min_periods=1).mean().values
        m_avg = c_series.rolling(window=self.window, min_periods=1).mean().values
        
        chi = np.where(voc > self.threshold, d_avg / m_avg, voc)
        return chi
        
    def fit(self, df_train: pd.DataFrame) -> 'B2_Rolling_DVDF':
        t0 = df_train['t_0'].values
        y_true = df_train['T_obs'].values
        
        chi = self._calculate_chi(df_train)
        
        def bpr_chi(chi_val, alpha, beta):
            return t0 * (1 + alpha * np.power(chi_val, beta))
            
        popt, _ = fit_nls(
            bpr_chi,
            chi,
            y_true,
            p0=[0.15, 4.0],
            bounds=([0.01, 1.01], [10.0, 20.0])
        )
        
        self.alpha, self.beta = popt
        self.params = {'alpha': self.alpha, 'beta': self.beta}
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        t0 = df_test['t_0'].values
        chi = self._calculate_chi(df_test)
        return t0 * (1 + self.alpha * np.power(chi, self.beta))


class B3_Stochastic(BaseVDF):
    """
    B3: Stochastic Demand/Capacity
    T = t0 * (1 + alpha * E[phi^n])
    phi = q / (varphi * C)
    varphi = exp(delta * weather)
    
    Simplified:
    1. Estimate varphi (capacity degradation) from weather? 
       Or assume varphi distribution?
    2. Fit alpha, n.
    
    Implementation:
    - Assume varphi = 1 for Dry, <1 for Rain.
    - Fit varphi parameters first? Or jointly?
    - Requirement: "Fit phi distribution... then NLS".
    
    Let's simplify:
    - Model varphi = 1 / (1 + delta * Rain)
    - phi = q * (1 + delta * Rain) / C
    - T = t0 * (1 + alpha * phi^beta)  (Approximating E[phi^n] as phi^n for single point, 
      or we need to sample? For benchmarking, point estimate is standard unless we do Monte Carlo)
      
    Requirement says: "E[T] = ... E[phi^n]".
    If we predict a single value T_t, we are predicting E[T_t].
    So we compute E[phi_t^n].
    If phi_t is random due to capacity fluctuation within the 15min?
    Or is it just deterministic based on weather?
    If deterministic based on weather, E[phi^n] = phi^n.
    
    Let's implement deterministic weather impact first (similar to D1 but in B cluster context).
    """
    def __init__(self):
        super().__init__("B3_Stochastic")
        self.params = {}
        
    def fit(self, df_train: pd.DataFrame) -> 'B3_Stochastic':
        # Simplified B3: Weather-weighted BPR
        # T = t0 * (1 + alpha * (q / (C * varphi))^beta)
        # varphi = exp(-delta * Rain)
        
        t0 = df_train['t_0'].values
        q = df_train['q_t'].values
        C = df_train['C'].values
        Rain = df_train['Rain_t'].values
        y_true = df_train['T_obs'].values
        
        def stochastic_bpr(inputs, alpha, beta, delta):
            # inputs: [q, C, Rain]
            q_in = inputs[:, 0]
            C_in = inputs[:, 1]
            R_in = inputs[:, 2]
            
            varphi = np.exp(-delta * R_in)
            phi = q_in / (C_in * varphi)
            
            return t0 * (1 + alpha * np.power(phi, beta))
            
        inputs = np.column_stack([q, C, Rain])
        
        popt, _ = fit_nls(
            stochastic_bpr,
            inputs,
            y_true,
            p0=[0.15, 4.0, 0.1],
            bounds=([0.01, 1.01, 0.0], [10.0, 20.0, 5.0])
        )
        
        self.params = {'alpha': popt[0], 'beta': popt[1], 'delta': popt[2]}
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        t0 = df_test['t_0'].values
        q = df_test['q_t'].values
        C = df_test['C'].values
        Rain = df_test['Rain_t'].values
        
        alpha = self.params['alpha']
        beta = self.params['beta']
        delta = self.params['delta']
        
        varphi = np.exp(-delta * Rain)
        phi = q / (C * varphi)
        
        return t0 * (1 + alpha * np.power(phi, beta))

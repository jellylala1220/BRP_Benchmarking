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
        # Features for alpha and beta:
        # tod_sin, tod_cos, is_weekend, Rain_t, HGV_share
        self.feature_cols = ['tod_sin', 'tod_cos', 'is_weekend', 'Rain_t', 'HGV_share']
        self.params = {}
        
    def fit(self, df_train: pd.DataFrame) -> 'B1_DP_BPR':
        # Prepare data
        # Ensure features exist
        for col in self.feature_cols:
            if col not in df_train.columns:
                # Add 'is_weekend' if not present, assuming 'daytype' is available
                if col == 'is_weekend' and 'daytype' in df_train.columns:
                    df_train['is_weekend'] = df_train['daytype'].isin([5, 6]).astype(int)
                else:
                    raise ValueError(f"Missing feature {col} for B1 model")
                
        X = df_train[self.feature_cols].values
        y_true = df_train['T_obs'].values
        t0 = df_train['t_0'].values
        voc = df_train['VOC_t'].values
        
        # Initial guess
        # eta0 ~ log(0.15), gamma0 ~ log(4.0)
        # others 0
        n_features = len(self.feature_cols)
        p0 = np.zeros(2 + 2 * n_features)
        p0[0] = np.log(0.15) # eta0
        p0[1 + n_features] = np.log(4.0) # gamma0
        
        def dp_bpr_func(X_combined, *params):
            # Unpack params
            # params: [eta0, eta_1...eta_k, gamma0, gamma_1...gamma_k]
            eta0 = params[0]
            eta = np.array(params[1:1+n_features])
            gamma0 = params[1+n_features]
            gamma = np.array(params[2+n_features:])
            
            # Split X_combined back to X and voc/t0
            # We need to pass X, voc, t0 together or use closure.
            # fit_nls passes X_combined to func.
            # Let's assume X_combined has everything.
            # Structure of X_combined: [X columns..., VOC, t0]
            
            X_curr = X_combined[:, :n_features]
            voc_curr = X_combined[:, n_features]
            t0_curr = X_combined[:, n_features+1]
            
            alpha = np.exp(eta0 + np.dot(X_curr, eta))
            beta = np.exp(gamma0 + np.dot(X_curr, gamma))
            
            return t0_curr * (1 + alpha * np.power(voc_curr, beta))
            
        # Combine X, voc, t0 for curve_fit
        X_combined = np.column_stack([X, voc, t0])
        
        popt, _ = fit_nls(
            dp_bpr_func,
            X_combined,
            y_true,
            p0=p0,
            bounds=(-np.inf, np.inf) 
        )
        
        self.params = {
            'eta0': popt[0],
            'eta': popt[1:1+n_features],
            'gamma0': popt[1+n_features],
            'gamma': popt[2+n_features:], 
            'features': self.feature_cols
        }
        
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        t0 = df_test['t_0'].values
        voc = df_test['VOC_t'].values
        
        # Features
        # Add 'is_weekend' if not present, assuming 'daytype' is available
        if 'is_weekend' not in df_test.columns and 'daytype' in df_test.columns:
            df_test['is_weekend'] = df_test['daytype'].isin([5, 6]).astype(int)
        
        X = df_test[self.params['features']].values
        
        eta0 = self.params['eta0']
        eta = self.params['eta']
        gamma0 = self.params['gamma0']
        gamma = self.params['gamma']
        
        # Force eta/gamma to be flat arrays
        eta = np.array(eta).flatten()
        gamma = np.array(gamma).flatten()
        
        alpha = np.exp(eta0 + np.dot(X, eta))
        beta = np.exp(gamma0 + np.dot(X, gamma))
        
        return t0 * (1 + alpha * np.power(voc, beta))


class B2_Rolling_DVDF(BaseVDF):
    """
    B2: Rolling-Horizon DVDF
    Uses Load Factor chi_t instead of VOC.
    chi_t = VOC if not oversaturated
          = D_h(t) / m_h(t) if oversaturated
          
    Oversaturated defined as VOC > threshold.
    Rolling window h (steps).
    """
    def __init__(self, window: int = 2, threshold: float = 0.9):
        super().__init__("B2_Rolling")
        self.alpha = 0.15
        self.beta = 4.0
        self.window = window # 30 mins = 2 * 15min by default
        self.threshold = threshold
        
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
    B3: Stochastic-Capacity BPR (Mean-Delay Variant)
    
    Per requirementsNEW.md Section 3 (B3):
    1. Define effective capacity factor φ_t = exp(δ0 + δ1*Rain + δ2*HeavyRain + δ3*LowVis)
    2. Define stochastic saturation variable phi_t = q_t / (φ_t * C)
    3. Fit empirical distribution g(phi) from training data
    4. Calculate E[phi^n] via numerical integration
    5. Model: T = t0 * (1 + α * E[phi^n])
    
    This is a MEAN-EFFECT stochastic capacity model.
    Reliability/quantile modeling is handled in D3.
    """
    def __init__(self):
        super().__init__("B3_Stochastic")
        self.params = {}
        self.phi_dist = None  # Will store empirical distribution
        
    def fit(self, df_train: pd.DataFrame) -> 'B3_Stochastic':
        from scipy.stats import gaussian_kde
        from scipy.integrate import quad
        
        t0 = df_train['t_0'].values
        q = df_train['q_t'].values
        C = df_train['C'].values
        Rain = df_train['Rain_t'].values
        HeavyRain = df_train['HeavyRain_t'].values if 'HeavyRain_t' in df_train.columns else np.zeros_like(Rain)
        LowVis = df_train['LowVis_t'].values if 'LowVis_t' in df_train.columns else np.zeros_like(Rain)
        y_true = df_train['T_obs'].values
        
        # Step 1: Fit capacity degradation function varphi_t
        # We'll estimate (delta0, delta1, delta2, delta3) jointly with (alpha, n)
        # Initial approach: fit varphi parameters separately first
        
        def stochastic_model(inputs, alpha, n, delta0, delta1, delta2, delta3):
            # inputs: [q, C, Rain, HeavyRain, LowVis]
            q_in = inputs[:, 0]
            C_in = inputs[:, 1]
            R_in = inputs[:, 2]
            HR_in = inputs[:, 3]
            LV_in = inputs[:, 4]
            
            # Capacity degradation factor
            varphi = np.exp(delta0 + delta1*R_in + delta2*HR_in + delta3*LV_in)
            
            # Stochastic saturation variable
            phi = q_in / (varphi * C_in)
            
            # For simplicity in NLS, use point estimate E[phi^n] ≈ phi^n
            # (True stochastic version would fit distribution then integrate)
            return t0 * (1 + alpha * np.power(phi, n))
            
        inputs = np.column_stack([q, C, Rain, HeavyRain, LowVis])
        
        # Initial parameter guess
        p0 = [0.15, 4.0, 0.0, -0.1, -0.2, -0.15]  # alpha, n, delta0, delta1, delta2, delta3
        
        # Bounds: alpha>0, n>0, deltas can be negative (capacity reduction)
        popt, _ = fit_nls(
            stochastic_model,
            inputs,
            y_true,
            p0=p0,
            bounds=([0.01, 0.1, -1.0, -5.0, -5.0, -5.0],   # lower
                    [10.0, 10.0,  1.0,  0.0,  0.0,  0.0])   # upper
        )
        
        # Step 2: Calculate phi_t with fitted parameters for distribution fitting
        delta0, delta1, delta2, delta3 = popt[2], popt[3], popt[4], popt[5]
        varphi = np.exp(delta0 + delta1*Rain + delta2*HeavyRain + delta3*LowVis)
        phi_t = q / (varphi * C)
        
        # Step 3: Fit empirical distribution (Kernel Density Estimation)
        # Filter out extreme values
        phi_clean = phi_t[(phi_t > 0) & (phi_t < np.quantile(phi_t, 0.99))]
        if len(phi_clean) > 10:
            self.phi_dist = gaussian_kde(phi_clean)
        else:
            self.phi_dist = None
        
        # Step 4: Calculate E[phi^n] via numerical integration
        n_param = popt[1]
        if self.phi_dist is not None:
            phi_max = phi_clean.max() * 1.5
            E_phi_n, _ = quad(lambda x: x**n_param * self.phi_dist(x)[0], 0, phi_max)
        else:
            # Fallback: use empirical mean
            E_phi_n = np.mean(phi_clean**n_param)
        
        self.params = {
            'alpha': popt[0],
            'n': popt[1],
            'delta0': delta0,
            'delta1': delta1,
            'delta2': delta2,
            'delta3': delta3,
            'E_phi_n': E_phi_n  # Store for prediction
        }
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        t0 = df_test['t_0'].values
        
        # Use stored E[phi^n] if in mean-prediction mode
        # Or recalculate phi_t for each test point
        # Per requirements, we predict E[T] = t0 * (1 + alpha * E[phi^n])
        # where E[phi^n] is from training distribution
        
        alpha = self.params['alpha']
        E_phi_n = self.params['E_phi_n']
        
        # Mean travel time prediction
        return t0 * (1 + alpha * E_phi_n)

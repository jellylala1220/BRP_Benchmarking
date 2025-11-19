import numpy as np
import pandas as pd
from .base import BaseVDF
from ..utils.estimation import fit_nls

class D1_Weather_Capacity(BaseVDF):
    """
    D1: Weather-Adjusted Capacity
    C_t = C * exp(delta0 + delta * Weather)
    T = t0 * (1 + alpha * (q / C_t)^beta)
    """
    def __init__(self):
        super().__init__("D1_Weather")
        self.params = {}
        
    def fit(self, df_train: pd.DataFrame) -> 'D1_Weather_Capacity':
        t0 = df_train['t_0'].values
        q = df_train['q_t'].values
        C = df_train['C'].values
        y_true = df_train['T_obs'].values
        
        # Weather features: Rain_t, LowVis_t
        # C_t = C * exp(d0 + d1*Rain + d2*Vis)
        
        Rain = df_train['Rain_t'].values
        Vis = df_train['LowVis_t'].values
        
        def weather_cap_func(inputs, alpha, beta, d0, d1, d2):
            q_in = inputs[:, 0]
            C_in = inputs[:, 1]
            R_in = inputs[:, 2]
            V_in = inputs[:, 3]
            
            adj = np.exp(d0 + d1*R_in + d2*V_in)
            C_t = C_in * adj
            
            return t0 * (1 + alpha * np.power(q_in / C_t, beta))
            
        inputs = np.column_stack([q, C, Rain, Vis])
        
        popt, _ = fit_nls(
            weather_cap_func,
            inputs,
            y_true,
            p0=[0.15, 4.0, 0.0, -0.1, -0.1],
            bounds=([0.01, 1.01, -1.0, -2.0, -2.0], [10.0, 20.0, 1.0, 0.5, 0.5])
        )
        
        self.params = {
            'alpha': popt[0], 'beta': popt[1], 
            'd0': popt[2], 'd1': popt[3], 'd2': popt[4]
        }
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        t0 = df_test['t_0'].values
        q = df_test['q_t'].values
        C = df_test['C'].values
        Rain = df_test['Rain_t'].values
        Vis = df_test['LowVis_t'].values
        
        d0, d1, d2 = self.params['d0'], self.params['d1'], self.params['d2']
        adj = np.exp(d0 + d1*Rain + d2*Vis)
        C_t = C * adj
        
        return t0 * (1 + self.params['alpha'] * np.power(q / C_t, self.params['beta']))


class D3_Reliability_ETT(BaseVDF):
    """
    D3: Reliability-Based (ETT)
    T = t0 * (1 + alpha * E[phi^n])
    Similar to B3 but framed as reliability.
    We'll implement it identically to B3 for now as the mathematical form is very similar
    in this simplified context.
    """
    def __init__(self):
        super().__init__("D3_Reliability")
        self.params = {}
        
    def fit(self, df_train: pd.DataFrame) -> 'D3_Reliability_ETT':
        # Re-using B3 logic
        t0 = df_train['t_0'].values
        q = df_train['q_t'].values
        C = df_train['C'].values
        Rain = df_train['Rain_t'].values
        y_true = df_train['T_obs'].values
        
        def ett_func(inputs, alpha, beta, delta):
            q_in = inputs[:, 0]
            C_in = inputs[:, 1]
            R_in = inputs[:, 2]
            
            varphi = np.exp(-delta * R_in)
            phi = q_in / (C_in * varphi)
            
            return t0 * (1 + alpha * np.power(phi, beta))
            
        inputs = np.column_stack([q, C, Rain])
        
        popt, _ = fit_nls(
            ett_func,
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

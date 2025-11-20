import numpy as np
import pandas as pd
from .base import BaseVDF
from ..utils.estimation import fit_nls

class C3_Mixed_FD_VDF(BaseVDF):
    """
    C3: Mixed-Traffic FD-based VDF
    
    FD parameters depend on HGV share:
    - v_f = v_f0 + v_f1 * HGV_share
    - k_c = k_c0 + k_c1 * HGV_share
    
    Then use FD-derived VDF:
    T = t0 * (1 + g(k/k_c, HGV_share; m))
    
    Simplified implementation:
    T = t0 * (1 + alpha * (k/k_c)^m * (1 + delta*HGV))
    """
    def __init__(self):
        super().__init__("C3_Mixed_FD")
        self.params = {}
        
    def fit(self, df_train: pd.DataFrame) -> 'C3_Mixed_FD_VDF':
        t0 = df_train['t_0'].values
        q = df_train['q_t'].values
        HGV_share = df_train['HGV_share'].values
        y_true = df_train['T_obs'].values
        
        # Calculate density
        L_km = 2.7138
        s = (L_km) / (df_train['T_obs'].values / 3600.0)
        k = q / s  # Density in veh/km
        
        # Estimate critical density with HGV adjustment
        k_sorted_idx = np.argsort(q)
        top_flow_idx = k_sorted_idx[-int(len(k)*0.1):]  # Top 10% flows
        k_c_base = np.median(k[top_flow_idx])
        
        def mixed_fd_vdf(inputs, alpha, m, delta, kc_mult):
            k_in = inputs[:, 0]
            hgv_in = inputs[:, 1]
            
            # HGV-adjusted critical density
            k_c_eff = k_c_base * kc_mult * (1 - delta * hgv_in)
            
            # Relative density
            k_ratio = k_in / np.maximum(k_c_eff, 0.1)
            
            # VDF with HGV effect
            return t0 * (1 + alpha * np.power(k_ratio, m))
        
        inputs = np.column_stack([k, HGV_share])
        
        popt, _ = fit_nls(
            mixed_fd_vdf,
            inputs,
            y_true,
            p0=[1.0, 2.0, 0.1, 1.0],  # alpha, m, delta, kc_mult
            bounds=([0.1, 0.5, 0.0, 0.5], [10.0, 5.0, 0.5, 2.0])
        )
        
        self.params = {
            'alpha': popt[0],
            'm': popt[1],
            'delta': popt[2],
            'kc_mult': popt[3],
            'kc_base': k_c_base
        }
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        t0 = df_test['t_0'].values
        q = df_test['q_t'].values
        HGV_share = df_test['HGV_share'].values
        
        # Approximate density using free-flow speed
        L_km = 2.7138
        vf = L_km / (t0 / 3600.0)
        k_approx = q / vf
        
        # HGV-adjusted critical density
        k_c_eff = self.params['kc_base'] * self.params['kc_mult'] * (1 - self.params['delta'] * HGV_share)
        
        # Relative density
        k_ratio = k_approx / np.maximum(k_c_eff, 0.1)
        
        # Predict
        return t0 * (1 + self.params['alpha'] * np.power(k_ratio, self.params['m']))

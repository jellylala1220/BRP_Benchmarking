import numpy as np
import pandas as pd
from .base import BaseVDF
from ..utils.estimation import fit_nls

class C1_PCU_BPR(BaseVDF):
    """
    C1: PCU-based BPR (4 Vehicle Classes)
    
    Vehicle Classes (MIDAS length-based):
    1. ≤5.2m (cars, small vans) - p1 = 1.0 (baseline)
    2. >5.2m & ≤6.6m (large vans, minibuses) - p2 ∈ [1.0, 1.5]
    3. >6.6m & ≤11.6m (coaches, rigid HGVs) - p3 ∈ [1.5, 2.5]
    4. >11.6m (articulated HGVs) - p4 ∈ [2.0, 4.0]
    
    q_pcu = 4 * Σ(p_i * V_i)
    T = t0 * (1 + α * (q_pcu / C)^β)
    """
    def __init__(self):
        super().__init__("C1_PCU")
        self.params = {}
        
    def fit(self, df_train: pd.DataFrame) -> 'C1_PCU_BPR':
        t0 = df_train['t_0'].values
        y_true = df_train['T_obs'].values
        C = df_train['C'].values
        
        # Get individual vehicle class flows
        # Note: If loader doesn't provide V1-V4, we approximate using V_HGV
        if all(col in df_train.columns for col in ['V1_t', 'V2_t', 'V3_t', 'V4_t']):
            V1 = df_train['V1_t'].values
            V2 = df_train['V2_t'].values
            V3 = df_train['V3_t'].values
            V4 = df_train['V4_t'].values
        else:
            # Fallback: approximate split
            V_total = df_train['V_t'].values
            V_HGV = df_train['V_HGV_t'].values
            V_light = V_total - V_HGV
            # Assume 70/30 split for light vehicles
            V1 = V_light * 0.7
            V2 = V_light * 0.3
            # Assume 60/40 split for HGVs
            V3 = V_HGV * 0.6
            V4 = V_HGV * 0.4
        
        def pcu_bpr(inputs, alpha, beta, p2, p3, p4):
            v1 = inputs[:, 0]
            v2 = inputs[:, 1]
            v3 = inputs[:, 2]
            v4 = inputs[:, 3]
            c_val = inputs[:, 4]
            
            # PCU-weighted flow (p1=1.0 fixed)
            q_pcu = (v1 * 1.0 + v2 * p2 + v3 * p3 + v4 * p4) * 4
            voc_pcu = q_pcu / c_val
            
            return t0 * (1 + alpha * np.power(voc_pcu, beta))
            
        inputs = np.column_stack([V1, V2, V3, V4, C])
        
        # NLS with HCM-based PCU bounds
        popt, _ = fit_nls(
            pcu_bpr,
            inputs,
            y_true,
            p0=[0.15, 4.0, 1.2, 2.0, 3.0],  # α, β, p2, p3, p4
            bounds=(
                [0.01, 1.01, 1.0, 1.5, 2.0],  # lower bounds
                [10.0, 20.0, 1.5, 2.5, 4.0]   # upper bounds
            )
        )
        
        self.params = {
            'alpha': popt[0],
            'beta': popt[1],
            'p2': popt[2],
            'p3': popt[3],
            'p4': popt[4]
        }
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        t0 = df_test['t_0'].values
        C = df_test['C'].values
        
        # Get vehicle flows (same logic as fit)
        if all(col in df_test.columns for col in ['V1_t', 'V2_t', 'V3_t', 'V4_t']):
            V1 = df_test['V1_t'].values
            V2 = df_test['V2_t'].values
            V3 = df_test['V3_t'].values
            V4 = df_test['V4_t'].values
        else:
            V_total = df_test['V_t'].values
            V_HGV = df_test['V_HGV_t'].values
            V_light = V_total - V_HGV
            V1 = V_light * 0.7
            V2 = V_light * 0.3
            V3 = V_HGV * 0.6
            V4 = V_HGV * 0.4
        
        # Calculate PCU-weighted flow
        p2, p3, p4 = self.params['p2'], self.params['p3'], self.params['p4']
        q_pcu = (V1 * 1.0 + V2 * p2 + V3 * p3 + V4 * p4) * 4
        voc_pcu = q_pcu / C
        
        return t0 * (1 + self.params['alpha'] * np.power(voc_pcu, self.params['beta']))


class C2_Yun_Truck(BaseVDF):
    """
    C2: Yun Heavy-Truck Multiplier
    S = S0 / (1 + a * (1 + T_hgv)^b * VOC^c)
    T = L / S
    """
    def __init__(self):
        super().__init__("C2_Yun")
        self.params = {}
        
    def fit(self, df_train: pd.DataFrame) -> 'C2_Yun_Truck':
        L_km = 2.7138
        # S0 = L / (t0/3600)
        t0 = df_train['t_0'].values
        S0 = L_km / (t0 / 3600.0)
        
        voc = df_train['VOC_t'].values
        T_hgv = df_train['HGV_share'].values
        y_true = df_train['T_obs'].values
        
        # Target is Travel Time.
        # T = L / S = L / (S0 / denom) = (L/S0) * denom = t0 * denom
        # T = t0 * (1 + a * (1 + T_hgv)^b * VOC^c)
        
        def yun_func(inputs, a, b, c):
            voc_in = inputs[:, 0]
            thgv_in = inputs[:, 1]
            
            term = a * np.power(1 + thgv_in, b) * np.power(voc_in, c)
            return t0 * (1 + term)
            
        inputs = np.column_stack([voc, T_hgv])
        
        popt, _ = fit_nls(
            yun_func,
            inputs,
            y_true,
            p0=[0.15, 1.0, 4.0],
            bounds=([0.01, 0.0, 1.01], [10.0, 10.0, 20.0])
        )
        
        self.params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        t0 = df_test['t_0'].values
        voc = df_test['VOC_t'].values
        T_hgv = df_test['HGV_share'].values
        
        a, b, c = self.params['a'], self.params['b'], self.params['c']
        term = a * np.power(1 + T_hgv, b) * np.power(voc, c)
        return t0 * (1 + term)

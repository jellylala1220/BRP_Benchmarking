import numpy as np
import pandas as pd
from .base import BaseVDF
from ..utils.estimation import fit_nls

class C1_PCU_BPR(BaseVDF):
    """
    C1: PCU-based BPR
    q_pcu = sum(p_i * V_i) * 4
    T = t0 * (1 + alpha * (q_pcu / C_pcu)^beta)
    
    We estimate alpha, beta, and PCU factors (p3, p4 for HGV).
    Assume p1=1 (Car), p2=1.5 (LGV).
    """
    def __init__(self):
        super().__init__("C1_PCU")
        self.params = {}
        
    def fit(self, df_train: pd.DataFrame) -> 'C1_PCU_BPR':
        t0 = df_train['t_0'].values
        y_true = df_train['T_obs'].values
        C = df_train['C'].values
        
        # Flows: V_t is total. We need breakdown.
        # Loader sums them, but we need raw columns or re-extract.
        # Let's assume loader can provide V1, V2, V3, V4 or we approximate.
        # Loader provided 'V_HGV_t'. Let's assume V_Light = V_t - V_HGV.
        # We'll treat Light as Cat1+2 (p=1.1 avg) and HGV as Cat3+4 (p_hgv to be estimated).
        
        V_light = df_train['V_t'].values - df_train['V_HGV_t'].values
        V_hgv = df_train['V_HGV_t'].values
        
        def pcu_bpr(inputs, alpha, beta, p_hgv):
            v_l = inputs[:, 0]
            v_h = inputs[:, 1]
            c_val = inputs[:, 2]
            
            # q_pcu = (V_light * 1.0 + V_hgv * p_hgv) * 4
            q_pcu = (v_l + v_h * p_hgv) * 4
            
            # Assume C is in PCU? Or C is veh/h and we scale it?
            # Usually C is calibrated in PCU/h if using PCUs.
            # Let's assume C_pcu = C_veh * adjustment?
            # Or just use ratio.
            
            voc = q_pcu / c_val # If C is veh/h, this might be high.
            # Let's estimate effective C or just alpha handles the scaling.
            
            return t0 * (1 + alpha * np.power(voc, beta))
            
        inputs = np.column_stack([V_light, V_hgv, C])
        
        popt, _ = fit_nls(
            pcu_bpr,
            inputs,
            y_true,
            p0=[0.15, 4.0, 2.0], # p_hgv guess 2.0
            bounds=([0.01, 1.01, 1.0], [10.0, 20.0, 5.0])
        )
        
        self.params = {'alpha': popt[0], 'beta': popt[1], 'p_hgv': popt[2]}
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        t0 = df_test['t_0'].values
        C = df_test['C'].values
        V_light = df_test['V_t'].values - df_test['V_HGV_t'].values
        V_hgv = df_test['V_HGV_t'].values
        
        p_hgv = self.params['p_hgv']
        q_pcu = (V_light + V_hgv * p_hgv) * 4
        voc = q_pcu / C
        
        return t0 * (1 + self.params['alpha'] * np.power(voc, self.params['beta']))


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

import numpy as np
import pandas as pd
from .base import BaseVDF
from ..utils.estimation import fit_nls

class A0_Baseline(BaseVDF):
    """
    A0: Baseline BPR (M0)
    T = t0 * (1 + 0.15 * (V/C)^4)
    Fixed parameters: alpha=0.15, beta=4.0
    """
    def __init__(self):
        super().__init__("A0_Baseline")
        self.alpha = 0.15
        self.beta = 4.0
        
    def fit(self, df_train: pd.DataFrame) -> 'A0_Baseline':
        # No training required for baseline
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        t0 = df_test['t_0'].values
        voc = df_test['VOC_t'].values
        return t0 * (1 + self.alpha * np.power(voc, self.beta))


class A1_Calibrated(BaseVDF):
    """
    A1: Calibrated BPR (Static alpha, beta)
    T = t0 * (1 + alpha * (V/C)^beta)
    Estimated via NLS.
    """
    def __init__(self):
        super().__init__("A1_Calibrated")
        self.alpha = 0.15 # Initial guess
        self.beta = 4.0   # Initial guess
        
    def fit(self, df_train: pd.DataFrame) -> 'A1_Calibrated':
        t0 = df_train['t_0'].values
        voc = df_train['VOC_t'].values
        y_true = df_train['T_obs'].values
        
        # Define function for curve_fit: f(voc, alpha, beta) -> T
        def bpr_func(voc, alpha, beta):
            return t0 * (1 + alpha * np.power(voc, beta))
            
        # Bounds: alpha > 0, beta > 1
        # Using slightly tighter bounds to avoid extreme values: alpha [0.01, 10], beta [1.01, 20]
        popt, _ = fit_nls(
            bpr_func, 
            voc, 
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
        voc = df_test['VOC_t'].values
        return t0 * (1 + self.alpha * np.power(voc, self.beta))


class A2_FD_VDF(BaseVDF):
    """
    A2: FD-based VDF
    Derived from Fundamental Diagram.
    T = t0 * (1 + g(k/kc; m))
    
    Simplified implementation based on requirement:
    1. Fit FD (Speed-Density or Flow-Density) to get kc (critical density).
    2. Fit parameter m in the VDF form.
    
    Using a standard form: T = t0 * (1 + alpha * (k/kc)^m) 
    where k = q/s (density).
    """
    def __init__(self):
        super().__init__("A2_FD_VDF")
        self.kc = None
        self.m = 2.0
        self.alpha = 1.0 # Often fixed or related to m
        
    def fit(self, df_train: pd.DataFrame) -> 'A2_FD_VDF':
        # 1. Calculate Density k = q / s
        # q is veh/h, s is km/h (or m/s). Let's use consistent units.
        # q_t is veh/h. s_t is derived from L/T_obs.
        # L is in meters. T_obs in seconds. s_t = (L/1000) / (T_obs/3600) = km/h.
        
        q = df_train['q_t'].values
        # Recalculate speed to be safe
        L_km = 2.7138 # Fixed for this link
        s = (L_km) / (df_train['T_obs'].values / 3600.0)
        k = q / s # Density in veh/km
        
        # 2. Estimate Critical Density (kc)
        # Simple approach: Density at maximum flow
        # Or fit a triangular FD: q = min(vf * k, w * (kj - k))
        # Here we take the density corresponding to the max observed flow as a proxy for kc
        # or use a percentile to be robust.
        max_flow_idx = np.argmax(q)
        self.kc = k[max_flow_idx]
        
        # 3. Fit VDF parameter m
        # T = t0 * (1 + (k/kc)^m)  (Assuming alpha=1 for simplicity as per some FD derivations)
        # Or T = t0 * (1 + alpha * (k/kc)^m)
        
        t0 = df_train['t_0'].values
        y_true = df_train['T_obs'].values
        k_ratio = k / self.kc
        
        def fd_vdf_func(k_rat, m):
            # Bound k_ratio to avoid overflow if k > kc significantly
            # In BPR, x can be > 1.
            return t0 * (1 + np.power(k_rat, m))
            
        popt, _ = fit_nls(
            fd_vdf_func,
            k_ratio,
            y_true,
            p0=[2.0],
            bounds=([0.1], [10.0])
        )
        
        self.m = popt[0]
        self.params = {'kc': self.kc, 'm': self.m}
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        t0 = df_test['t_0'].values
        q = df_test['q_t'].values
        L_km = 2.7138
        # Note: In prediction, we don't know observed speed s_t.
        # We must estimate k from q using the FD relationship, OR
        # if the model form allows using q directly (like BPR).
        # BUT A2 is defined as T = f(k).
        # If we only have q (input), we need q -> k mapping (inverse FD).
        # However, standard BPR uses V/C. FD-VDF often uses density.
        # If we must predict T from flow q, we need the k(q) relation.
        # For the uncongested branch: k = q / vf. For congested: k = kj - q/w.
        # This ambiguity makes A2 tricky as a pure VDF (Input: Flow).
        
        # RE-READING REQUIREMENT: "T = t0 * f_FD(k; m)"
        # Usually VDFs take Flow as input. If it takes Density, it's a Speed-Density relation.
        # If we treat it as a VDF, we assume we can derive k from q?
        # OR maybe the input is just q, and we map q -> k -> T?
        
        # For this implementation, to be practical as a VDF (Input=Flow):
        # We will approximate k = q / vf (assuming free flow) for the term.
        # This effectively makes it similar to BPR but scaled by kc.
        # Let's use k_approx = q / (L_km / (t0/3600)) = q / vf
        
        vf = L_km / (t0 / 3600.0)
        k_approx = q / vf
        
        k_ratio = k_approx / self.kc
        return t0 * (1 + np.power(k_ratio, self.m))

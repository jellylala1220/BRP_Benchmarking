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
    D3: Reliability-Oriented BPR (Quantile / P95)
    
    Per requirementsNEW.md Section 5 (D3):
    1. Aggregate by time-of-day bins (e.g. 1-hour bins)
    2. Calculate P95 travel time and mean flow per bin
    3. Fit static BPR to (mean_flow, P95_TT)
    
    This produces reliability curves rather than mean travel time.
    """
    def __init__(self):
        super().__init__("D3_Reliability")
        self.alpha_p95 = 0.15
        self.beta_p95 = 4.0
        self.bin_stats = None  # Store for diagnostics
        
    def fit(self, df_train: pd.DataFrame) -> 'D3_Reliability_ETT':
        # Step 1: Create time-of-day bins (hourly)
        df_work = df_train.copy()
        df_work['hour'] = df_work['timestamp'].dt.hour
        
        # Step 2: Aggregate by hour bins
        bin_data = []
        for hour_bin in range(24):
            bin_df = df_work[df_work['hour'] == hour_bin]
            if len(bin_df) < 3:  # Skip bins with too few samples
                continue
                
            # Calculate P95 travel time
            T_p95 = bin_df['T_obs'].quantile(0.95)
            
            # Calculate mean flow
            q_mean = bin_df['q_t'].mean()
            
            # Capacity (assumed constant)
            C = bin_df['C'].iloc[0]
            
            # Free-flow travel time (assumed constant)
            t0 = bin_df['t_0'].iloc[0]
            
            bin_data.append({'hour': hour_bin, 'T_p95': T_p95, 'q_mean': q_mean, 'C': C, 't0': t0})
        
        self.bin_stats = pd.DataFrame(bin_data)
        
        if len(self.bin_stats) < 5:
            # Fall back to global fit if not enough bins
            print("Warning: D3 insufficient bins, using global data")
            t0_val = df_train['t_0'].iloc[0]
            voc = df_train['VOC_t'].values
            T_p95_global = df_train.groupby(df_train.index // 4)['T_obs'].quantile(0.95).values
            
            def bpr_p95_global(voc_in, alpha_p, beta_p):
                return t0_val * (1 + alpha_p * np.power(voc_in, beta_p))
            
            popt, _ = fit_nls(
                bpr_p95_global,
                voc[:len(T_p95_global)],
                T_p95_global,
                p0=[0.15, 4.0],
                bounds=([0.01, 1.01], [10.0, 20.0])
            )
            
            self.alpha_p95, self.beta_p95 = popt
        else:
            # Step 3: Fit BPR to (mean_flow, P95_TT)
            t0_arr = self.bin_stats['t0'].values
            q_mean_arr = self.bin_stats['q_mean'].values
            C_arr = self.bin_stats['C'].values
            T_p95_arr = self.bin_stats['T_p95'].values
            
            voc_bin = q_mean_arr / C_arr
            
            def bpr_p95(voc_in, alpha_p, beta_p):
                return t0_arr * (1 + alpha_p * np.power(voc_in, beta_p))
            
            popt, _ = fit_nls(
                bpr_p95,
                voc_bin,
                T_p95_arr,
                p0=[0.15, 4.0],
                bounds=([0.01, 1.01], [10.0, 20.0])
            )
            
            self.alpha_p95, self.beta_p95 = popt
        
        self.params = {'alpha_p95': self.alpha_p95, 'beta_p95': self.beta_p95}
        self.is_fitted = True
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Predict P95 travel time using reliability-calibrated BPR.
        
        Note: This predicts P95, NOT mean travel time.
        For evaluation metrics to be comparable, we might need to account for this.
        """
        t0 = df_test['t_0'].values
        voc = df_test['VOC_t'].values
        
        # Predict P95 travel time
        return t0 * (1 + self.alpha_p95 * np.power(voc, self.beta_p95))

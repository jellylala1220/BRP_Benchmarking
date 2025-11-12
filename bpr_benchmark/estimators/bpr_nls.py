"""
非线性最小二乘BPR估计器

使用scipy.optimize.curve_fit拟合α和β参数
目标：最小化 Σ(T_pred - T_true)²

这是最常用的BPR参数估计方法
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from .base_estimator import BPREstimator


class BPRNLS(BPREstimator):
    """
    非线性最小二乘BPR估计器
    
    使用scipy.optimize.curve_fit直接拟合BPR公式的参数
    """
    
    def __init__(self):
        super().__init__()
        self.covariance = None
    
    def fit(self, df: pd.DataFrame, *, t0: float = None) -> 'BPRNLS':
        """
        使用非线性最小二乘法拟合α和β
        
        Args:
            df: FinalData DataFrame，必须包含：
                - v_over_c: V/C比
                - fused_tt_15min: 真实行程时间（秒）
                - t0_ff: 自由流行程时间（秒）（可选）
            t0: 自由流行程时间（秒），如果为None则从df中获取
            
        Returns:
            self
        """
        # 获取t0
        if t0 is None:
            if 't0_ff' in df.columns:
                self.t0 = df['t0_ff'].iloc[0]
            else:
                raise ValueError("必须提供t0或df中包含t0_ff列")
        else:
            self.t0 = t0
        
        # 提取数据
        v_over_c = df['v_over_c'].values
        t_true = df['fused_tt_15min'].values
        
        # 过滤无效值
        valid_mask = (v_over_c > 0) & (t_true > 0) & np.isfinite(v_over_c) & np.isfinite(t_true)
        
        if np.sum(valid_mask) < 10:
            print("警告：有效数据点太少，使用经典BPR参数")
            self.alpha = 0.15
            self.beta = 4.0
            self.is_fitted = True
            self.params = {'alpha': self.alpha, 'beta': self.beta, 't0': self.t0}
            return self
        
        v_over_c_valid = v_over_c[valid_mask]
        t_true_valid = t_true[valid_mask]
        
        # 定义BPR函数用于curve_fit
        def bpr_for_fit(x, alpha, beta):
            """BPR函数（用于curve_fit）"""
            return self.t0 * (1 + alpha * np.power(x, beta))
        
        try:
            # 使用curve_fit拟合参数
            # 初始猜测：α=0.15, β=4.0
            # 参数边界：α∈[0.01, 1.0], β∈[1.0, 10.0]
            popt, pcov = curve_fit(
                bpr_for_fit,
                v_over_c_valid,
                t_true_valid,
                p0=[0.15, 4.0],
                bounds=([0.01, 1.0], [1.0, 10.0]),
                maxfev=10000
            )
            
            self.alpha, self.beta = popt
            self.covariance = pcov
            
        except Exception as e:
            print(f"警告：NLS拟合失败 ({e})，使用经典BPR参数")
            self.alpha = 0.15
            self.beta = 4.0
        
        self.params = {
            'alpha': self.alpha,
            'beta': self.beta,
            't0': self.t0
        }
        
        self.is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        使用拟合的参数预测
        
        Args:
            df: FinalData DataFrame，必须包含v_over_c列
            
        Returns:
            预测的行程时间数组（秒）
        """
        if not self.is_fitted:
            raise ValueError("估计器未拟合，请先调用fit()")
        
        v_over_c = df['v_over_c'].values
        
        return self.bpr_function(v_over_c, self.t0, self.alpha, self.beta)
    
    def info(self) -> dict:
        """获取估计器信息（包含协方差矩阵）"""
        info_dict = super().info()
        if self.covariance is not None:
            info_dict['covariance'] = self.covariance.tolist()
        return info_dict


if __name__ == "__main__":
    """测试非线性最小二乘BPR估计器"""
    
    # 生成测试数据
    np.random.seed(42)
    n = 1000
    
    # 真实参数
    alpha_true = 0.20
    beta_true = 3.5
    t0_true = 100
    
    # 生成V/C比
    v_over_c = np.random.uniform(0.1, 1.2, n)
    
    # 生成真实行程时间（加噪声）
    t_true = t0_true * (1 + alpha_true * np.power(v_over_c, beta_true))
    t_true += np.random.normal(0, 5, n)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'v_over_c': v_over_c,
        'fused_tt_15min': t_true,
        't0_ff': t0_true
    })
    
    # 测试估计器
    estimator = BPRNLS()
    estimator.fit(df)
    
    print(f"真实参数: α={alpha_true}, β={beta_true}")
    print(f"\n非线性最小二乘BPR估计器:")
    print(estimator)
    print("\n参数信息:")
    info = estimator.info()
    for k, v in info.items():
        if k != 'covariance':
            print(f"  {k}: {v}")
    
    # 预测
    y_pred = estimator.predict(df)
    mae = np.mean(np.abs(t_true - y_pred))
    print(f"\nMAE: {mae:.4f}")


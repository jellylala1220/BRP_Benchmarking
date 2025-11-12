"""
经典BPR估计器

使用固定的经典参数：α = 0.15, β = 4.0
这是美国公路局(Bureau of Public Roads)在1964年提出的经典参数

不需要训练，直接使用固定参数预测
"""

import pandas as pd
import numpy as np
from .base_estimator import BPREstimator


class BPRClassical(BPREstimator):
    """
    经典BPR估计器
    
    使用固定参数：α = 0.15, β = 4.0
    """
    
    def __init__(self):
        super().__init__()
        # 经典BPR参数
        self.alpha = 0.15
        self.beta = 4.0
    
    def fit(self, df: pd.DataFrame, *, t0: float = None) -> 'BPRClassical':
        """
        经典BPR不需要训练，只需获取t0
        
        Args:
            df: FinalData DataFrame
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
        
        self.params = {
            'alpha': self.alpha,
            'beta': self.beta,
            't0': self.t0
        }
        
        self.is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        使用经典BPR公式预测
        
        T = t0 * (1 + 0.15 * (V/C)^4)
        
        Args:
            df: FinalData DataFrame，必须包含v_over_c列
            
        Returns:
            预测的行程时间数组（秒）
        """
        if not self.is_fitted:
            raise ValueError("估计器未拟合，请先调用fit()")
        
        v_over_c = df['v_over_c'].values
        
        return self.bpr_function(v_over_c, self.t0, self.alpha, self.beta)


if __name__ == "__main__":
    """测试经典BPR估计器"""
    
    # 生成测试数据
    np.random.seed(42)
    n = 100
    
    v_over_c = np.random.uniform(0.1, 1.2, n)
    t0 = 100
    
    df = pd.DataFrame({
        'v_over_c': v_over_c,
        't0_ff': t0
    })
    
    # 测试估计器
    estimator = BPRClassical()
    estimator.fit(df)
    
    print("经典BPR估计器:")
    print(estimator)
    print("\n参数信息:")
    print(estimator.info())
    
    # 预测
    y_pred = estimator.predict(df)
    print(f"\n预测样本: {y_pred[:5]}")


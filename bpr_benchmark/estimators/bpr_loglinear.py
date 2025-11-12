"""
对数线性BPR估计器

方法：
1. 对BPR公式取对数：log((T/t0) - 1) = log(α) + β*log(V/C)
2. 设 Y = log((T/t0) - 1), X = log(V/C)
3. 使用线性回归：Y = a + b*X
4. 反推参数：α = exp(a), β = b

优点：转换为线性问题，计算速度快，不需要初始猜测
缺点：对数转换可能引入偏差
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from .base_estimator import BPREstimator


class BPRLogLinear(BPREstimator):
    """
    对数线性BPR估计器
    
    使用对数转换将BPR参数估计转换为线性回归问题
    """
    
    def __init__(self):
        super().__init__()
        self.linear_model = LinearRegression()
    
    def fit(self, df: pd.DataFrame, *, t0: float = None) -> 'BPRLogLinear':
        """
        使用对数线性回归拟合α和β
        
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
        
        # 计算 (T/t0) - 1
        ratio = (t_true / self.t0) - 1
        
        # 过滤无效值
        # ratio必须 > 0 (即 T > t0)
        # v_over_c必须 > 0
        valid_mask = (ratio > 0) & (v_over_c > 0) & np.isfinite(ratio) & np.isfinite(v_over_c)
        
        if np.sum(valid_mask) < 10:
            print("警告：有效数据点太少，使用经典BPR参数")
            self.alpha = 0.15
            self.beta = 4.0
            self.is_fitted = True
            self.params = {'alpha': self.alpha, 'beta': self.beta, 't0': self.t0}
            return self
        
        ratio_valid = ratio[valid_mask]
        v_over_c_valid = v_over_c[valid_mask]
        
        # 对数转换
        Y = np.log(ratio_valid)
        X = np.log(v_over_c_valid).reshape(-1, 1)
        
        # 线性回归
        self.linear_model.fit(X, Y)
        
        # 提取参数
        a = self.linear_model.intercept_
        b = self.linear_model.coef_[0]
        
        # 反推BPR参数
        self.alpha = np.exp(a)
        self.beta = b
        
        # 确保参数在合理范围内
        self.alpha = np.clip(self.alpha, 0.01, 1.0)
        self.beta = np.clip(self.beta, 1.0, 10.0)
        
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


if __name__ == "__main__":
    """测试对数线性BPR估计器"""
    
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
    estimator = BPRLogLinear()
    estimator.fit(df)
    
    print(f"真实参数: α={alpha_true}, β={beta_true}")
    print(f"\n对数线性BPR估计器:")
    print(estimator)
    print("\n参数信息:")
    print(estimator.info())
    
    # 预测
    y_pred = estimator.predict(df)
    mae = np.mean(np.abs(t_true - y_pred))
    print(f"\nMAE: {mae:.4f}")


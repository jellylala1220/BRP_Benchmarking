"""
M6: 随机/可靠性BPR (Stochastic/Reliability BPR)

核心思想：
提供预测的不确定性估计，而不仅仅是点估计

方法：
- 贝叶斯回归：提供后验分布
- 分位数回归：估计特定分位数（如P90）

对应报告 M6 (不确定性)

对应论文：
- Manzo, Nielsen, & Prato (File 6)

实现：
使用原有的BayesianBPR，但统一接口
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.m6_reliability import BayesianBPR


class M6_SC_BPR:
    """
    M6: 随机/可靠性BPR模型
    
    提供不确定性估计
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.bayesian_model = None
        self.method = 'bayes'
        self.is_fitted = False
    
    def fit(self, df_train: pd.DataFrame, *, method: str = 'bayes', **kwargs) -> 'M6_SC_BPR':
        """
        训练模型
        """
        
        print(f"\n{'='*60}")
        print(f"M6_SC_BPR 训练（可靠性）")
        print(f"  方法: {method}")
        print(f"{'='*60}")
        
        self.method = method
        
        # 使用贝叶斯BPR
        t0 = df_train['t0_ff'].iloc[0] if 't0_ff' in df_train.columns else 100
        capacity = df_train['capacity'].iloc[0] if 'capacity' in df_train.columns else 6000
        
        self.bayesian_model = BayesianBPR(self.config, t0, capacity)
        self.bayesian_model.fit(df_train)
        
        self.is_fitted = True
        
        params = self.bayesian_model.get_params()
        print(f"\n拟合参数:")
        print(f"  α = {params.get('alpha', 'N/A'):.4f}")
        print(f"  β = {params.get('beta', 'N/A'):.4f}")
        print(f"  提供不确定性估计")
        
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        预测（返回中位数）
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合")
        
        return self.bayesian_model.predict(df_test, return_std=False)
    
    def predict_with_uncertainty(self, df_test: pd.DataFrame, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测（带置信区间）
        
        Returns:
            (预测值, 下界, 上界)
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合")
        
        return self.bayesian_model.predict_with_confidence(df_test, confidence)
    
    def info(self) -> Dict[str, Any]:
        info_dict = {
            'model': 'M6_SC_BPR',
            'method': self.method,
            'is_fitted': self.is_fitted
        }
        if self.is_fitted:
            info_dict['bayesian_params'] = self.bayesian_model.get_params()
        return info_dict
    
    def __repr__(self) -> str:
        if self.is_fitted:
            return f"M6_SC_BPR(method={self.method}, model={self.bayesian_model})"
        return "M6_SC_BPR(未拟合)"


if __name__ == "__main__":
    print("M6_SC_BPR: 随机/可靠性BPR模型")


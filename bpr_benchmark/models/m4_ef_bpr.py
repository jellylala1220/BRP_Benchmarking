"""
M4: 外部因素BPR (External Factors BPR)

核心思想：
考虑外部因素（天气、事故等）对BPR参数的影响

方法：
α(z) = α_0 * f(z)
t_0(z) = t_0_0 * g(z)

其中z是外部因素向量

对应报告 Eq. 4.10

对应论文：
- Klieman, Zhang, et al. (File 24)

简化实现：
使用ML方法直接学习外部因素的影响
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from estimators.base_estimator import create_estimator


class M4_EF_BPR:
    """
    M4: 外部因素BPR模型
    
    考虑外部因素的影响
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.estimator = None
        self.method = None
        self.is_fitted = False
    
    def fit(self, df_train: pd.DataFrame, *, method: str = 'rf', **kwargs) -> 'M4_EF_BPR':
        """
        训练模型
        
        使用ML方法学习外部因素影响
        """
        
        print(f"\n{'='*60}")
        print(f"M4_EF_BPR 训练（外部因素）")
        print(f"  方法: {method}")
        print(f"{'='*60}")
        
        self.method = method
        
        # 检查外部因素列
        external_factors = []
        if 'is_raining' in df_train.columns:
            external_factors.append('is_raining')
        if 'temperature' in df_train.columns:
            external_factors.append('temperature')
        
        if external_factors:
            print(f"\n使用外部因素: {external_factors}")
        else:
            print(f"\n警告：无外部因素数据")
        
        # 使用ML方法（自动学习外部因素影响）
        self.estimator = create_estimator(method, mode='direct', **kwargs)
        self.estimator.fit(df_train)
        
        self.is_fitted = True
        
        print(f"\n✓ M4_EF_BPR训练完成")
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未拟合")
        
        return self.estimator.predict(df_test)
    
    def info(self) -> Dict[str, Any]:
        return {
            'model': 'M4_EF_BPR',
            'method': self.method,
            'is_fitted': self.is_fitted
        }


if __name__ == "__main__":
    print("M4_EF_BPR: 外部因素BPR模型（简化实现）")


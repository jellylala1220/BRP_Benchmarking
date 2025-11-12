"""
M3: 多类别BPR (Multi-Class BPR)

核心思想：
考虑不同车辆类别（特别是HGV）对交通流的不同影响

方法：等效流量法
V_eq = V_car + V_hgv * PCE_hgv

其中PCE_hgv是HGV的乘用车当量系数

对应报告 Eq. 4.8

对应论文：
- Yun, White, et al. (File 7)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from estimators.base_estimator import create_estimator


class M3_MC_BPR:
    """
    M3: 多类别BPR模型
    
    使用等效流量考虑HGV影响
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.estimator = None
        self.method = None
        self.pce_hgv = 2.0  # HGV乘用车当量（默认）
        self.is_fitted = False
    
    def fit(self, df_train: pd.DataFrame, *, method: str = 'nls', **kwargs) -> 'M3_MC_BPR':
        """
        训练模型
        
        同时估计α, β和PCE_hgv
        """
        
        print(f"\n{'='*60}")
        print(f"M3_MC_BPR 训练（多类别）")
        print(f"  方法: {method}")
        print(f"{'='*60}")
        
        self.method = method
        
        # 简化：使用固定PCE，只估计α和β
        # 完整实现需要同时优化三个参数
        
        if 'hgv_share' in df_train.columns:
            # 计算等效V/C比
            df_train_eq = df_train.copy()
            df_train_eq['v_over_c'] = df_train['v_over_c'] * (1 + df_train['hgv_share'] * (self.pce_hgv - 1))
            print(f"\n使用等效V/C比（PCE_hgv={self.pce_hgv}）")
        else:
            df_train_eq = df_train
            print(f"\n警告：无HGV数据，使用原始V/C比")
        
        # 训练估计器
        self.estimator = create_estimator(method, **kwargs)
        self.estimator.fit(df_train_eq)
        
        self.is_fitted = True
        
        info = self.estimator.info()
        params = info['params']
        print(f"\n拟合参数:")
        print(f"  α = {params.get('alpha', 'N/A'):.4f}")
        print(f"  β = {params.get('beta', 'N/A'):.4f}")
        print(f"  PCE_hgv = {self.pce_hgv:.2f} (固定)")
        
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未拟合")
        
        if 'hgv_share' in df_test.columns:
            df_test_eq = df_test.copy()
            df_test_eq['v_over_c'] = df_test['v_over_c'] * (1 + df_test['hgv_share'] * (self.pce_hgv - 1))
        else:
            df_test_eq = df_test
        
        return self.estimator.predict(df_test_eq)
    
    def info(self) -> Dict[str, Any]:
        info_dict = {
            'model': 'M3_MC_BPR',
            'method': self.method,
            'pce_hgv': self.pce_hgv,
            'is_fitted': self.is_fitted
        }
        if self.is_fitted:
            info_dict['estimator_info'] = self.estimator.info()
        return info_dict


if __name__ == "__main__":
    print("M3_MC_BPR: 多类别BPR模型（简化实现）")


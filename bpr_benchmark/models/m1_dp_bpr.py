"""
M1: 动态参数BPR (Dynamic Parameter BPR)

核心思想：
BPR参数α和β在不同时段可能不同
- 高峰时段：拥堵更严重，参数可能不同
- 非高峰时段：流量平稳，参数可能不同

实现方法：
1. 按时段（高峰/非高峰）分组训练
2. 每个时段独立估计α和β
3. 预测时根据时段选择对应参数

对应论文：
- Kucharski & Drabicki (File 14)
- Chen, Zeng, et al. (File 48)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from estimators.base_estimator import create_estimator


class M1_DP_BPR:
    """
    M1: 动态参数BPR模型
    
    按时段分别估计BPR参数
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.estimator_peak = None
        self.estimator_offpeak = None
        self.method = None
        self.is_fitted = False
    
    def fit(self, df_train: pd.DataFrame, *, method: str = 'nls', **kwargs) -> 'M1_DP_BPR':
        """
        训练模型
        
        Args:
            df_train: 训练数据（必须包含is_peak列）
            method: 估计方法
            **kwargs: 传递给估计器的额外参数
        """
        
        print(f"\n{'='*60}")
        print(f"M1_DP_BPR 训练（动态参数）")
        print(f"  方法: {method}")
        print(f"{'='*60}")
        
        if 'is_peak' not in df_train.columns:
            # 如果没有is_peak列，根据hour创建
            if 'hour' in df_train.columns:
                df_train = df_train.copy()
                df_train['is_peak'] = ((df_train['hour'] >= 7) & (df_train['hour'] < 9) |
                                      (df_train['hour'] >= 15) & (df_train['hour'] < 18)).astype(int)
            else:
                raise ValueError("训练数据必须包含is_peak或hour列")
        
        self.method = method
        
        # 分割高峰和非高峰数据
        df_peak = df_train[df_train['is_peak'] == 1]
        df_offpeak = df_train[df_train['is_peak'] == 0]
        
        print(f"\n数据分割:")
        print(f"  高峰时段: {len(df_peak)} 条")
        print(f"  非高峰时段: {len(df_offpeak)} 条")
        
        # 训练高峰时段估计器
        print(f"\n训练高峰时段估计器...")
        self.estimator_peak = create_estimator(method, **kwargs)
        self.estimator_peak.fit(df_peak)
        
        info_peak = self.estimator_peak.info()
        params_peak = info_peak['params']
        print(f"  高峰参数: α={params_peak.get('alpha', 'N/A'):.4f}, β={params_peak.get('beta', 'N/A'):.4f}")
        
        # 训练非高峰时段估计器
        print(f"\n训练非高峰时段估计器...")
        self.estimator_offpeak = create_estimator(method, **kwargs)
        self.estimator_offpeak.fit(df_offpeak)
        
        info_offpeak = self.estimator_offpeak.info()
        params_offpeak = info_offpeak['params']
        print(f"  非高峰参数: α={params_offpeak.get('alpha', 'N/A'):.4f}, β={params_offpeak.get('beta', 'N/A'):.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        预测行程时间
        
        根据时段选择对应的估计器
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合")
        
        if 'is_peak' not in df_test.columns:
            if 'hour' in df_test.columns:
                df_test = df_test.copy()
                df_test['is_peak'] = ((df_test['hour'] >= 7) & (df_test['hour'] < 9) |
                                     (df_test['hour'] >= 15) & (df_test['hour'] < 18)).astype(int)
            else:
                raise ValueError("测试数据必须包含is_peak或hour列")
        
        # 初始化预测数组
        y_pred = np.zeros(len(df_test))
        
        # 高峰时段预测
        peak_mask = df_test['is_peak'] == 1
        if peak_mask.sum() > 0:
            y_pred[peak_mask] = self.estimator_peak.predict(df_test[peak_mask])
        
        # 非高峰时段预测
        offpeak_mask = df_test['is_peak'] == 0
        if offpeak_mask.sum() > 0:
            y_pred[offpeak_mask] = self.estimator_offpeak.predict(df_test[offpeak_mask])
        
        return y_pred
    
    def info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info_dict = {
            'model': 'M1_DP_BPR',
            'method': self.method,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            info_dict['peak_estimator'] = self.estimator_peak.info()
            info_dict['offpeak_estimator'] = self.estimator_offpeak.info()
        
        return info_dict
    
    def __repr__(self) -> str:
        if self.is_fitted:
            return f"M1_DP_BPR(method={self.method}, peak={self.estimator_peak}, offpeak={self.estimator_offpeak})"
        return "M1_DP_BPR(未拟合)"


if __name__ == "__main__":
    """测试M1_DP_BPR模型"""
    
    print("测试M1_DP_BPR模型（动态参数）")
    
    # 生成测试数据
    np.random.seed(42)
    n = 1000
    
    # 高峰和非高峰使用不同的参数
    alpha_peak = 0.25
    beta_peak = 4.0
    alpha_offpeak = 0.15
    beta_offpeak = 3.0
    t0 = 100
    
    v_over_c = np.random.uniform(0.1, 1.2, n)
    is_peak = np.random.choice([0, 1], n, p=[0.7, 0.3])
    
    # 根据时段使用不同参数生成数据
    t_true = np.where(
        is_peak == 1,
        t0 * (1 + alpha_peak * np.power(v_over_c, beta_peak)),
        t0 * (1 + alpha_offpeak * np.power(v_over_c, beta_offpeak))
    )
    t_true += np.random.normal(0, 5, n)
    
    df = pd.DataFrame({
        'v_over_c': v_over_c,
        'fused_tt_15min': t_true,
        't0_ff': t0,
        'is_peak': is_peak
    })
    
    # 分割数据
    split_idx = int(0.8 * n)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    # 训练模型
    model = M1_DP_BPR()
    model.fit(df_train, method='nls')
    
    # 预测
    y_pred = model.predict(df_test)
    y_true = df_test['fused_tt_15min'].values
    
    # 评估
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"\n总体MAE: {mae:.4f}")
    
    # 分时段评估
    peak_mask = df_test['is_peak'] == 1
    mae_peak = np.mean(np.abs(y_true[peak_mask] - y_pred[peak_mask]))
    mae_offpeak = np.mean(np.abs(y_true[~peak_mask] - y_pred[~peak_mask]))
    
    print(f"高峰MAE: {mae_peak:.4f}")
    print(f"非高峰MAE: {mae_offpeak:.4f}")


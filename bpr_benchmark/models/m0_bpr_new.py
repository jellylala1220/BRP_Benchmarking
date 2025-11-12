"""
M0: 基础BPR模型（重构版）

使用可插拔的估计器架构
支持多种参数估计方法：classical, loglinear, nls

这是新架构的核心示范：
- 模型形态层（M0）
- 估计方法层（estimators）
完全解耦
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from estimators.base_estimator import create_estimator, BaseEstimator


class M0_BPR:
    """
    M0: 基础BPR模型
    
    这是最基本的BPR模型形态，使用可插拔的估计器
    
    特点：
    - 只使用V/C比作为输入
    - 支持多种参数估计方法
    - 形态与估计方法解耦
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化M0_BPR模型
        
        Args:
            config: 配置字典（可选）
        """
        self.config = config or {}
        self.estimator = None
        self.method = None
        self.is_fitted = False
    
    def fit(self, df_train: pd.DataFrame, *, method: str = 'nls', **kwargs) -> 'M0_BPR':
        """
        训练模型
        
        Args:
            df_train: 训练数据（FinalData格式）
            method: 估计方法
                - 'classical': 经典BPR (α=0.15, β=4.0)
                - 'loglinear': 对数线性回归
                - 'nls': 非线性最小二乘法
            **kwargs: 传递给估计器的额外参数
            
        Returns:
            self
        """
        
        print(f"\n{'='*60}")
        print(f"M0_BPR 训练")
        print(f"  方法: {method}")
        print(f"{'='*60}")
        
        # 创建估计器
        self.method = method
        self.estimator = create_estimator(method, **kwargs)
        
        # 拟合估计器
        self.estimator.fit(df_train)
        
        self.is_fitted = True
        
        # 打印参数
        info = self.estimator.info()
        params = info['params']
        print(f"\n拟合参数:")
        print(f"  α = {params.get('alpha', 'N/A'):.4f}")
        print(f"  β = {params.get('beta', 'N/A'):.4f}")
        print(f"  t0 = {params.get('t0', 'N/A'):.2f} 秒")
        
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        预测行程时间
        
        Args:
            df_test: 测试数据（FinalData格式）
            
        Returns:
            预测的行程时间数组（秒）
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用fit()")
        
        return self.estimator.predict(df_test)
    
    def info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            包含模型和估计器信息的字典
        """
        info_dict = {
            'model': 'M0_BPR',
            'method': self.method,
            'is_fitted': self.is_fitted
        }
        
        if self.estimator:
            info_dict['estimator_info'] = self.estimator.info()
        
        return info_dict
    
    def __repr__(self) -> str:
        """字符串表示"""
        if self.is_fitted:
            return f"M0_BPR(method={self.method}, estimator={self.estimator})"
        else:
            return "M0_BPR(未拟合)"


if __name__ == "__main__":
    """测试M0_BPR模型"""
    
    print("测试M0_BPR模型（新架构）")
    print("="*60)
    
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
    
    # 创建FinalData格式的DataFrame
    df = pd.DataFrame({
        'v_over_c': v_over_c,
        'fused_tt_15min': t_true,
        't0_ff': t0_true
    })
    
    # 分割数据
    split_idx = int(0.8 * n)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    print(f"\n数据集:")
    print(f"  训练集: {len(df_train)} 条")
    print(f"  测试集: {len(df_test)} 条")
    print(f"  真实参数: α={alpha_true}, β={beta_true}, t0={t0_true}")
    
    # 测试三种方法
    methods = ['classical', 'loglinear', 'nls']
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"测试方法: {method}")
        print(f"{'='*60}")
        
        # 创建和训练模型
        model = M0_BPR()
        model.fit(df_train, method=method)
        
        # 预测
        y_pred = model.predict(df_test)
        y_true = df_test['fused_tt_15min'].values
        
        # 评估
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        results[method] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        print(f"\n评估结果:")
        print(f"  MAE:  {mae:.4f} 秒")
        print(f"  RMSE: {rmse:.4f} 秒")
        print(f"  MAPE: {mape:.2f} %")
        
        # 打印模型信息
        print(f"\n模型信息:")
        info = model.info()
        print(f"  {info}")
    
    # 对比结果
    print(f"\n{'='*60}")
    print("方法对比")
    print(f"{'='*60}")
    print(f"\n{'方法':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-"*60)
    for method, metrics in results.items():
        print(f"{method:<15} {metrics['MAE']:<10.4f} {metrics['RMSE']:<10.4f} {metrics['MAPE']:<10.2f}")
    
    # 找出最佳方法
    best_method = min(results.items(), key=lambda x: x[1]['MAE'])
    print(f"\n最佳方法: {best_method[0]} (MAE = {best_method[1]['MAE']:.4f})")


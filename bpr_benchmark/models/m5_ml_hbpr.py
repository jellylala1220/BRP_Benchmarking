"""
M5: ML混合BPR (Machine Learning Hybrid BPR)

核心思想：
两阶段方法 - BPR基础 + ML残差修正

阶段1: 使用BPR模型（M0/M1）得到基础预测 t_bpr
阶段2: 使用ML模型学习残差 Δ = t_true - t_bpr
最终预测: t_final = t_bpr + Δ

对应报告 Eq. 4.13: 残差修正法

优点：
- 结合BPR的物理意义和ML的学习能力
- ML只需学习残差，任务更简单
- 可以捕捉BPR无法建模的复杂模式
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from estimators.base_estimator import create_estimator
from models.m0_bpr_new import M0_BPR


class M5_ML_HBPR:
    """
    M5: ML混合BPR模型
    
    两阶段方法：
    1. BPR基础预测
    2. ML残差修正
    """
    
    def __init__(self, config: Dict = None, base_model='M0', base_method='nls'):
        """
        初始化M5模型
        
        Args:
            config: 配置字典
            base_model: 基础模型类型 ('M0' 或 'M1')
            base_method: 基础模型使用的估计方法
        """
        self.config = config or {}
        self.base_model_type = base_model
        self.base_method = base_method
        self.base_model = None
        self.ml_estimator = None
        self.ml_method = None
        self.is_fitted = False
    
    def fit(self, df_train: pd.DataFrame, *, method: str = 'gbdt', **kwargs) -> 'M5_ML_HBPR':
        """
        训练混合模型
        
        Args:
            df_train: 训练数据
            method: ML方法 (svr, tree, rf, gbdt, nn)
            **kwargs: 传递给ML估计器的参数
        """
        
        print(f"\n{'='*60}")
        print(f"M5_ML_HBPR 训练（混合模型）")
        print(f"  基础模型: {self.base_model_type} ({self.base_method})")
        print(f"  ML方法: {method}")
        print(f"{'='*60}")
        
        self.ml_method = method
        
        # 阶段1: 训练基础BPR模型
        print(f"\n[阶段1] 训练基础BPR模型...")
        if self.base_model_type == 'M0':
            self.base_model = M0_BPR(self.config)
        elif self.base_model_type == 'M1':
            from models.m1_dp_bpr import M1_DP_BPR
            self.base_model = M1_DP_BPR(self.config)
        else:
            raise ValueError(f"不支持的基础模型: {self.base_model_type}")
        
        self.base_model.fit(df_train, method=self.base_method)
        
        # 获取BPR预测
        t_bpr = self.base_model.predict(df_train)
        t_true = df_train['fused_tt_15min'].values
        
        # 计算残差
        residuals = t_true - t_bpr
        
        print(f"\n[阶段1] BPR基础预测完成")
        print(f"  BPR MAE: {np.mean(np.abs(residuals)):.4f} 秒")
        print(f"  残差范围: [{residuals.min():.2f}, {residuals.max():.2f}]")
        
        # 阶段2: 训练ML残差模型
        print(f"\n[阶段2] 训练ML残差模型...")
        
        # 创建包含残差的训练数据
        df_train_ml = df_train.copy()
        df_train_ml['fused_tt_15min'] = residuals  # 用残差替换目标变量
        
        # 创建ML估计器（residual模式）
        self.ml_estimator = create_estimator(method, mode='residual', **kwargs)
        self.ml_estimator.fit(df_train_ml)
        
        # 验证残差预测
        residual_pred = self.ml_estimator.predict(df_train_ml)
        residual_mae = np.mean(np.abs(residuals - residual_pred))
        
        print(f"\n[阶段2] ML残差模型完成")
        print(f"  残差预测MAE: {residual_mae:.4f} 秒")
        
        # 最终预测
        t_final = t_bpr + residual_pred
        final_mae = np.mean(np.abs(t_true - t_final))
        
        print(f"\n[最终] 混合模型训练完成")
        print(f"  混合模型MAE: {final_mae:.4f} 秒")
        print(f"  改进: {np.mean(np.abs(residuals)) - final_mae:.4f} 秒")
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        预测行程时间
        
        t_final = t_bpr + Δ_ml
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合")
        
        # BPR基础预测
        t_bpr = self.base_model.predict(df_test)
        
        # ML残差预测
        residual_pred = self.ml_estimator.predict(df_test)
        
        # 最终预测
        t_final = t_bpr + residual_pred
        
        return t_final
    
    def info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info_dict = {
            'model': 'M5_ML_HBPR',
            'base_model': self.base_model_type,
            'base_method': self.base_method,
            'ml_method': self.ml_method,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            info_dict['base_model_info'] = self.base_model.info()
            info_dict['ml_estimator_info'] = self.ml_estimator.info()
        
        return info_dict
    
    def __repr__(self) -> str:
        if self.is_fitted:
            return f"M5_ML_HBPR(base={self.base_model_type}+{self.base_method}, ml={self.ml_method})"
        return "M5_ML_HBPR(未拟合)"


if __name__ == "__main__":
    """测试M5_ML_HBPR模型"""
    
    print("测试M5_ML_HBPR模型（混合模型）")
    
    # 生成测试数据
    np.random.seed(42)
    n = 1000
    
    alpha_true = 0.20
    beta_true = 3.5
    t0 = 100
    
    v_over_c = np.random.uniform(0.1, 1.2, n)
    hgv_share = np.random.uniform(0, 0.3, n)
    hour = np.random.randint(0, 24, n)
    
    # 生成真实行程时间（复杂模式）
    t_true = t0 * (1 + alpha_true * np.power(v_over_c, beta_true))
    t_true *= (1 + 0.1 * hgv_share)  # HGV效应
    t_true *= (1 + 0.05 * ((hour >= 7) & (hour < 9) | (hour >= 15) & (hour < 18)))  # 高峰效应
    t_true += np.random.normal(0, 5, n)
    
    df = pd.DataFrame({
        'v_over_c': v_over_c,
        'fused_tt_15min': t_true,
        't0_ff': t0,
        'hgv_share': hgv_share,
        'hour': hour,
        'weekday': 1,
        'is_peak': ((hour >= 7) & (hour < 9) | (hour >= 15) & (hour < 18)).astype(int)
    })
    
    # 分割数据
    split_idx = int(0.8 * n)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    # 训练模型
    model = M5_ML_HBPR(base_model='M0', base_method='nls')
    model.fit(df_train, method='gbdt')
    
    # 预测
    y_pred = model.predict(df_test)
    y_true = df_test['fused_tt_15min'].values
    
    # 评估
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    print(f"\n最终评估:")
    print(f"  MAE: {mae:.4f} 秒")
    print(f"  RMSE: {rmse:.4f} 秒")


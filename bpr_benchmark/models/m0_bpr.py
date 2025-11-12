"""
M0: BPR参数估计器

实现三种基本的BPR参数估计方法：
1. ClassicalBPR - 经典BPR (α=0.15, β=4.0)
2. NLS_BPR - 非线性最小二乘法
3. LogLinearBPR - 对数线性回归

对应报告中的基线方法
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from typing import Dict

from .base import BPRModel


class ClassicalBPR(BPRModel):
    """
    经典BPR模型
    
    使用固定参数：α = 0.15, β = 4.0
    这是美国公路局(Bureau of Public Roads)在1964年提出的经典参数
    
    不需要训练，直接使用固定参数预测
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        super().__init__(config, t_0, capacity)
        # 经典BPR参数
        self.alpha = 0.15
        self.beta = 4.0
    
    def fit(self, df_train: pd.DataFrame) -> 'ClassicalBPR':
        """
        经典BPR不需要训练
        
        Args:
            df_train: 训练数据（不使用）
            
        Returns:
            self
        """
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        使用经典BPR公式预测
        
        T = t_0 * (1 + 0.15 * (V/C)^4)
        
        Args:
            df_test: 测试数据
            
        Returns:
            预测的行程时间
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 fit()")
        
        vcr = df_test['V_C_Ratio'].values
        t_0 = df_test['t_0'].values if 't_0' in df_test.columns else self.t_0
        
        return self.bpr_function(vcr, t_0, self.alpha, self.beta)


class NLS_BPR(BPRModel):
    """
    非线性最小二乘法BPR
    
    使用scipy.optimize.curve_fit拟合α和β参数
    目标：最小化 Σ(T_pred - T_true)²
    
    这是最常用的BPR参数估计方法
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        super().__init__(config, t_0, capacity)
    
    def fit(self, df_train: pd.DataFrame) -> 'NLS_BPR':
        """
        使用非线性最小二乘法拟合α和β
        
        Args:
            df_train: 训练数据
            
        Returns:
            self
        """
        
        # 提取数据
        vcr = df_train['V_C_Ratio'].values
        t_true = df_train['t_ground_truth'].values
        t_0 = df_train['t_0'].values if 't_0' in df_train.columns else self.t_0
        
        # 定义BPR函数用于curve_fit
        def bpr_for_fit(vcr, alpha, beta):
            """
            BPR函数（用于curve_fit）
            
            注意：这里假设t_0是常数
            """
            if isinstance(t_0, np.ndarray):
                return t_0 * (1 + alpha * np.power(vcr, beta))
            else:
                return t_0 * (1 + alpha * np.power(vcr, beta))
        
        try:
            # 使用curve_fit拟合参数
            # 初始猜测：α=0.15, β=4.0
            # 参数边界：α∈[0.01, 1.0], β∈[1.0, 10.0]
            popt, pcov = curve_fit(
                bpr_for_fit,
                vcr,
                t_true,
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
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        使用拟合的参数预测
        
        Args:
            df_test: 测试数据
            
        Returns:
            预测的行程时间
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 fit()")
        
        vcr = df_test['V_C_Ratio'].values
        t_0 = df_test['t_0'].values if 't_0' in df_test.columns else self.t_0
        
        return self.bpr_function(vcr, t_0, self.alpha, self.beta)


class LogLinearBPR(BPRModel):
    """
    对数线性回归BPR
    
    方法：
    1. 对BPR公式取对数：log((T/t_0) - 1) = log(α) + β*log(V/C)
    2. 设 Y = log((T/t_0) - 1), X = log(V/C)
    3. 使用线性回归：Y = a + b*X
    4. 反推参数：α = exp(a), β = b
    
    优点：转换为线性问题，计算速度快，不需要初始猜测
    缺点：对数转换可能引入偏差
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        super().__init__(config, t_0, capacity)
        self.linear_model = LinearRegression()
    
    def fit(self, df_train: pd.DataFrame) -> 'LogLinearBPR':
        """
        使用对数线性回归拟合α和β
        
        Args:
            df_train: 训练数据
            
        Returns:
            self
        """
        
        # 提取数据
        vcr = df_train['V_C_Ratio'].values
        t_true = df_train['t_ground_truth'].values
        t_0 = df_train['t_0'].values if 't_0' in df_train.columns else self.t_0
        
        # 计算 (T/t_0) - 1
        if isinstance(t_0, np.ndarray):
            ratio = (t_true / t_0) - 1
        else:
            ratio = (t_true / t_0) - 1
        
        # 过滤无效值
        # ratio必须 > 0 (即 T > t_0)
        # vcr必须 > 0
        valid_mask = (ratio > 0) & (vcr > 0)
        
        if np.sum(valid_mask) < 10:
            print("警告：有效数据点太少，使用经典BPR参数")
            self.alpha = 0.15
            self.beta = 4.0
            self.is_fitted = True
            return self
        
        ratio_valid = ratio[valid_mask]
        vcr_valid = vcr[valid_mask]
        
        # 对数转换
        Y = np.log(ratio_valid)
        X = np.log(vcr_valid).reshape(-1, 1)
        
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
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        使用拟合的参数预测
        
        Args:
            df_test: 测试数据
            
        Returns:
            预测的行程时间
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 fit()")
        
        vcr = df_test['V_C_Ratio'].values
        t_0 = df_test['t_0'].values if 't_0' in df_test.columns else self.t_0
        
        return self.bpr_function(vcr, t_0, self.alpha, self.beta)


if __name__ == "__main__":
    """测试BPR模型"""
    
    # 生成模拟数据
    np.random.seed(42)
    n = 1000
    
    # 真实参数
    alpha_true = 0.20
    beta_true = 3.5
    t_0_true = 100
    
    # 生成V/C比
    vcr = np.random.uniform(0.1, 1.2, n)
    
    # 生成真实行程时间（加噪声）
    t_true = t_0_true * (1 + alpha_true * np.power(vcr, beta_true))
    t_true += np.random.normal(0, 5, n)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'V_C_Ratio': vcr,
        't_ground_truth': t_true,
        't_0': t_0_true
    })
    
    # 分割训练/测试集
    split_idx = int(0.8 * n)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    config = {}
    
    # 测试三种模型
    models = [
        ClassicalBPR(config, t_0_true, 6000),
        NLS_BPR(config, t_0_true, 6000),
        LogLinearBPR(config, t_0_true, 6000)
    ]
    
    print(f"真实参数: α={alpha_true}, β={beta_true}\n")
    
    for model in models:
        # 训练
        model.fit(df_train)
        
        # 预测
        y_pred = model.predict(df_test)
        y_true = df_test['t_ground_truth'].values
        
        # 评估
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        print(f"{model.__class__.__name__}:")
        print(f"  拟合参数: α={model.alpha:.4f}, β={model.beta:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print()


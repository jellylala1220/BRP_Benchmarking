"""
M6: 可靠性/不确定性模型

实现贝叶斯回归方法，用于估计预测的不确定性

对应报告中的 M6 (Stochastic/Reliability BPR)
核心论文: File 6 (Manzo, Nielsen, & Prato) - 讨论参数不确定性的影响
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from .base import BPRModel


class BayesianBPR(BPRModel):
    """
    贝叶斯BPR模型
    
    使用贝叶斯线性回归在对数空间中估计BPR参数
    
    方法：
    1. 与LogLinearBPR类似，进行对数转换
    2. 使用BayesianRidge而不是普通线性回归
    3. 可以返回预测的置信区间
    
    优点：
    - 提供预测不确定性的估计
    - 对过拟合有更好的抵抗力
    - 适合M6的"可靠性"目标
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float):
        super().__init__(config, t_0, capacity)
        
        # 贝叶斯岭回归模型
        self.bayesian_model = BayesianRidge(
            n_iter=300,
            tol=1e-3,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
            compute_score=True
        )
        
        # 用于存储对数空间的标准差
        self.log_std = None
    
    def fit(self, df_train: pd.DataFrame) -> 'BayesianBPR':
        """
        使用贝叶斯回归拟合α和β
        
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
        
        # 贝叶斯回归
        self.bayesian_model.fit(X, Y)
        
        # 提取参数
        a = self.bayesian_model.intercept_
        b = self.bayesian_model.coef_[0]
        
        # 反推BPR参数
        self.alpha = np.exp(a)
        self.beta = b
        
        # 确保参数在合理范围内
        self.alpha = np.clip(self.alpha, 0.01, 1.0)
        self.beta = np.clip(self.beta, 1.0, 10.0)
        
        # 计算残差标准差（用于预测区间）
        Y_pred = self.bayesian_model.predict(X)
        residuals = Y - Y_pred
        self.log_std = np.std(residuals)
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame, return_std: bool = False) -> np.ndarray:
        """
        预测行程时间
        
        Args:
            df_test: 测试数据
            return_std: 是否返回标准差（用于置信区间）
            
        Returns:
            如果return_std=False: 预测的行程时间
            如果return_std=True: (预测值, 标准差) 元组
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 fit()")
        
        vcr = df_test['V_C_Ratio'].values
        t_0 = df_test['t_0'].values if 't_0' in df_test.columns else self.t_0
        
        # 基础预测
        predictions = self.bpr_function(vcr, t_0, self.alpha, self.beta)
        
        if not return_std:
            return predictions
        
        # 计算标准差（如果需要）
        # 这是一个简化的估计，基于对数空间的残差标准差
        std_estimates = predictions * self.log_std if self.log_std else np.zeros_like(predictions)
        
        return predictions, std_estimates
    
    def predict_with_confidence(self, df_test: pd.DataFrame, 
                               confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测行程时间及其置信区间
        
        Args:
            df_test: 测试数据
            confidence: 置信水平（默认95%）
            
        Returns:
            (预测值, 下界, 上界) 元组
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 fit()")
        
        predictions, std = self.predict(df_test, return_std=True)
        
        # 计算置信区间
        # 使用正态分布的分位数
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower_bound = predictions - z_score * std
        upper_bound = predictions + z_score * std
        
        # 确保下界非负
        lower_bound = np.maximum(lower_bound, 0)
        
        return predictions, lower_bound, upper_bound
    
    def get_params(self) -> Dict:
        """获取模型参数"""
        params = super().get_params()
        
        if self.is_fitted:
            params['log_std'] = self.log_std
            params['alpha_precision'] = self.bayesian_model.alpha_
            params['lambda_precision'] = self.bayesian_model.lambda_
        
        return params


class QuantileBPR(BPRModel):
    """
    分位数回归BPR（可选实现）
    
    使用分位数回归估计特定分位数的行程时间
    例如：90%分位数用于可靠性分析
    
    对应报告 Eq. 4.15 的分位数回归方法
    
    注意：这是一个高级实现，需要额外的优化
    """
    
    def __init__(self, config: Dict, t_0: float, capacity: float, quantile: float = 0.9):
        super().__init__(config, t_0, capacity)
        self.quantile = quantile
    
    def pinball_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        分位数损失函数（Pinball Loss）
        
        ρ_p(u) = u * (p - I{u<0})
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            损失值
        """
        residual = y_true - y_pred
        return np.mean(np.where(residual >= 0, 
                                self.quantile * residual, 
                                (self.quantile - 1) * residual))
    
    def fit(self, df_train: pd.DataFrame) -> 'QuantileBPR':
        """
        使用分位数回归拟合α和β
        
        Args:
            df_train: 训练数据
            
        Returns:
            self
        """
        from scipy.optimize import minimize
        
        # 提取数据
        vcr = df_train['V_C_Ratio'].values
        t_true = df_train['t_ground_truth'].values
        t_0 = df_train['t_0'].values if 't_0' in df_train.columns else self.t_0
        
        # 定义目标函数
        def objective(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return 1e10
            
            t_pred = self.bpr_function(vcr, t_0, alpha, beta)
            return self.pinball_loss(t_true, t_pred)
        
        # 优化
        result = minimize(
            objective,
            x0=[0.15, 4.0],
            method='L-BFGS-B',
            bounds=[(0.01, 1.0), (1.0, 10.0)]
        )
        
        if result.success:
            self.alpha, self.beta = result.x
        else:
            print(f"警告：分位数回归优化失败，使用经典BPR参数")
            self.alpha = 0.15
            self.beta = 4.0
        
        self.is_fitted = True
        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        预测指定分位数的行程时间
        
        Args:
            df_test: 测试数据
            
        Returns:
            预测的行程时间（第p分位数）
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 fit()")
        
        vcr = df_test['V_C_Ratio'].values
        t_0 = df_test['t_0'].values if 't_0' in df_test.columns else self.t_0
        
        return self.bpr_function(vcr, t_0, self.alpha, self.beta)


if __name__ == "__main__":
    """测试可靠性模型"""
    
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
    t_true += np.random.normal(0, 10, n)  # 较大的噪声
    
    # 创建DataFrame
    df = pd.DataFrame({
        'V_C_Ratio': vcr,
        't_ground_truth': t_true,
        't_0': t_0_true
    })
    
    # 分割数据
    split_idx = int(0.8 * n)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    config = {}
    
    # 测试贝叶斯BPR
    print("=" * 60)
    print("测试贝叶斯BPR")
    print("=" * 60)
    
    model = BayesianBPR(config, t_0_true, 6000)
    model.fit(df_train)
    
    print(f"\n拟合参数: α={model.alpha:.4f}, β={model.beta:.4f}")
    print(f"真实参数: α={alpha_true:.4f}, β={beta_true:.4f}")
    
    # 预测（带标准差）
    y_pred, y_std = model.predict(df_test, return_std=True)
    y_true = df_test['t_ground_truth'].values
    
    # 评估
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    print(f"\nMAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"平均标准差: {np.mean(y_std):.4f}")
    
    # 置信区间
    y_pred_ci, y_lower, y_upper = model.predict_with_confidence(df_test, confidence=0.95)
    
    # 计算覆盖率（真实值落在置信区间内的比例）
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    print(f"95% 置信区间覆盖率: {coverage:.2%}")
    
    # 测试分位数BPR
    print("\n" + "=" * 60)
    print("测试分位数BPR (90%分位数)")
    print("=" * 60)
    
    quantile_model = QuantileBPR(config, t_0_true, 6000, quantile=0.9)
    quantile_model.fit(df_train)
    
    print(f"\n拟合参数: α={quantile_model.alpha:.4f}, β={quantile_model.beta:.4f}")
    
    y_pred_q90 = quantile_model.predict(df_test)
    
    # 计算实际的90%分位数覆盖率
    coverage_q90 = np.mean(y_true <= y_pred_q90)
    print(f"90% 分位数覆盖率: {coverage_q90:.2%} (理想值: 90%)")


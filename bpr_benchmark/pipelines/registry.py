"""
模型和估计器注册表

职责：
- 维护所有可用模型的注册表
- 维护所有可用估计器的注册表
- 提供统一的创建接口
"""

from typing import Dict, List, Any
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))


# ========== 估计器注册表 ==========

ESTIMATORS = {
    # BPR参数估计方法
    'classical': {
        'name': 'Classical BPR',
        'type': 'BPR',
        'description': '经典BPR参数 (α=0.15, β=4.0)',
        'module': 'estimators.bpr_classical',
        'class': 'BPRClassical'
    },
    'loglinear': {
        'name': 'Log-Linear BPR',
        'type': 'BPR',
        'description': '对数线性回归',
        'module': 'estimators.bpr_loglinear',
        'class': 'BPRLogLinear'
    },
    'nls': {
        'name': 'NLS BPR',
        'type': 'BPR',
        'description': '非线性最小二乘法',
        'module': 'estimators.bpr_nls',
        'class': 'BPRNLS'
    },
    
    'svr': {
        'name': 'SVR',
        'type': 'ML',
        'description': '支持向量回归',
        'module': 'estimators.ml_svr',
        'class': 'SVREstimator'
    },
    'tree': {
        'name': 'Decision Tree',
        'type': 'ML',
        'description': '决策树',
        'module': 'estimators.ml_tree',
        'class': 'TreeEstimator'
    },
    'rf': {
        'name': 'Random Forest',
        'type': 'ML',
        'description': '随机森林',
        'module': 'estimators.ml_rf',
        'class': 'RFEstimator'
    },
    'gbdt': {
        'name': 'Gradient Boosting',
        'type': 'ML',
        'description': '梯度提升',
        'module': 'estimators.ml_gbdt',
        'class': 'GBDTEstimator'
    },
    'nn': {
        'name': 'Neural Network',
        'type': 'ML',
        'description': '神经网络',
        'module': 'estimators.ml_nn',
        'class': 'NNEstimator'
    },
    'bayes': {
        'name': 'Bayesian',
        'type': 'Reliability',
        'description': '贝叶斯回归',
        'module': 'estimators.bpr_loglinear',
        'class': 'BPRLogLinear'
    },
}


# ========== 模型注册表 ==========

MODELS = {
    'M0': {
        'name': 'M0_BPR',
        'description': '基础BPR模型',
        'module': 'models.m0_bpr_new',
        'class': 'M0_BPR',
        'compatible_methods': ['classical', 'loglinear', 'nls'],
        'default_method': 'nls'
    },
    
    'M1': {
        'name': 'M1_DP_BPR',
        'description': '动态参数BPR',
        'module': 'models.m1_dp_bpr',
        'class': 'M1_DP_BPR',
        'compatible_methods': ['classical', 'loglinear', 'nls'],
        'default_method': 'nls'
    },
    
    'M2': {
        'name': 'M2_FD_VDF',
        'description': '基本图VDF',
        'module': 'models.m2_fd_vdf',
        'class': 'M2_FD_VDF',
        'compatible_methods': ['tree', 'rf', 'gbdt'],
        'default_method': 'tree'
    },
    
    'M3': {
        'name': 'M3_MC_BPR',
        'description': '多类别BPR',
        'module': 'models.m3_mc_bpr',
        'class': 'M3_MC_BPR',
        'compatible_methods': ['classical', 'loglinear', 'nls'],
        'default_method': 'nls'
    },
    
    'M4': {
        'name': 'M4_EF_BPR',
        'description': '外部因素BPR',
        'module': 'models.m4_ef_bpr',
        'class': 'M4_EF_BPR',
        'compatible_methods': ['tree', 'rf', 'gbdt'],
        'default_method': 'rf'
    },
    
    'M5': {
        'name': 'M5_ML_HBPR',
        'description': 'ML混合BPR',
        'module': 'models.m5_ml_hbpr',
        'class': 'M5_ML_HBPR',
        'compatible_methods': ['svr', 'tree', 'rf', 'gbdt', 'nn'],
        'default_method': 'gbdt'
    },
    
    'M6': {
        'name': 'M6_SC_BPR',
        'description': '随机/可靠性BPR',
        'module': 'models.m6_sc_bpr',
        'class': 'M6_SC_BPR',
        'compatible_methods': ['bayes'],
        'default_method': 'bayes'
    },
}


def get_available_estimators() -> List[str]:
    """获取所有可用的估计器名称"""
    return list(ESTIMATORS.keys())


def get_available_models() -> List[str]:
    """获取所有可用的模型名称"""
    return list(MODELS.keys())


def get_estimator_info(method: str) -> Dict[str, Any]:
    """获取估计器信息"""
    if method not in ESTIMATORS:
        raise ValueError(f"未知的估计方法: {method}")
    return ESTIMATORS[method]


def get_model_info(model_name: str) -> Dict[str, Any]:
    """获取模型信息"""
    if model_name not in MODELS:
        raise ValueError(f"未知的模型: {model_name}")
    return MODELS[model_name]


def get_compatible_methods(model_name: str) -> List[str]:
    """获取模型兼容的估计方法"""
    model_info = get_model_info(model_name)
    return model_info.get('compatible_methods', [])


def create_model(model_name: str, config: Dict = None):
    """
    创建模型实例
    
    Args:
        model_name: 模型名称 (如 'M0')
        config: 配置字典
        
    Returns:
        模型实例
    """
    model_info = get_model_info(model_name)
    
    # 动态导入
    module_name = model_info['module']
    class_name = model_info['class']
    
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)
    
    return model_class(config)


def print_registry():
    """打印注册表信息"""
    print("\n" + "="*60)
    print("BPR框架注册表")
    print("="*60)
    
    print("\n可用估计器:")
    print("-"*60)
    for method, info in ESTIMATORS.items():
        print(f"  {method:<12} - {info['name']:<20} ({info['description']})")
    
    print("\n可用模型:")
    print("-"*60)
    for model_name, info in MODELS.items():
        methods_str = ', '.join(info['compatible_methods'])
        print(f"  {model_name:<5} - {info['name']:<15} - 兼容方法: {methods_str}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    """测试注册表"""
    
    print_registry()
    
    print("\n测试模型创建:")
    try:
        model = create_model('M0')
        print(f"  ✓ 成功创建 {model}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")


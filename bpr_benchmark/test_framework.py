"""
框架测试脚本

用于验证BPR基准测试框架的各个组件是否正常工作

使用方法：
    python test_framework.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """测试所有模块是否可以正常导入"""
    print("=" * 60)
    print("测试1: 模块导入")
    print("=" * 60)
    
    try:
        from utils import data, metrics
        print("✓ utils 模块导入成功")
    except Exception as e:
        print(f"✗ utils 模块导入失败: {e}")
        return False
    
    try:
        from models import base, m0_bpr, m5_ml, m6_reliability
        print("✓ models 模块导入成功")
    except Exception as e:
        print(f"✗ models 模块导入失败: {e}")
        return False
    
    try:
        import yaml
        print("✓ yaml 导入成功")
    except Exception as e:
        print(f"✗ yaml 导入失败: {e}")
        print("  请运行: pip install pyyaml")
        return False
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        print("✓ scikit-learn 导入成功")
    except Exception as e:
        print(f"✗ scikit-learn 导入失败: {e}")
        print("  请运行: pip install scikit-learn")
        return False
    
    print("\n✓ 所有模块导入成功！\n")
    return True


def test_config():
    """测试配置文件是否可以正常加载"""
    print("=" * 60)
    print("测试2: 配置文件")
    print("=" * 60)
    
    try:
        from utils.data import load_config
        config = load_config("configs/default.yaml")
        print("✓ 配置文件加载成功")
        
        # 检查关键配置项
        assert 'data' in config, "缺少 'data' 配置"
        assert 'roads' in config, "缺少 'roads' 配置"
        assert 'models_to_run' in config, "缺少 'models_to_run' 配置"
        
        print(f"  - 数据文件: {config['data']['precleaned_file']}")
        print(f"  - 测试路段: {config['roads_to_test']}")
        print(f"  - 测试模型: {len(config['models_to_run'])} 个")
        
        print("\n✓ 配置文件验证通过！\n")
        return True
        
    except Exception as e:
        print(f"✗ 配置文件测试失败: {e}")
        return False


def test_models():
    """测试模型是否可以正常创建和运行"""
    print("=" * 60)
    print("测试3: 模型创建和运行")
    print("=" * 60)
    
    try:
        from models.base import create_model
        from utils.data import load_config
        
        config = load_config("configs/default.yaml")
        
        # 生成模拟数据
        np.random.seed(42)
        n = 100
        
        vcr = np.random.uniform(0.1, 1.2, n)
        t_0 = 100
        t_true = t_0 * (1 + 0.2 * np.power(vcr, 3.5))
        t_true += np.random.normal(0, 5, n)
        
        df = pd.DataFrame({
            'V_C_Ratio': vcr,
            't_ground_truth': t_true,
            't_0': t_0,
            'p_H': np.random.uniform(0, 0.3, n),
            'is_peak': np.random.choice([0, 1], n),
            'is_weekday': 1,
            'hour': np.random.randint(0, 24, n),
            'is_raining': 0,
            'temperature': 20
        })
        
        df_train = df.iloc[:80]
        df_test = df.iloc[80:]
        
        # 测试每个模型
        test_models = ['ClassicalBPR', 'NLS_BPR', 'LogLinearBPR', 
                      'RandomForest', 'BayesianBPR']
        
        for model_name in test_models:
            try:
                model = create_model(model_name, config, t_0=100, capacity=6000)
                model.fit(df_train)
                y_pred = model.predict(df_test)
                
                assert len(y_pred) == len(df_test), f"{model_name}: 预测长度不匹配"
                assert not np.any(np.isnan(y_pred)), f"{model_name}: 预测包含NaN"
                
                print(f"  ✓ {model_name} 测试通过")
                
            except Exception as e:
                print(f"  ✗ {model_name} 测试失败: {e}")
                return False
        
        print("\n✓ 所有模型测试通过！\n")
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """测试评估指标计算"""
    print("=" * 60)
    print("测试4: 评估指标")
    print("=" * 60)
    
    try:
        from utils.metrics import (
            calculate_all_metrics,
            calculate_stratified_metrics,
            create_metrics_comparison_table
        )
        
        # 生成测试数据
        np.random.seed(42)
        n = 100
        
        y_true = np.random.uniform(100, 300, n)
        y_pred = y_true + np.random.normal(0, 20, n)
        vcr = np.random.uniform(0, 1.5, n)
        
        # 测试基本指标
        metrics = calculate_all_metrics(y_true, y_pred)
        
        assert 'MAE' in metrics, "缺少 MAE"
        assert 'RMSE' in metrics, "缺少 RMSE"
        assert 'MAPE' in metrics, "缺少 MAPE"
        assert 'R2' in metrics, "缺少 R2"
        
        print(f"  ✓ 基本指标计算成功")
        print(f"    MAE: {metrics['MAE']:.2f}")
        print(f"    RMSE: {metrics['RMSE']:.2f}")
        print(f"    MAPE: {metrics['MAPE']:.2f}%")
        print(f"    R²: {metrics['R2']:.4f}")
        
        # 测试分层指标
        stratified = calculate_stratified_metrics(y_true, y_pred, vcr)
        assert len(stratified) > 0, "分层指标为空"
        print(f"  ✓ 分层指标计算成功 ({len(stratified)} 层)")
        
        # 测试对比表
        results = {
            'Model1': metrics,
            'Model2': metrics
        }
        comparison = create_metrics_comparison_table(results)
        assert len(comparison) == 2, "对比表行数不正确"
        print(f"  ✓ 对比表生成成功")
        
        print("\n✓ 评估指标测试通过！\n")
        return True
        
    except Exception as e:
        print(f"✗ 评估指标测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_file():
    """测试数据文件是否存在和可读"""
    print("=" * 60)
    print("测试5: 数据文件")
    print("=" * 60)
    
    try:
        from utils.data import load_config
        
        config = load_config("configs/default.yaml")
        data_file = Path(__file__).parent / config['data']['precleaned_file']
        
        if not data_file.exists():
            print(f"✗ 数据文件不存在: {data_file}")
            print("  请确保数据文件路径正确")
            return False
        
        print(f"✓ 数据文件存在: {data_file}")
        
        # 尝试读取文件头
        try:
            df = pd.read_excel(data_file, nrows=5)
            print(f"✓ 数据文件可读")
            print(f"  - 列数: {len(df.columns)}")
            print(f"  - 前5列: {list(df.columns[:5])}")
            
            # 检查关键列
            required_cols = ['LinkUID']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  警告: 缺少列 {missing_cols}")
            else:
                print(f"✓ 包含必需的列")
            
        except Exception as e:
            print(f"✗ 数据文件读取失败: {e}")
            return False
        
        print("\n✓ 数据文件测试通过！\n")
        return True
        
    except Exception as e:
        print(f"✗ 数据文件测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("*" * 60)
    print("BPR 基准测试框架 - 系统测试")
    print("*" * 60)
    print("\n")
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_imports()))
    results.append(("配置文件", test_config()))
    results.append(("模型运行", test_models()))
    results.append(("评估指标", test_metrics()))
    results.append(("数据文件", test_data_file()))
    
    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！框架已准备就绪！")
        print("=" * 60)
        print("\n下一步:")
        print("1. 检查配置文件: configs/default.yaml")
        print("2. 运行基准测试: python run_benchmark.py")
        print("3. 查看结果: outputs/ 目录")
        print("\n")
        return 0
    else:
        print("\n" + "=" * 60)
        print("⚠️  部分测试失败，请检查上述错误信息")
        print("=" * 60)
        print("\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


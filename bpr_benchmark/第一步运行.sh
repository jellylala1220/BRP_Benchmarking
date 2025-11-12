#!/bin/bash
# BPR框架 - 第一步运行脚本

echo "============================================================"
echo "BPR框架 - 第一步运行"
echo "============================================================"
echo ""

# 检查Python
echo "[1/4] 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3"
    exit 1
fi
echo "✓ Python版本: $(python3 --version)"

# 安装依赖
echo ""
echo "[2/4] 安装依赖包..."
pip3 install pandas numpy scipy scikit-learn openpyxl pyyaml --quiet
echo "✓ 依赖安装完成"

# 运行测试
echo ""
echo "[3/4] 运行快速测试..."
python3 test_single_model.py

echo ""
echo "[4/4] 完成！"
echo ""
echo "如果看到评估指标，说明运行成功！"
echo "下一步: 查看 outputs/ 目录中的结果"


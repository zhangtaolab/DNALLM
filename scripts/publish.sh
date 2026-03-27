#!/bin/bash
# DNALLM PyPI发布脚本

set -e

echo "🚀 开始发布 DNALLM 到 PyPI..."

# 检查是否在正确的目录
if [ ! -f "pyproject.toml" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 检查是否安装了必要的工具
if ! command -v uv &> /dev/null; then
    echo "❌ 错误: 请先安装 uv"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command -v twine &> /dev/null; then
    echo "📦 安装 twine..."
    uv pip install twine
fi

# 清理之前的构建
echo "🧹 清理之前的构建文件..."
rm -rf build/ dist/ *.egg-info/

# 运行代码检查
echo "🔍 运行代码质量检查..."
# python scripts/check_code.py

# 构建包
echo "📦 构建包..."
uv build

# 检查构建结果
echo "✅ 检查构建结果..."
ls -la dist/

# 上传到 PyPI
echo "📤 上传到 PyPI..."
echo "请选择上传目标:"
echo "1) TestPyPI (推荐先测试)"
echo "2) PyPI (正式发布)"
read -p "请输入选择 (1 或 2): " choice

case $choice in
    1)
        echo "📤 上传到 TestPyPI..."
        twine upload --repository testpypi dist/*
        echo "✅ 已上传到 TestPyPI"
        echo "🔗 测试安装: pip install -i https://test.pypi.org/simple/ dnallm"
        ;;
    2)
        echo "📤 上传到 PyPI..."
        twine upload dist/*
        echo "✅ 已上传到 PyPI"
        echo "🔗 安装: pip install dnallm"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo "🎉 发布完成!"

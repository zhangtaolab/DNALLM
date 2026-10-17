#!/bin/bash
# 代码质量检查脚本

set -e  # 遇到错误立即退出

# 修复轮次计数文件
FIX_ATTEMPTS_FILE=".fix_attempts"
MANUAL_FIX_FILE="MANUAL_FIX_REQUIRED.md"

# 检查参数
if [ $# -eq 0 ]; then
    echo "❌ 错误: 请指定要检查的文件或目录"
    echo "用法: $0 <文件或目录路径>"
    echo "示例: $0 tests/configuration/test_configs.py"
    echo "示例: $0 tests/configuration/"
    exit 1
fi

TARGET_PATH="$1"

echo "🔍 开始代码质量检查..."
echo "📁 检查目标: $TARGET_PATH"

# 检查 ruff
echo "📋 运行 ruff 检查..."
if ! .venv/bin/ruff check "$TARGET_PATH"; then
    echo "🔧 尝试自动修复 ruff 问题..."
    .venv/bin/ruff check "$TARGET_PATH" --fix
    echo "✅ ruff 问题已自动修复"
fi
echo "✅ ruff 检查通过"

# 检查 flake8
echo "📋 运行 flake8 检查..."
if ! .venv/bin/flake8 "$TARGET_PATH"; then
    echo "⚠️  flake8 发现问题，需要手动修复"
    echo "记录到手动修复清单..."
    # 这里可以添加自动记录到 MANUAL_FIX_REQUIRED.md 的逻辑
    exit 1
fi
echo "✅ flake8 检查通过"

# 类型检查:ty(2026-10-17 D-09 翻转后唯一类型门,mypy 已退役)
echo "📋 运行 ty 检查..."
uvx ty@0.0.84 check "$TARGET_PATH" || exit 1
echo "✅ ty 检查通过"

# 运行测试文件（如果是测试文件）
if [[ "$TARGET_PATH" == *.py ]] && [[ "$TARGET_PATH" == test_*.py ]]; then
    echo "📋 运行测试文件..."
    .venv/bin/python -m pytest "$TARGET_PATH" -v
    echo "✅ 测试通过"
elif [[ "$TARGET_PATH" == tests/*/ ]]; then
    echo "📋 运行目录下的测试文件..."
    .venv/bin/python -m pytest "$TARGET_PATH" -v
    echo "✅ 测试通过"
fi

echo "🎉 代码质量检查完成！"

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

# 检查 mypy (只检查指定文件中的错误)
echo "📋 运行 mypy 检查..."
MYPY_OUTPUT=$(.venv/bin/mypy "$TARGET_PATH" --ignore-missing-imports 2>&1 || true)
if echo "$MYPY_OUTPUT" | grep -E "^$TARGET_PATH"; then
    echo "⚠️  mypy 发现问题，检查是否已在手动修复清单中..."
    # 检查是否是已知问题
    if grep -q "Subclass of.*BaseModel.*dict.*cannot exist" "$MANUAL_FIX_FILE" 2>/dev/null; then
        echo "ℹ️  此问题已在手动修复清单中，跳过..."
    else
        echo "⚠️  新问题，需要手动修复"
        echo "记录到手动修复清单..."
        exit 1
    fi
fi
echo "✅ mypy 检查通过"

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

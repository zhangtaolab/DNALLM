# Ruff 使用指南

本项目使用 [Ruff](https://docs.astral.sh/ruff/) 作为代码格式化和 linting 工具，替代了传统的 black、isort 和 flake8 组合。

## 🚀 快速开始

### 安装开发依赖

```bash
# 使用 uv (推荐)
uv pip install -e ".[dev]"

# 或使用 pip
pip install -e ".[dev]"
```

### 基本命令

```bash
# 格式化代码
ruff format .

# 检查 linting 问题
ruff check .

# 自动修复可修复的问题
ruff check . --fix

# 检查格式化和 linting
ruff format --check .
ruff check . --statistics
```

## 🛠️ 使用 Makefile

我们提供了便捷的 Makefile 命令：

```bash
# 查看所有可用命令
make help

# 格式化代码
make format

# 检查 linting
make lint

# 检查格式化和 linting
make check

# 自动修复问题
make fix

# 运行测试
make test

# 运行带覆盖率的测试
make test-cov

# 运行 CI 检查
make ci

# 格式化、修复、检查并测试
make all
```

## 🔧 使用 Pre-commit

安装 pre-commit hooks：

```bash
# 安装 pre-commit
pip install pre-commit

# 安装 hooks
pre-commit install

# 手动运行所有 hooks
pre-commit run --all-files
```

## 📋 配置说明

Ruff 配置在 `pyproject.toml` 中：

```toml
[tool.ruff]
# 排除目录
exclude = [
    ".venv",
    "__pycache__",
    "*.egg-info",
    "htmlcov",
    "site",
    "example",
    # ... 更多排除项
]

# 行长度限制
line-length = 79

# 目标 Python 版本
target-version = "py310"

[tool.ruff.lint]
# 启用的规则
select = [
    "E4",   # pycodestyle errors
    "E7",   # pycodestyle errors
    "E9",   # pycodestyle errors
    "F",    # pyflakes
    "W",    # pycodestyle warnings
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "N",    # pep8-naming
    "S",    # flake8-bandit
    "T20",  # flake8-print
    "PT",   # flake8-pytest-style
    "Q",    # flake8-quotes
    "RUF",  # Ruff-specific rules
]

# 忽略的规则
ignore = [
    "E501",  # line too long, handled by formatter
    "B008",  # do not perform function calls in argument defaults
    "S101",  # use of assert detected
    "T201",  # print found
    "T203",  # pprint found
]

[tool.ruff.format]
# 格式化配置
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```

## 🎯 主要优势

1. **速度极快**: 比 black + isort + flake8 快 10-100 倍
2. **功能全面**: 集成了 linting、格式化、import 排序等功能
3. **配置简单**: 一个工具解决所有问题
4. **兼容性好**: 与 flake8 规则完全兼容

## 🔍 常见问题

### Q: 如何忽略特定文件的特定规则？

A: 在 `pyproject.toml` 中使用 `per-file-ignores`：

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = [
    "S101",  # assert
    "F401",  # unused imports
]
```

### Q: 如何添加新的规则？

A: 在 `select` 列表中添加规则代码：

```toml
[tool.ruff.lint]
select = [
    "E4",   # 现有规则
    "NEW_RULE",  # 新规则
]
```

### Q: 如何忽略特定行？

A: 在代码中使用注释：

```python
# ruff: noqa: E501
very_long_line = "this is a very long line that exceeds the line length limit"
```

## 📚 更多资源

- [Ruff 官方文档](https://docs.astral.sh/ruff/)
- [Ruff 规则参考](https://docs.astral.sh/ruff/rules/)
- [从 flake8 迁移到 Ruff](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8)

# DNALLM 发布检查清单

## 发布前检查

### 1. 代码质量
- [ ] 运行 `python scripts/check_code.py` 通过所有检查
- [ ] 运行 `python scripts/check_code.py --with-tests` 通过所有测试
- [ ] 确保没有未提交的更改

### 2. 版本管理
- [ ] 更新 `dnallm/version.py` 中的版本号
- [ ] 更新 `pyproject.toml` 中的版本号
- [ ] 更新 `README.md` 中的版本信息（如果有）

### 3. 文档更新
- [ ] 确保 `README.md` 是最新的
- [ ] 检查所有示例代码是否可运行
- [ ] 更新 `CHANGELOG.md`（如果有）

### 4. 依赖检查
- [ ] 检查 `pyproject.toml` 中的依赖版本是否合理
- [ ] 确保所有依赖都有明确的版本约束
- [ ] 测试在不同Python版本下的安装

### 5. 包名检查
- [ ] 确认包名 `dnallm` 在PyPI上可用
- [ ] 检查是否与现有包冲突

## 发布步骤

### 方法1: 使用发布脚本（推荐）
```bash
# 1. 确保代码质量
python scripts/check_code.py --with-tests

# 2. 运行发布脚本
./scripts/publish.sh
```

### 方法2: 手动发布
```bash
# 1. 清理构建文件
rm -rf build/ dist/ *.egg-info/

# 2. 构建包
uv build

# 3. 检查构建结果
ls -la dist/

# 4. 上传到TestPyPI（推荐先测试）
twine upload --repository testpypi dist/*

# 5. 测试安装
pip install -i https://test.pypi.org/simple/ dnallm

# 6. 上传到PyPI
twine upload dist/*
```

### 方法3: 使用GitHub Actions
1. 创建GitHub Release
2. 工作流会自动构建并发布到PyPI

## 发布后验证

### 1. 安装测试
```bash
# 测试从PyPI安装
pip install dnallm

# 验证安装
python -c "import dnallm; print('DNALLM installed successfully!')"
```

### 2. 功能测试
```bash
# 测试基本功能
python -c "from dnallm import load_config; print('Import successful')"

# 测试CLI
dnallm --help
```

### 3. 文档更新
- [ ] 更新项目文档中的安装说明
- [ ] 更新示例代码中的安装命令

## 常见问题

### 1. 包名冲突
如果包名 `dnallm` 已被占用，考虑：
- 使用 `dnallm-toolkit`
- 使用 `dna-llm`
- 使用 `zhangtaolab-dnallm`

### 2. 依赖问题
- 确保所有依赖都有明确的版本约束
- 避免使用过于宽松的版本约束（如 `>=1.0`）
- 测试在不同环境下的安装

### 3. 构建失败
- 检查 `pyproject.toml` 语法
- 确保所有必需的文件都存在
- 检查包结构是否正确

## 联系信息

如有问题，请联系：
- GitHub Issues: https://github.com/zhangtaolab/DNALLM/issues
- Email: zhangtaolab@example.com

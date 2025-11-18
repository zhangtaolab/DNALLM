# DNALLM 双语文档使用说明

## 概述

DNALLM 项目现已支持中英双语文档，使用 `mkdocs-static-i18n` 插件实现。用户可以在文档网站上方便地切换语言。

## 已完成的工作

### 第一阶段：基础设施搭建 ✅

1. **添加插件依赖**
   - 在 `pyproject.toml` 中添加了 `mkdocs-static-i18n>=1.3.2` 依赖

2. **配置 MkDocs**
   - 更新了 `mkdocs.yml`，添加了完整的 i18n 插件配置
   - 配置了中文和英文两种语言
   - 添加了 115 个导航元素的中文翻译映射

3. **创建术语表**
   - 创建了 `docs/GLOSSARY.md`，包含200+专业术语的中英对照
   - 涵盖核心概念、模型架构、生物学术语、任务类型等多个类别

### 第二阶段：P0 文档翻译 ✅

已完成以下核心文档的中文翻译：

1. **首页优化与翻译**
   - `docs/index.md` - 优化为更详细和专业的版本
   - `docs/index.zh.md` - 完整的中文翻译版本
   - 新增内容：为什么选择 DNALLM、应用场景、支持的模型、社区与支持等章节

2. **快速入门文档翻译**
   - `docs/getting_started/quick_start.md`
   - `docs/getting_started/quick_start.zh.md`
   - 包含320行详细的入门指导

3. **安装指南翻译**
   - `docs/getting_started/installation.md`
   - `docs/getting_started/installation.zh.md`
   - 包含216行详细的安装说明

## 文档结构

双语文档采用后缀命名方式组织：

```
docs/
├── index.md              # 英文首页
├── index.zh.md           # 中文首页
├── GLOSSARY.md           # 术语表（双语）
├── getting_started/
│   ├── quick_start.md    # 英文
│   ├── quick_start.zh.md # 中文
│   ├── installation.md   # 英文
│   └── installation.zh.md# 中文
└── ...
```

## 如何使用

### 本地预览

1. **安装依赖**
   ```bash
   pip install -e '.[docs]'
   # 或者
   uv pip install -e '.[docs]'
   ```

2. **启动本地服务器**
   ```bash
   mkdocs serve
   ```

3. **访问文档**
   - 打开浏览器访问 `http://127.0.0.1:8000`
   - 在页面右上角可以看到语言切换器
   - 点击切换器可在中英文之间切换

### 构建静态文件

```bash
mkdocs build
```

构建后的文件结构：
```
site/
├── en/           # 英文版本
├── zh/           # 中文版本
└── index.html    # 重定向到默认语言（英文）
```

### 部署到 GitHub Pages

```bash
mkdocs gh-deploy --force
```

## 添加新的双语文档

### 步骤

1. **创建英文版本**
   ```bash
   # 在相应目录下创建 .md 文件
   touch docs/new_section/new_page.md
   ```

2. **创建中文版本**
   ```bash
   # 创建对应的 .zh.md 文件
   touch docs/new_section/new_page.zh.md
   ```

3. **更新导航**
   在 `mkdocs.yml` 的 `nav` 部分添加：
   ```yaml
   nav:
     - New Section:
       - New Page: new_section/new_page.md
   ```

4. **添加中文导航翻译**
   在 `mkdocs.yml` 的 `i18n.languages[zh].nav_translations` 中添加：
   ```yaml
   nav_translations:
     New Section: 新章节
     New Page: 新页面
   ```

### 注意事项

- 确保两个版本的文件结构和标题层级对应
- 代码块、函数名、变量名保持英文
- 专业术语参考 `GLOSSARY.md` 中的翻译
- 使用 `mkdocs build --strict` 检查链接有效性

## 翻译进度

### P0 优先级文档（已完成 ✅）
- [x] 首页 `index.md`
- [x] 快速入门 `getting_started/quick_start.md`
- [x] 安装指南 `getting_started/installation.md`

### P1 优先级文档（待完成）
- [ ] 用户指南核心页面
  - [ ] `user_guide/getting_started.md`
  - [ ] `user_guide/common_workflows.md`
  - [ ] `user_guide/best_practices.md`
- [ ] 数据处理相关
- [ ] 微调相关
- [ ] 推理相关
- [ ] MCP 相关
- [ ] 常见问题详细页面

### P2 优先级文档（待完成）
- [ ] 概念文档
- [ ] 案例研究
- [ ] API 参考文档（可选择性翻译）

## 质量保证

### 自动化检查

运行以下命令进行检查：

```bash
# 检查构建是否成功
mkdocs build --strict

# 检查双语文档配对（需要自定义脚本）
# python scripts/check_i18n_pairs.py
```

### 翻译质量标准

- **准确性**：专业术语翻译准确，参考 GLOSSARY.md
- **一致性**：同一术语在所有文档中使用相同翻译
- **可读性**：语句通顺自然，符合中文表达习惯
- **完整性**：所有图表说明、警告提示均需翻译
- **格式保持**：Markdown 格式、链接、代码块与原文一致

## 维护指南

### 更新现有文档

1. **更新英文版本**
   ```bash
   # 编辑英文文件
   vi docs/section/page.md
   ```

2. **同步更新中文版本**
   ```bash
   # 编辑中文文件
   vi docs/section/page.zh.md
   ```

3. **提交说明**
   ```bash
   git commit -m "docs: update section/page with latest changes (en + zh)"
   ```

### 翻译延迟处理

如果中文翻译暂时无法完成，可在中文版本顶部添加：

```markdown
!!! warning "翻译更新中"
    此页面的中文翻译正在更新中，部分内容可能与英文版本不同步。
    请以[英文版本](../en/path/to/page/)为准。
```

## 参考资源

- [MkDocs Static i18n 官方文档](https://ultrabug.github.io/mkdocs-static-i18n/)
- [Material for MkDocs 国际化指南](https://squidfunk.github.io/mkdocs-material/setup/changing-the-language/)
- [项目术语表](docs/GLOSSARY.md)

## 联系方式

如有文档翻译相关问题，请在 GitHub Issues 中提出或在 Discussions 中讨论。

---

**最后更新**: 2025-01-18

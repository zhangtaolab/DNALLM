# datasets
该模块主要用于读取在线或者本地的数据集、生成数据集或处理已读取的数据集。模块定义了一个 `DNADataset` 的类。  
常用功能主要包含以下几种：

- 读取数据  
  在线数据集可以通过从 `HuggingFace` 或 `Modelscope` 两个数据库进行下载和读取。本地数据集支持读取多种类型保存的序列数据（.csv, .tsv, .json, .arrow, .parquet, .txt, dict, fasta等）。  

- 数据检查（支持序列长度过滤、GC含量过滤、包含的碱基检查，不符合要求的序列将被过滤）

- 数据清洗（过滤掉序列或标签信息缺失的数据）

- 数据增强
  - 随机反向互补
  - 添加反向互补序列（原有数据集扩大1倍）
  - 添加随机数据（可调整序列长度、序列数、GC含量分布、是否包含碱基N，序列是否需要为指定倍数的长度）

- 数据打乱

- 数据拆分

- 序列分词（tokenization）


## 示例：
```python
from dnallm import DNADataset
from transformers import AutoTokenizer
```

```python
# 读取tokenizer
tokenizer = AutoTokenizer.from_pretrained("zhangtaolab/plant-dnabert-BPE")

# 读取数据
# 1. 读取本地数据（需要制定序列和标签列的头名），指定tokenizer
# 1.1 单个文件
dna_ds = DNADataset.load_local_data(
    "/path_to_your_datasets/data.csv", seq_col="sequence", label_col="labels",
    tokenizer=tokenizer, max_length=512
)
# 1.2 多个文件（如数据集已拆分）
dna_ds = DNADataset.load_local_data(
    {"train": "train.csv", "test": "test.csv", "validation": "validation.csv"},
    seq_col="sequence", label_col="labels",
    tokenizer=tokenizer, max_length=512
)

# 2. 读取在线数据
# 2.1 HuggingFace
dna_ds = DNADataset.from_huggingface("zhangtaolab/plant-multi-species-open-chromatin", seq_field="sequence", label_field="label", tokenizer=tokenizer, max_length=512)
# 2.2 ModelScope
dna_ds = DNADataset.from_modelscope("zhangtaolab/plant-multi-species-open-chromatin", seq_field="sequence", label_field="label", tokenizer=tokenizer, max_length=512)
```

```python
# 常用功能介绍
# 1. 数据检查（支持序列长度过滤、GC含量过滤、包含的碱基检查，不符合要求的序列将被过滤）
dna_ds.validate_sequences(minl=200, maxl=1000, valid_chars="ACGT")

# 2. 数据清洗（过滤掉缺乏序列或标签信息的数据）
dna_ds.process_missing_data()

# 3. 拆分数据
dna_ds.split_data(test_size=0.2, val_size=0.1)

# 4. 数据打乱
dna_ds.shuffle(seed=42)

# 5. 数据增强
# 5.1 随机反向互补
dna_ds.raw_reverse_complement(ratio=0.5) # 50%的序列进行反向互补

# 5.2 添加反向互补序列（原有数据集扩大1倍）
dna_ds.augment_reverse_complement()

# 5.3 加入随机数据（可调整序列长度、序列数、GC含量分布、是否包含碱基N，序列是否需要为指定倍数的长度）
dna_ds.random_generate(minl=200, maxl=2000, samples=3000, gc=(0.1, 0.9), N_ratio=0.0, padding_size=1, append=True, label_func=None)
## `label_func`是用来自定义数据标签的函数
## 如果不加`append=True`的参数，即使用生成的数据集覆盖原始数据集（可用于随机初始化数据集）

# 6. 数据降采样
new_ds = dna_ds.sampling(ratio=0.1)

# 7. 数据展示
dna_ds.show(head=20)          # 显示格式化后的前N个数据
tmp_ds = dna_ds.head(head=5)  # 提取前N个数据

# 8. 序列tokenization（要求提前定义好DNADataset.tokenizer）
dna_ds.encode_sequences()
```

## 函数及参数说明

请参考 [API](../api/datasets/data.md) 部分说明


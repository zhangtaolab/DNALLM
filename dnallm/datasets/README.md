### 测试Datasets模块功能

```python
from dnallm.datasets.data import DNADataset
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
dna_ds = DNADataset(Dataset, tokenizer=tokenizer, max_length=512)
# 2.1 HuggingFace
dna_ds.from_huggingface("zhangtaolab/plant-multi-species-open-chromatin", seq_field="sequence", label_field="label")

# 2.2 ModelScope
dna_ds.from_modelscope("zhangtaolab/plant-multi-species-open-chromatin", seq_field="sequence", label_field="label")
```
```python
# 常用功能介绍
# 1. 数据检查（支持序列长度过滤、GC含量过滤、包含的碱基检查，不符合要求的序列将被过滤）
dna_ds.validate_sequences(minl=200, maxl=1000, valid_chars="ACGT")

# 2. 数据清洗（过滤掉缺乏序列或标签信息的数据）
dna_ds.process_missing_data()

# 3. 数据增强
# 3.1 随机反向互补
dna_ds.raw_reverse_complement(ratio=0.5) # 50%的序列进行反向互补

# 3.2 添加反向互补序列（原有数据集扩大1倍）
dna_ds.augment_reverse_complement()

# 3.3 加入随机数据（可调整序列长度、序列数、GC含量分布、是否包含碱基N，序列是否需要为指定倍数的长度）
dna_ds.random_generate(minl=200, maxl=2000, samples=3000, gc=(0.1, 0.9), N_ratio=0.0, padding_size=1, append=True, label_func=None)
## `label_func`是用来自定义数据标签的函数
## 如果不加`append=True`的参数，即使用生成的数据集覆盖原始数据集（可用于随机初始化数据集）

## 4. 数据打乱
dna_ds.shuffle(seed=42)

## 5. 拆分数据
dna_ds.split_data(test_size=0.2, val_size=0.1)

## 6. 序列tokenization（要求提前定义好DNADataset.tokenizer）
dna_ds.encode_sequences()
```
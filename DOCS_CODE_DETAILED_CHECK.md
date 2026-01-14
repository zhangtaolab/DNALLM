# DNALLM Documentation Code Check Detailed Report

> Check Date: 2025-12-28  
> Check Scope: All Markdown Files in docs Directory  
> Status: In Progress

---

## Check List

### 1. quick_start.md Check

#### 1.1 Python Code Example Checks

**Code Block 1: Basic Model Loading and Inference**
```python
# File Location: Lines 118-140
from dnallm import load_config, load_model_and_tokenizer
from dnallm.inference import DNAInference

# Load configuration
configs = load_config("./example/notebooks/inference/inference_config.yaml")

# Load model and tokenizer
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
model, tokenizer = load_model_and_tokenizer(
    model_name, 
    task_config=configs["task"], 
    source="huggingface"
)
```

**Issue Identification**:
- ❓ Need to verify: Does `zhangtaolab/plant-dnagpt-BPE-promoter` model exist on Hugging Face?
- ❓ Need to verify: Is configuration file path correct?
- ✅ Import statements should be correct

**Check Commands**:
```bash
# 1. Verify model exists
huggingface-cli model-info zhangtaolab/plant-dnagpt-BPE-promoter 2>/dev/null || echo "Model may not exist or requires authentication"

# 2. Verify configuration file exists
ls -la example/notebooks/inference/inference_config.yaml

# 3. Verify imports
python -c "from dnallm import load_config, load_model_and_tokenizer; from dnallm.inference import DNAInference; print('✅ Import successful')"
```

**Code Block 2: Mutagenesis Analysis**
```python
# File Location: Lines 144-171
from dnallm import load_config
from dnallm.inference import Mutagenesis

# Load configuration
configs = load_config("./example/notebooks/in_silico_mutagenesis/inference_config.yaml")

# Load model and tokenizer
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs["task"],
    source="huggingface"
)
```

**Issue Identification**:
- ❓ Need to verify: Does `zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast` model exist?
- ❓ Need to verify: Are mutagenesis analysis parameters `replace_mut=True` correct?

**Code Block 3: Model Fine-tuning**
```python
# File Location: Lines 173-211
from dnallm import load_config
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

# Load configuration
configs = load_config("./example/notebooks/finetune_binary/finetune_config.yaml")

# Load model and tokenizer
model_name = "zhangtaolab/plant-dnabert-BPE"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs["task"],
    source="huggingface"
)

# Prepare dataset
dataset = DNADataset.load_local_data(
    file_paths="./tests/test_data/binary_classification/train.csv",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
)

# Encode the sequences in the dataset
dataset.encode_sequences()

# Initialize trainer
trainer = DNATrainer(
    config=configs,
    model=model,
    datasets=dataset  # Note: should be datasets not dataset
)

# Start training
trainer.train()
```

**Potential Issues**:
- ⚠️ Line 206: `datasets=dataset` parameter name may be inconsistent with variable name
  - Variable name in documentation is `dataset` (singular)
  - But `DNATrainer` expects parameter name `datasets` (plural)
  - Need to verify in code

#### 1.2 Bash Code Example Checks

**Code Block 4: Installation Commands**
```bash
# File Location: Lines 24-48
# Clone repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/MacOS
# or
.venv\Scripts\activate     # Windows

# Upgrade pip (recommended)
pip install --upgrade pip

# Install uv in virtual environment
pip install uv

# Install DNALLM with base dependencies
uv pip install -e '.[base]'

# Verify installation
python -c "import dnallm; print('DNALLM installed successfully!')"
```

**Check Result**:
- ✅ All command syntax is correct
- ✅ Installation method is consistent with pyproject.toml

**Code Block 5: GPU Support**
```bash
# File Location: Lines 77-91
# CUDA 12.4 (recommended for recent GPUs)
uv pip install -e '.[cuda124]'

# Other supported versions: cpu, cuda121, cuda126, cuda128
uv pip install -e '.[cuda121]'
```

**Check Result**:
- ✅ Consistent with dependency groups in pyproject.toml
- ✅ cuda124 is the recommended version

#### 1.3 Jupyter Notebooks List Check

**Code Block 6: Available Notebooks List**
```bash
# File Location: Lines 267-274
# Available notebooks:
# - example/notebooks/finetune_plant_dnabert/ - Classification fine-tuning
# - example/notebooks/finetune_multi_labels/ - Multi-label classification
# - example/notebooks/finetune_NER_task/ - Named Entity Recognition
# - example/notebooks/inference_and_benchmark/ - Model evaluation
# - example/notebooks/in_silico_mutagenesis/ - Mutation analysis
# - example/notebooks/embedding_attention.ipynb - Embedding and attention analysis
```

**Path Verification**:

| Path in Documentation | Actual Path | Status |
|----------------------|-------------|--------|
| `example/notebooks/finetune_plant_dnabert/` | ❌ Does not exist | ❌ Needs fix |
| `example/notebooks/finetune_multi_labels/` | ✅ Exists | ✅ Correct |
| `example/notebooks/finetune_NER_task/` | ✅ Exists | ✅ Correct |
| `example/notebooks/inference_and_benchmark/` | ❌ Does not exist (should be `inference/`) | ❌ Needs fix |
| `example/notebooks/in_silico_mutagenesis/` | ✅ Exists | ✅ Correct |
| `example/notebooks/embedding_attention.ipynb` | ✅ Exists | ✅ Correct |

**Issues Found**:
- ❌ `finetune_plant_dnabert/` - should be changed to `finetune_binary/`
- ❌ `inference_and_benchmark/` - should be changed to `inference/`

---

### 2. best_practices.md Check

#### 2.1 Data Preparation Code Check

**Code Block 7: Parquet Format**
```python
# File Location: Lines 15-22
# Save your processed DataFrame to Parquet for faster loading next time
my_dataframe.to_parquet("processed_data.parquet")

# Load it quickly later
from dnallm.datahandling import DNADataset
dna_ds = DNADataset.load_local_data("processed_data.parquet")
```

**Issue Identification**:
- ⚠️ Does `DNADataset.load_local_data()` support Parquet format?
- Need to verify if parameter `file_paths` accepts single file path

**Code Block 8: Data Augmentation**
```python
# File Location: Line 26
# Use `dna_ds.augment_reverse_complement()` to double your dataset size.
```

**Issue Identification**:
- ⚠️ Should method name be `augment_reverse_complement` or `augment()`?
- Need to verify in code

---

### 3. common_workflows.md Check

#### 3.1 Workflow Example Check

**Code Block 9: Fine-tuning Workflow**
```python
# File Location: Lines 21-55
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

# 1. Load configuration from a file
configs = load_config("./example/notebooks/finetune_binary/finetune_config.yaml")

# 2. Load model and tokenizer
model_name = "zhangtaolab/plant-dnabert-BPE"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs["task"],
    source="huggingface"
)

# 3. Prepare dataset
dataset = DNADataset.load_local_data(
    file_paths="./tests/test_data/binary_classification/train.csv",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
)
dataset.encode_sequences() # Tokenize the sequences

# 4. Initialize the trainer
trainer = DNATrainer(
    config=configs,
    model=model,
    datasets=dataset  # parameter name is datasets, passing dataset
)

# 5. Start the fine-tuning process
trainer.train()
```

**Verification Checklist**:
- ✅ `load_config` import is correct
- ✅ `load_model_and_tokenizer` parameters are correct
- ✅ `DNADataset.load_local_data` parameters are correct
- ⚠️ `DNATrainer` parameter name: `datasets` vs `dataset` needs verification

---

### 4. Key Issues to Verify

#### 4.1 API Call Parameter Verification

| Issue | Location | Verification Command |
|-------|----------|---------------------|
| `DNATrainer` parameter name | quick_start.md, common_workflows.md | `python -c "from dnallm.finetune import DNATrainer; import inspect; sig = inspect.signature(DNATrainer.__init__); print(sig)"` |
| `DNADataset` method name | best_practices.md | `python -c "from dnallm.datahandling import DNADataset; print([m for m in dir(DNADataset) if 'augment' in m.lower()])"` |
| Model list | quick_start.md | `huggingface-cli list-models zhangtaolab/* --search promoter 2>/dev/null | head -20` |
| Configuration file fields | Multiple locations | `python -c "from dnallm import load_config; c = load_config('example/notebooks/finetune_binary/finetune_config.yaml'); print(list(c.keys()))"` |

#### 4.2 Path Verification

```bash
# Verify all referenced paths
for path in \
  "example/notebooks/inference/inference_config.yaml" \
  "example/notebooks/finetune_binary/finetune_config.yaml" \
  "example/notebooks/in_silico_mutagenesis/inference_config.yaml" \
  "example/notebooks/benchmark/benchmark_config.yaml" \
  "tests/test_data/binary_classification/train.csv"; do
  if [ -f "$path" ]; then
    echo "✅ $path"
  else
    echo "❌ $path - File does not exist"
  fi
done
```

---

### 5. Known Issue List

#### 5.1 High Priority Issues

| # | Issue Description | Location | Suggested Fix |
|---|------------------|----------|---------------|
| 1 | Notebook path error: `finetune_plant_dnabert/` should be `finetune_binary/` | quick_start.md:268 | Modify path |
| 2 | Notebook path error: `inference_and_benchmark/` should be `inference/` | quick_start.md:271 | Modify path |
| 3 | `DNATrainer` parameter name inconsistency: `datasets` vs `dataset` | Multiple files | Verify and unify parameter names |

#### 5.2 Medium Priority Issues

| # | Issue Description | Location | Suggested Fix |
|---|------------------|----------|---------------|
| 4 | Model name may not exist | quick_start.md:126, 152 | Verify model is publicly available |
| 5 | Does `DNADataset.load_local_data` support Parquet | best_practices.md:20 | Verify or add explanation |
| 6 | Data augmentation method name unclear | best_practices.md:26 | Verify and provide complete method call |

#### 5.3 Low Priority Issues

| # | Issue Description | Location | Suggested Fix |
|---|------------------|----------|---------------|
| 7 | Method names mentioned in comments may be inaccurate | common_workflows.md:44 | Provide complete call example |
| 8 | Some commands missing newline explanation | quick_start.md:250 | Add newline explanation |

---

### 6. Verification Command Summary

#### 6.1 Environment Verification
```bash
# Verify DNALLM installation
python -c "import dnallm; print(f'Version: {dnallm.__version__}')"

# Verify all module imports
python -c "
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer
from dnallm.inference import DNAInference, Mutagenesis, Benchmark
print('✅ All modules imported successfully')
"
```

#### 6.2 API Signature Verification
```bash
# Verify DNATrainer initialization parameters
python -c "
from dnallm.finetune import DNATrainer
import inspect
sig = inspect.signature(DNATrainer.__init__)
print('DNATrainer parameters:', list(sig.parameters.keys()))
"

# Verify DNADataset methods
python -c "
from dnallm.datahandling import DNADataset
methods = [m for m in dir(DNADataset) if not m.startswith('_')]
print('DNADataset methods:', methods)
"
```

#### 6.3 Configuration File Verification
```bash
# Verify configuration file structure
python -c "
from dnallm import load_config
configs = load_config('example/notebooks/finetune_binary/finetune_config.yaml')
print('Finetune config keys:', list(configs.keys()))
print('Task config:', list(configs.get('task', {}).keys()))
"
```

---

### 7. Next Actions

1. **Immediate Verification** (High Priority):
   - Run above verification commands
   - Fix path errors
   - Unify parameter names

2. **Code Testing** (Medium Priority):
   - Create a test script to verify all code examples
   - Ensure each code block executes correctly

3. **Documentation Update** (Low Priority):
   - Update documentation based on verification results
   - Add missing explanations

---

**Report Generation Time**: 2025-12-28  
**Next Step**: Start executing verification commands and fixing discovered issues

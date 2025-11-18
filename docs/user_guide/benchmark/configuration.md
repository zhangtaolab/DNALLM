# Configuration Guide

This guide provides detailed information about all configuration options available for DNALLM benchmarking, including examples and best practices.

## Overview

DNALLM benchmarking configuration is defined in YAML format and supports:
- **Model Configuration**: Multiple models from different sources
- **Dataset Configuration**: Various data formats and preprocessing options
- **Evaluation Settings**: Metrics, batch sizes, and hardware options
- **Output Options**: Report formats and visualization settings

## Configuration Structure

### Basic Configuration Schema

```yaml
benchmark:
  # Basic information
  name: "string"
  description: "string"
  
  # Model definitions
  models: []
  
  # Dataset definitions
  datasets: []
  
  # Evaluation settings
  evaluation: {}
  
  # Output configuration
  output: {}
  
  # Advanced options
  advanced: {}
```

## Model Configuration

### Basic Model Definition

```yaml
models:
  - name: "Plant DNABERT"
    path: "zhangtaolab/plant-dnabert-BPE"
    source: "huggingface"
    task_type: "classification"
```

### Advanced Model Configuration

```yaml
models:
  - name: "Plant DNABERT"
    path: "zhangtaolab/plant-dnabert-BPE"
    source: "huggingface"
    task_type: "classification"
    revision: "main"  # Git branch/tag
    trust_remote_code: true
    torch_dtype: "float16"  # or "float32", "bfloat16"
    device_map: "auto"
    load_in_8bit: false
    load_in_4bit: false
    
  - name: "Custom Model"
    path: "/path/to/local/model"
    source: "local"
    task_type: "generation"
    model_class: "CustomModelClass"
    tokenizer_class: "CustomTokenizerClass"
```

### Model Source Types

| Source | Description | Example |
|--------|-------------|---------|
| `huggingface` | Hugging Face Hub | `"zhangtaolab/plant-dnabert-BPE"` |
| `modelscope` | ModelScope repository | `"zhangtaolab/plant-dnabert-BPE"` |
| `local` | Local file system | `"/path/to/model"` |
| `s3` | AWS S3 bucket | `"s3://bucket/model"` |

### Task Types

| Task Type | Description | Use Case |
|-----------|-------------|----------|
| `classification` | Binary/multi-class classification | Promoter prediction, motif detection |
| `generation` | Sequence generation | DNA synthesis, sequence design |
| `masked` | Masked language modeling | Sequence completion, mutation analysis |
| `embedding` | Feature extraction | Sequence representation, similarity |
| `regression` | Continuous value prediction | Expression level, binding affinity |

## Dataset Configuration

### Basic Dataset Definition

```yaml
datasets:
  - name: "promoter_data"
    path: "path/to/promoter_data.csv"
    task: "binary_classification"
    text_column: "sequence"
    label_column: "label"
```

### Advanced Dataset Configuration

```yaml
datasets:
  - name: "promoter_data"
    path: "path/to/promoter_data.csv"
    task: "binary_classification"
    text_column: "sequence"
    label_column: "label"
    
    # Preprocessing options
    max_length: 512
    truncation: true
    padding: "max_length"
    
    # Data splitting
    test_size: 0.2
    val_size: 0.1
    random_state: 42
    
    # Data filtering
    min_length: 100
    max_length: 1000
    valid_chars: "ACGT"
    
    # Data augmentation
    augment: true
    reverse_complement_ratio: 0.5
    random_mutation_ratio: 0.1
    
    # Custom preprocessing
    preprocessors:
      - "remove_n_bases"
      - "normalize_case"
      - "add_padding"
```

### Dataset Formats

#### CSV/TSV Format
```yaml
datasets:
  - name: "csv_dataset"
    path: "data.csv"
    format: "csv"
    separator: ","  # or "\t" for TSV
    encoding: "utf-8"
    text_column: "sequence"
    label_column: "label"
    additional_columns: ["metadata", "source"]
```

#### JSON Format
```yaml
datasets:
  - name: "json_dataset"
    path: "data.json"
    format: "json"
    text_key: "sequence"
    label_key: "label"
    nested_path: "data.items"  # For nested JSON structures
```

#### FASTA Format
```yaml
datasets:
  - name: "fasta_dataset"
    path: "sequences.fasta"
    format: "fasta"
    label_parser: "header"  # Extract label from header
    header_format: "sequence_id|label:value"  # Custom header format
```

#### Arrow/Parquet Format
```yaml
datasets:
  - name: "arrow_dataset"
    path: "data.arrow"
    format: "arrow"
    text_column: "sequence"
    label_column: "label"
```

### Data Preprocessing Options

```yaml
datasets:
  - name: "processed_data"
    path: "raw_data.csv"
    
    # Sequence processing
    preprocessing:
      remove_n_bases: true
      normalize_case: true
      add_padding: true
      padding_size: 512
      
    # Quality filtering
    filtering:
      min_length: 200
      max_length: 1000
      min_gc_content: 0.2
      max_gc_content: 0.8
      valid_chars: "ACGT"
      
    # Data augmentation
    augmentation:
      reverse_complement: true
      random_mutations: true
      mutation_rate: 0.01
      synthetic_samples: 1000
```

## Evaluation Configuration

### Basic Evaluation Settings

```yaml
evaluation:
  batch_size: 32
  max_length: 512
  device: "cuda"
  num_workers: 4
```

### Advanced Evaluation Options

```yaml
evaluation:
  # Batch processing
  batch_size: 32
  gradient_accumulation_steps: 1
  
  # Sequence processing
  max_length: 512
  truncation: true
  padding: "max_length"
  
  # Hardware settings
  device: "cuda"  # or "cpu", "auto"
  num_workers: 4
  pin_memory: true
  
  # Performance optimization
  use_fp16: true
  use_bf16: false
  mixed_precision: true
  
  # Memory management
  max_memory: "16GB"
  memory_efficient_attention: true
  
  # Reproducibility
  seed: 42
  deterministic: true
  
  # Evaluation strategy
  eval_strategy: "steps"  # or "epoch"
  eval_steps: 100
  eval_accumulation_steps: 1
```

### Device Configuration

```yaml
evaluation:
  # Single GPU
  device: "cuda:0"
  
  # Multiple GPUs
  device: "cuda"
  parallel_strategy: "data_parallel"
  
  # CPU only
  device: "cpu"
  num_threads: 8
  
  # Auto device selection
  device: "auto"
  device_map: "auto"
  
  # Mixed precision
  use_fp16: true
  use_bf16: false
  mixed_precision: true
```

## Metrics Configuration

### Basic Metrics

```yaml
metrics:
  - "accuracy"
  - "f1_score"
  - "precision"
  - "recall"
  - "roc_auc"
  - "mse"
  - "mae"
```

### Advanced Metrics

```yaml
metrics:
  # Classification metrics
  - "accuracy"
  - "f1_score"
  - "precision"
  - "recall"
  - "roc_auc"
  - "pr_auc"
  - "matthews_correlation"
  
  # Regression metrics
  - "mse"
  - "mae"
  - "rmse"
  - "r2_score"
  - "pearson_correlation"
  - "spearman_correlation"
  
  # Custom metrics
  - name: "gc_content_accuracy"
    class: "GCContentMetric"
    parameters:
      threshold: 0.1
      
  - name: "conservation_score"
    class: "ConservationMetric"
    parameters:
      window_size: 10
      similarity_threshold: 0.8
```

### Custom Metric Configuration

```yaml
metrics:
  - name: "custom_dna_metric"
    class: "CustomDNAMetric"
    parameters:
      gc_weight: 0.3
      conservation_weight: 0.4
      motif_weight: 0.3
      threshold: 0.5
    file_path: "path/to/custom_metric.py"
    class_name: "CustomDNAMetric"
```

## Output Configuration

### Basic Output Settings

```yaml
output:
  format: "html"
  path: "benchmark_results"
  save_predictions: true
  generate_plots: true
```

### Advanced Output Options

```yaml
output:
  # Output formats
  formats: ["html", "csv", "json", "pdf"]
  
  # File paths
  path: "benchmark_results"
  predictions_file: "predictions.csv"
  metrics_file: "metrics.json"
  plots_dir: "plots"
  
  # Content options
  save_predictions: true
  save_embeddings: false
  save_attention_maps: false
  save_token_probabilities: false
  
  # Visualization
  generate_plots: true
  plot_types: ["bar", "line", "heatmap", "scatter"]
  plot_style: "seaborn"
  plot_colors: ["#1f77b4", "#ff7f0e", "#2ca02c"]
  
  # Report customization
  report_title: "DNA Model Benchmark Report"
  report_description: "Comprehensive comparison of DNA language models"
  include_summary: true
  include_details: true
  include_recommendations: true
  
  # Export options
  export_predictions: true
  export_metrics: true
  export_config: true
  export_logs: true
```

### Report Customization

```yaml
output:
  report:
    title: "DNA Model Benchmark Report"
    subtitle: "Performance Comparison on Promoter Prediction"
    author: "Your Name"
    date: "auto"
    
    # Sections to include
    sections:
      - "executive_summary"
      - "model_overview"
      - "dataset_description"
      - "results_summary"
      - "detailed_results"
      - "performance_analysis"
      - "recommendations"
      - "appendix"
    
    # Custom styling
    styling:
      theme: "modern"
      color_scheme: "blue"
      font_family: "Arial"
      font_size: 12
      
    # Interactive elements
    interactive:
      enable_zoom: true
      enable_hover: true
      enable_selection: true
```

## Advanced Configuration

### Cross-Validation Settings

```yaml
advanced:
  cross_validation:
    enabled: true
    method: "k_fold"  # or "stratified_k_fold", "time_series_split"
    n_splits: 5
    shuffle: true
    random_state: 42
    
    # Stratified options
    stratification:
      enabled: true
      column: "label"
      bins: 10
      
    # Time series options
    time_series:
      column: "date"
      test_size: 0.2
      gap: 0
```

### Performance Profiling

```yaml
advanced:
  performance_profiling:
    enabled: true
    
    # Memory profiling
    memory:
      track_gpu: true
      track_cpu: true
      track_peak: true
      profile_allocations: true
      
    # Time profiling
    timing:
      track_inference: true
      track_preprocessing: true
      track_postprocessing: true
      warmup_runs: 10
      
    # Resource monitoring
    resources:
      track_cpu_usage: true
      track_gpu_usage: true
      track_io: true
      sampling_interval: 0.1
```

### Custom Evaluation Pipeline

```yaml
advanced:
  custom_pipeline:
    enabled: true
    pipeline_file: "path/to/custom_pipeline.py"
    
    # Pipeline steps
    steps:
      - name: "data_preprocessing"
        function: "custom_preprocess"
        parameters:
          normalize: true
          augment: false
          
      - name: "model_evaluation"
        function: "custom_evaluate"
        parameters:
          metric: "custom_metric"
          threshold: 0.5
          
      - name: "result_aggregation"
        function: "custom_aggregate"
        parameters:
          method: "weighted_average"
          weights: [0.4, 0.3, 0.3]
```

## Configuration Examples

### Complete Example: Promoter Prediction

```yaml
benchmark:
  name: "Promoter Prediction Benchmark"
  description: "Comparing DNA language models on promoter prediction tasks"
  
  models:
    - name: "Plant DNABERT"
      path: "zhangtaolab/plant-dnabert-BPE"
      source: "huggingface"
      task_type: "classification"
      
    - name: "Plant DNAGPT"
      path: "zhangtaolab/plant-dnagpt-BPE"
      source: "huggingface"
      task_type: "generation"
      
    - name: "Nucleotide Transformer"
      path: "InstaDeepAI/nucleotide-transformer-500m-human-ref"
      source: "huggingface"
      task_type: "classification"

  datasets:
    - name: "promoter_strength"
      path: "data/promoter_strength.csv"
      task: "binary_classification"
      text_column: "sequence"
      label_column: "label"
      max_length: 512
      test_size: 0.2
      val_size: 0.1
      
    - name: "open_chromatin"
      path: "data/open_chromatin.csv"
      task: "binary_classification"
      text_column: "sequence"
      label_column: "label"
      max_length: 512

  metrics:
    - "accuracy"
    - "f1_score"
    - "precision"
    - "recall"
    - "roc_auc"
    - name: "gc_content_accuracy"
      class: "GCContentMetric"

  evaluation:
    batch_size: 32
    max_length: 512
    device: "cuda"
    num_workers: 4
    use_fp16: true
    seed: 42

  output:
    format: "html"
    path: "promoter_benchmark_results"
    save_predictions: true
    generate_plots: true
    report_title: "Promoter Prediction Model Comparison"

  advanced:
    cross_validation:
      enabled: true
      method: "stratified_k_fold"
      n_splits: 5
      
    performance_profiling:
      enabled: true
      memory:
        track_gpu: true
        track_peak: true
```

### Minimal Example

```yaml
benchmark:
  name: "Quick Model Test"
  
  models:
    - name: "Test Model"
      path: "zhangtaolab/plant-dnabert-BPE"
      source: "huggingface"
      task_type: "classification"

  datasets:
    - name: "test_data"
      path: "test.csv"
      task: "binary_classification"
      text_column: "sequence"
      label_column: "label"

  metrics:
    - "accuracy"
    - "f1_score"

  evaluation:
    batch_size: 16
    device: "cuda"

  output:
    format: "csv"
    path: "quick_test_results"
```

## Configuration Validation

### Schema Validation

DNALLM automatically validates your configuration:

```python
from dnallm import validate_config

# Validate configuration
try:
    validate_config("benchmark_config.yaml")
    print("Configuration is valid!")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

### Common Validation Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Model not found` | Invalid model path | Check model exists on specified source |
| `Invalid task type` | Unsupported task | Use supported task types |
| `Missing required field` | Incomplete configuration | Add missing required fields |
| `Invalid metric name` | Unknown metric | Use supported metric names |
| `Path not found` | Invalid file path | Check file exists and is accessible |

## Best Practices

### 1. **Configuration Organization**
```yaml
# Use descriptive names
benchmark:
  name: "Comprehensive DNA Model Evaluation 2024"
  
# Group related settings
evaluation:
  # Hardware settings
  device: "cuda"
  num_workers: 4
  
  # Performance settings
  batch_size: 32
  use_fp16: true
```

### 2. **Environment-Specific Configs**
```yaml
# Development config
evaluation:
  batch_size: 8
  device: "cpu"
  
# Production config  
evaluation:
  batch_size: 64
  device: "cuda"
  use_fp16: true
```

### 3. **Version Control**
```yaml
# Include version information
benchmark:
  version: "1.0.0"
  config_version: "2024.1"
  created_by: "Your Name"
  created_date: "2024-01-15"
```

## Next Steps

After configuring your benchmark:

1. **Run Your Benchmark**: Follow the [Getting Started](getting_started.md) guide
2. **Explore Advanced Features**: Learn about [Advanced Techniques](advanced_techniques.md)
3. **See Real Examples**: Check [Examples and Use Cases](examples.md)
4. **Troubleshoot Issues**: Visit [Troubleshooting](../../faq/benchmark_troubleshooting.md)

---

**Need help with configuration?** Check our [FAQ](../../faq/index.md) or open an issue on [GitHub](https://github.com/zhangtaolab/DNALLM/issues).

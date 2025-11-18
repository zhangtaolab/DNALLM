# Advanced Benchmarking Techniques

This guide covers advanced benchmarking techniques including cross-validation, custom metrics, performance profiling, and optimization strategies.

## Overview

Advanced benchmarking techniques help you:
- Ensure robust and reliable model evaluation
- Implement custom evaluation metrics
- Profile and optimize model performance
- Handle complex benchmarking scenarios

## Cross-Validation Benchmarking

Cross-validation provides more robust performance estimates by testing models on multiple data splits.

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold
import numpy as np
from dnallm import Benchmark

def run_cross_validation_benchmark(models, datasets, k_folds=5):
    """Run k-fold cross-validation benchmark."""
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_results = {}
    
    for model_name, model_info in models.items():
        cv_results[model_name] = {}
        
        for dataset_name, dataset in datasets.items():
            fold_scores = []
            
            # Split dataset into k folds
            for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
                print(f"Running fold {fold + 1}/{k_folds} for {model_name} on {dataset_name}")
                
                # Split data for this fold
                train_data = dataset.select(train_idx)
                val_data = dataset.select(val_idx)
                
                # Evaluate on validation fold
                fold_result = benchmark.evaluate_single_model(
                    model_info["model"],
                    model_info["tokenizer"],
                    val_data,
                    metrics=["accuracy", "f1_score", "precision", "recall"]
                )
                
                fold_scores.append(fold_result)
            
            # Aggregate fold results
            cv_results[model_name][dataset_name] = {
                "mean_accuracy": np.mean([s["accuracy"] for s in fold_scores]),
                "std_accuracy": np.mean([s["accuracy"] for s in fold_scores]),
                "mean_f1": np.mean([s["f1_score"] for s in fold_scores]),
                "std_f1": np.std([s["f1_score"] for s in fold_scores]),
                "fold_results": fold_scores
            }
    
    return cv_results

# Usage
cv_results = run_cross_validation_benchmark(loaded_models, datasets, k_folds=5)

# Display results
for model_name, results in cv_results.items():
    print(f"\n{model_name} Cross-Validation Results:")
    for dataset_name, metrics in results.items():
        print(f"  {dataset_name}:")
        print(f"    Accuracy: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
        print(f"    F1 Score: {metrics['mean_f1']:.4f} ± {metrics['std_f1']:.4f}")
```

### Stratified K-Fold for Imbalanced Data

```python
from sklearn.model_selection import StratifiedKFold

def run_stratified_cv_benchmark(models, datasets, k_folds=5):
    """Run stratified k-fold cross-validation for imbalanced datasets."""
    
    stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_results = {}
    
    for model_name, model_info in models.items():
        cv_results[model_name] = {}
        
        for dataset_name, dataset in datasets.items():
            # Get labels for stratification
            labels = [item["label"] for item in dataset]
            
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(dataset, labels)):
                # ... rest of the implementation similar to above
                pass
    
    return cv_results
```

## Custom Evaluation Metrics

DNALLM allows you to implement custom evaluation metrics for specific use cases.

### Basic Custom Metric

```python
from dnallm.tasks.metrics import CustomMetric
import numpy as np

class GCContentMetric(CustomMetric):
    """Custom metric to evaluate GC content prediction accuracy."""
    
    def __init__(self):
        super().__init__()
        self.name = "gc_content_accuracy"
    
    def compute(self, predictions, targets, sequences=None):
        """Compute GC content prediction accuracy."""
        if sequences is None:
            return {"gc_content_accuracy": 0.0}
        
        gc_accuracy = []
        for pred, target, seq in zip(predictions, targets, sequences):
            # Calculate predicted GC content
            pred_gc = self._calculate_gc_content(seq, pred)
            # Calculate actual GC content
            actual_gc = self._calculate_gc_content(seq, target)
            
            # Compute accuracy
            accuracy = 1.0 - abs(pred_gc - actual_gc) / max(actual_gc, 0.01)
            gc_accuracy.append(max(0.0, accuracy))
        
        return {"gc_content_accuracy": np.mean(gc_accuracy)}
    
    def _calculate_gc_content(self, sequence, mask):
        """Calculate GC content based on sequence and mask."""
        gc_count = 0
        total_count = 0
        
        for i, char in enumerate(sequence):
            if mask[i] == 1:  # If position is masked
                if char in ['G', 'C']:
                    gc_count += 1
                total_count += 1
        
        return gc_count / max(total_count, 1)

# Usage in benchmark
benchmark = Benchmark(
    models=loaded_models,
    datasets=datasets,
    metrics=["accuracy", "f1_score", GCContentMetric()],
    batch_size=32
)
```

### Advanced Custom Metric with Multiple Outputs

```python
class ComprehensiveDNAMetric(CustomMetric):
    """Comprehensive DNA sequence evaluation metric."""
    
    def __init__(self):
        super().__init__()
        self.name = "comprehensive_dna_score"
    
    def compute(self, predictions, targets, sequences=None, **kwargs):
        """Compute comprehensive DNA evaluation score."""
        results = {}
        
        # Base accuracy
        results["base_accuracy"] = self._compute_accuracy(predictions, targets)
        
        # Sequence-specific metrics
        if sequences is not None:
            results["gc_content_score"] = self._compute_gc_content_score(predictions, targets, sequences)
            results["conservation_score"] = self._compute_conservation_score(predictions, targets, sequences)
            results["motif_score"] = self._compute_motif_score(predictions, targets, sequences)
        
        # Overall score (weighted average)
        weights = [0.4, 0.2, 0.2, 0.2]  # Adjust weights as needed
        scores = [results["base_accuracy"], results["gc_content_score"], 
                 results["conservation_score"], results["motif_score"]]
        
        results["overall_score"] = np.average(scores, weights=weights)
        
        return results
    
    def _compute_accuracy(self, predictions, targets):
        """Compute basic accuracy."""
        return np.mean(np.array(predictions) == np.array(targets))
    
    def _compute_gc_content_score(self, predictions, targets, sequences):
        """Compute GC content prediction score."""
        # Implementation details...
        return 0.85
    
    def _compute_conservation_score(self, predictions, targets, sequences):
        """Compute conservation prediction score."""
        # Implementation details...
        return 0.78
    
    def _compute_motif_score(self, predictions, targets, sequences):
        """Compute motif prediction score."""
        # Implementation details...
        return 0.92
```

## Performance Profiling

Performance profiling helps you understand model efficiency and identify bottlenecks.

### Basic Performance Profiling

```python
import time
import psutil
import torch
from memory_profiler import profile

def profile_model_performance(model, tokenizer, dataset, num_samples=100):
    """Profile model performance including time and memory usage."""
    
    # Select subset for profiling
    profile_data = dataset.select(range(min(num_samples, len(dataset))))
    
    # Warm up (important for accurate timing)
    print("Warming up model...")
    for _ in range(10):
        _ = model(torch.randn(1, 512).to(model.device))
    
    # Memory profiling
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()
        memory_reserved_before = torch.cuda.memory_reserved()
    
    # Time profiling
    print("Running performance profiling...")
    start_time = time.time()
    
    predictions = []
    batch_times = []
    
    for i, batch in enumerate(profile_data.get_dataloader(batch_size=1)):
        batch_start = time.time()
        
        with torch.no_grad():
            outputs = model(batch["input_ids"].to(model.device))
            predictions.append(outputs.logits.argmax(-1).cpu())
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        if i % 20 == 0:
            print(f"Processed {i+1}/{num_samples} samples...")
    
    total_time = time.time() - start_time
    
    # Memory after
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated()
        memory_reserved_after = torch.cuda.memory_reserved()
        memory_used = memory_after - memory_before
        memory_reserved = memory_reserved_after - memory_reserved_before
    
    # CPU profiling
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Calculate statistics
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    
    return {
        "total_inference_time": total_time,
        "avg_batch_time": avg_batch_time,
        "std_batch_time": std_batch_time,
        "samples_per_second": num_samples / total_time,
        "memory_used_mb": memory_used / 1024 / 1024 if torch.cuda.is_available() else 0,
        "memory_reserved_mb": memory_reserved / 1024 / 1024 if torch.cuda.is_available() else 0,
        "cpu_usage_percent": cpu_percent,
        "throughput": num_samples / total_time
    }

# Profile all models
performance_profiles = {}
for model_name, model_info in loaded_models.items():
    print(f"\nProfiling {model_name}...")
    performance_profiles[model_name] = profile_model_performance(
        model_info["model"],
        model_info["tokenizer"],
        datasets["promoter_strength"],
        num_samples=200
    )
```

### Advanced Memory Profiling

```python
import tracemalloc
from contextlib import contextmanager

@contextmanager
def memory_profiler():
    """Context manager for detailed memory profiling."""
    tracemalloc.start()
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
        tracemalloc.stop()

def detailed_memory_profile(model, dataset, batch_size=32):
    """Detailed memory profiling with tracemalloc."""
    
    with memory_profiler():
        # Load data
        dataloader = dataset.get_dataloader(batch_size=batch_size)
        
        # Run inference
        for batch in dataloader:
            with torch.no_grad():
                outputs = model(batch["input_ids"].to(model.device))
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

## Optimization Strategies

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

def benchmark_with_mixed_precision(model, tokenizer, dataset):
    """Benchmark model with mixed precision for improved performance."""
    
    # Enable mixed precision
    scaler = GradScaler()
    
    start_time = time.time()
    predictions = []
    
    for batch in dataset.get_dataloader(batch_size=32):
        with autocast():
            outputs = model(batch["input_ids"].to(model.device))
            predictions.append(outputs.logits.argmax(-1).cpu())
    
    mixed_precision_time = time.time() - start_time
    
    # Compare with full precision
    start_time = time.time()
    predictions_fp32 = []
    
    for batch in dataset.get_dataloader(batch_size=32):
        outputs = model(batch["input_ids"].to(model.device))
        predictions_fp32.append(outputs.logits.argmax(-1).cpu())
    
    fp32_time = time.time() - start_time
    
    return {
        "mixed_precision_time": mixed_precision_time,
        "fp32_time": fp32_time,
        "speedup": fp32_time / mixed_precision_time,
        "memory_savings": "~50% (estimated)"
    }
```

### Batch Size Optimization

```python
def find_optimal_batch_size(model, dataset, max_batch_size=128):
    """Find optimal batch size for given model and hardware."""
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > max_batch_size:
            continue
            
        try:
            # Test batch size
            start_time = time.time()
            memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Run inference
            dataloader = dataset.get_dataloader(batch_size=batch_size)
            for batch in dataloader:
                with torch.no_grad():
                    outputs = model(batch["input_ids"].to(model.device))
                break  # Just test one batch
            
            inference_time = time.time() - start_time
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            results[batch_size] = {
                "inference_time": inference_time,
                "memory_used": (memory_after - memory_before) / 1024 / 1024,
                "throughput": batch_size / inference_time
            }
            
            print(f"Batch size {batch_size}: {results[batch_size]}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size} failed: OOM")
                break
            else:
                print(f"Batch size {batch_size} failed: {e}")
    
    # Find optimal batch size
    optimal_batch_size = max(results.keys(), key=lambda x: results[x]["throughput"])
    
    return optimal_batch_size, results
```

## Advanced Benchmarking Scenarios

### Multi-Dataset Benchmarking

```python
def run_multi_dataset_benchmark(models, datasets, metrics):
    """Run benchmark across multiple datasets with aggregated results."""
    
    all_results = {}
    
    for model_name, model_info in models.items():
        all_results[model_name] = {
            "dataset_results": {},
            "aggregated_metrics": {}
        }
        
        # Run on each dataset
        for dataset_name, dataset in datasets.items():
            dataset_result = benchmark.evaluate_single_model(
                model_info["model"],
                model_info["tokenizer"],
                dataset,
                metrics=metrics
            )
            
            all_results[model_name]["dataset_results"][dataset_name] = dataset_result
        
        # Aggregate across datasets
        all_results[model_name]["aggregated_metrics"] = aggregate_metrics(
            all_results[model_name]["dataset_results"]
        )
    
    return all_results

def aggregate_metrics(dataset_results):
    """Aggregate metrics across multiple datasets."""
    aggregated = {}
    
    for metric in dataset_results[list(dataset_results.keys())[0]].keys():
        values = [dataset_results[ds][metric] for ds in dataset_results.keys()]
        aggregated[f"{metric}_mean"] = np.mean(values)
        aggregated[f"{metric}_std"] = np.std(values)
        aggregated[f"{metric}_min"] = np.min(values)
        aggregated[f"{metric}_max"] = np.max(values)
    
    return aggregated
```

### Time-Series Benchmarking

```python
def run_time_series_benchmark(model, dataset, time_column, interval_days=30):
    """Run benchmark on time-series data with temporal splits."""
    
    # Sort by time
    sorted_data = sorted(dataset, key=lambda x: x[time_column])
    
    # Create temporal splits
    total_days = (sorted_data[-1][time_column] - sorted_data[0][time_column]).days
    num_splits = total_days // interval_days
    
    temporal_results = []
    
    for i in range(num_splits):
        start_idx = i * len(sorted_data) // num_splits
        end_idx = (i + 1) * len(sorted_data) // num_splits
        
        # Test on future data
        test_data = sorted_data[end_idx:]
        if len(test_data) == 0:
            continue
        
        # Evaluate performance
        result = benchmark.evaluate_single_model(
            model, tokenizer, test_data, metrics=["accuracy", "f1_score"]
        )
        
        temporal_results.append({
            "time_period": i,
            "start_date": sorted_data[start_idx][time_column],
            "end_date": sorted_data[end_idx][time_column],
            "test_size": len(test_data),
            **result
        })
    
    return temporal_results
```

## Best Practices

### 1. **Reproducibility**
```python
# Set random seeds
import random
import numpy as np
import torch

def set_reproducibility(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Use in benchmark
set_reproducibility(42)
```

### 2. **Resource Management**
```python
def cleanup_resources():
    """Clean up GPU memory and other resources."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    import gc
    gc.collect()

# Call between model evaluations
for model_name, model_info in loaded_models.items():
    # Run benchmark
    result = benchmark.evaluate_single_model(...)
    
    # Clean up
    cleanup_resources()
```

### 3. **Progress Monitoring**
```python
from tqdm import tqdm
import logging

def setup_logging():
    """Setup logging for benchmark progress."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('benchmark.log'),
            logging.StreamHandler()
        ]
    )

# Use in benchmark
setup_logging()
logger = logging.getLogger(__name__)

for model_name in tqdm(loaded_models.keys(), desc="Benchmarking models"):
    logger.info(f"Starting benchmark for {model_name}")
    # ... benchmark code
    logger.info(f"Completed benchmark for {model_name}")
```

## Next Steps

After mastering these advanced techniques:

1. **Explore Real-world Examples**: See [Examples and Use Cases](examples.md)
2. **Learn Configuration Options**: Check [Configuration Guide](configuration.md)
3. **Troubleshoot Issues**: Visit [Troubleshooting](../../faq/benchmark_troubleshooting.md)
4. **Contribute**: Help improve DNALLM's benchmarking capabilities

---

**Ready for real-world examples?** Continue to [Examples and Use Cases](examples.md) to see these techniques in action.

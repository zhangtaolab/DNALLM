# Examples and Use Cases

This guide provides real-world examples and practical use cases for DNALLM benchmarking, demonstrating how to apply the concepts learned in previous sections.

## Overview

The examples in this guide cover:
- **Research Applications**: Academic model comparison and evaluation
- **Production Use Cases**: Model selection for deployment
- **Performance Analysis**: Optimization and profiling scenarios
- **Custom Scenarios**: Specialized benchmarking requirements

## Research Applications

### Example 1: Academic Model Comparison

**Scenario**: Comparing multiple DNA language models for publication in a research paper.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dnallm import Benchmark

# Define research models
research_models = [
    {
        "name": "Plant DNABERT",
        "path": "zhangtaolab/plant-dnabert-BPE",
        "source": "huggingface",
        "task_type": "classification"
    },
    {
        "name": "Nucleotide Transformer",
        "path": "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        "source": "huggingface",
        "task_type": "classification"
    },
    {
        "name": "DNABERT-2",
        "path": "zhangtaolab/DNABERT-2",
        "source": "huggingface",
        "task_type": "classification"
    }
]

# Load research datasets
datasets = {
    "promoter_prediction": DNADataset.load_local_data(
        "data/promoter_prediction.csv",
        seq_col="sequence",
        label_col="label",
        max_length=512
    ),
    "motif_detection": DNADataset.load_local_data(
        "data/motif_detection.csv",
        seq_col="sequence",
        label_col="label",
        max_length=512
    )
}

# Run comprehensive benchmark with cross-validation
benchmark = Benchmark(
    models=research_models,
    datasets=datasets,
    metrics=["accuracy", "f1_score", "precision", "recall", "roc_auc"],
    batch_size=32,
    device="cuda"
)

# Execute with cross-validation
results = benchmark.run_with_cross_validation(k_folds=5)

# Generate publication-ready results
publication_results = generate_publication_results(results)
create_publication_plots(publication_results)
export_research_results(publication_results, "research_paper_results")
```

### Example 2: Cross-Species Model Evaluation

**Scenario**: Evaluating how well models trained on one species perform on related species.

```python
def run_cross_species_benchmark():
    """Evaluate model performance across different species."""
    
    # Define species-specific datasets
    species_datasets = {
        "Arabidopsis_thaliana": "data/arabidopsis_promoters.csv",
        "Oryza_sativa": "data/rice_promoters.csv",
        "Zea_mays": "data/maize_promoters.csv"
    }
    
    # Load datasets
    datasets = {}
    for species, path in species_datasets.items():
        datasets[species] = DNADataset.load_local_data(
            path,
            seq_col="sequence",
            label_col="label",
            max_length=512
        )
    
    # Run cross-species evaluation
    cross_species_results = {}
    
    for model_name, model_info in research_models.items():
        cross_species_results[model_name] = {}
        
        for source_species in species_datasets.keys():
            for target_species in species_datasets.keys():
                if source_species == target_species:
                    continue
                
                # Evaluate model trained on source species on target species
                result = benchmark.evaluate_single_model(
                    model_info["model"],
                    model_info["tokenizer"],
                    datasets[target_species],
                    metrics=["accuracy", "f1_score"]
                )
                
                cross_species_results[model_name][f"{source_species}_to_{target_species}"] = result
    
    return cross_species_results

# Run cross-species benchmark
cross_species_results = run_cross_species_benchmark()
analyze_transfer_learning(cross_species_results)
```

## Production Use Cases

### Example 3: Model Selection for Production

**Scenario**: Choosing the best model for deployment in a production environment.

```python
def production_model_selection():
    """Select the best model for production deployment."""
    
    # Define production criteria
    production_criteria = {
        "accuracy_threshold": 0.85,
        "inference_time_threshold": 0.1,  # seconds per sequence
        "memory_threshold": 8,  # GB
        "cost_threshold": 100  # dollars per month
    }
    
    # Production datasets (representative of real-world usage)
    production_datasets = {
        "high_throughput": load_production_dataset("data/high_throughput.csv"),
        "edge_device": load_production_dataset("data/edge_device.csv"),
        "real_time": load_production_dataset("data/real_time.csv")
    }
    
    # Run production benchmark
    production_benchmark = Benchmark(
        models=candidate_models,
        datasets=production_datasets,
        metrics=["accuracy", "inference_time", "memory_usage", "throughput"],
        batch_size=64,  # Production batch size
        device="cuda"
    )
    
    # Execute with production settings
    production_results = production_benchmark.run()
    
    # Apply production criteria
    qualified_models = filter_by_production_criteria(
        production_results, 
        production_criteria
    )
    
    # Rank by production suitability
    ranked_models = rank_production_models(qualified_models)
    
    return ranked_models

# Run production model selection
production_rankings = production_model_selection()

# Display results
print("Production Model Rankings:")
for i, (model_name, score) in enumerate(production_rankings, 1):
    print(f"{i}. {model_name}: {score:.3f}")

# Generate production report
generate_production_report(production_rankings, "production_selection_report")
```

### Example 4: Performance Monitoring and Alerting

**Scenario**: Continuous monitoring of model performance in production with automated alerting.

```python
class ProductionModelMonitor:
    """Monitor model performance in production."""
    
    def __init__(self, model, dataset, alert_thresholds):
        self.model = model
        self.dataset = dataset
        self.alert_thresholds = alert_thresholds
        self.performance_history = []
        self.setup_logging()
    
    def monitor_performance(self, interval_minutes=60):
        """Monitor model performance continuously."""
        self.logger.info("Starting production performance monitoring...")
        
        while True:
            try:
                # Run performance evaluation
                current_performance = self.evaluate_current_performance()
                
                # Store in history
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'performance': current_performance
                })
                
                # Check for performance degradation
                if self.check_performance_degradation(current_performance):
                    self.send_alert(current_performance)
                
                # Log performance
                self.logger.info(f"Performance: {current_performance}")
                
                # Wait for next evaluation
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                self.send_error_alert(str(e))
                time.sleep(interval_minutes * 60)
    
    def evaluate_current_performance(self):
        """Evaluate current model performance."""
        benchmark = Benchmark(
            models=[self.model],
            datasets=[self.dataset],
            metrics=["accuracy", "f1_score", "inference_time"],
            batch_size=32,
            device="cuda"
        )
        
        results = benchmark.run()
        return results[list(results.keys())[0]]

# Usage example
alert_thresholds = {
    "accuracy": 0.80,
    "f1_score": 0.75,
    "inference_time": 0.2
}

monitor = ProductionModelMonitor(
    model=selected_model,
    dataset=production_dataset,
    alert_thresholds=alert_thresholds
)

# Start monitoring (in production, run this as a service)
# monitor.monitor_performance(interval_minutes=60)
```

## Performance Analysis Scenarios

### Example 5: Model Optimization Analysis

**Scenario**: Analyzing model performance to identify optimization opportunities.

```python
def analyze_model_optimization():
    """Analyze model performance for optimization opportunities."""
    
    # Test different optimization strategies
    optimization_results = {}
    
    # 1. Batch size optimization
    batch_size_results = optimize_batch_size(selected_model, test_dataset)
    optimization_results["batch_size"] = batch_size_results
    
    # 2. Mixed precision analysis
    precision_results = analyze_mixed_precision(selected_model, test_dataset)
    optimization_results["mixed_precision"] = precision_results
    
    # 3. Memory optimization
    memory_results = analyze_memory_optimization(selected_model, test_dataset)
    optimization_results["memory"] = memory_results
    
    return optimization_results

def optimize_batch_size(model, dataset):
    """Find optimal batch size for the model."""
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    results = {}
    
    for batch_size in batch_sizes:
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
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size} failed: OOM")
                break
    
    return results

# Run optimization analysis
optimization_results = analyze_model_optimization()

# Generate optimization report
generate_optimization_report(optimization_results, "model_optimization_report")
```

## Custom Benchmarking Scenarios

### Example 6: Multi-Task Model Evaluation

**Scenario**: Evaluating models that can perform multiple tasks simultaneously.

```python
def multi_task_benchmark():
    """Benchmark models on multiple tasks simultaneously."""
    
    # Define multiple tasks
    tasks = {
        "promoter_prediction": {
            "dataset": "data/promoter_data.csv",
            "task_type": "binary_classification",
            "metrics": ["accuracy", "f1_score", "roc_auc"]
        },
        "motif_detection": {
            "dataset": "data/motif_data.csv",
            "task_type": "binary_classification",
            "metrics": ["accuracy", "f1_score", "precision"]
        },
        "sequence_generation": {
            "dataset": "data/generation_data.csv",
            "task_type": "generation",
            "metrics": ["perplexity", "bleu_score", "diversity"]
        }
    }
    
    # Multi-task models
    multi_task_models = [
        {
            "name": "Unified DNA Model",
            "path": "path/to/unified_model",
            "source": "local",
            "task_type": "multi_task"
        }
    ]
    
    # Run multi-task benchmark
    multi_task_results = {}
    
    for model_info in multi_task_models:
        model_name = model_info["name"]
        multi_task_results[model_name] = {}
        
        for task_name, task_config in tasks.items():
            # Load task-specific dataset
            dataset = DNADataset.load_local_data(
                task_config["dataset"],
                seq_col="sequence",
                label_col="label",
                max_length=512
            )
            
            # Evaluate on this task
            task_result = benchmark.evaluate_single_model(
                model_info["model"],
                model_info["tokenizer"],
                dataset,
                metrics=task_config["metrics"]
            )
            
            multi_task_results[model_name][task_name] = task_result
    
    return multi_task_results

# Run multi-task benchmark
multi_task_results = multi_task_benchmark()

# Analyze multi-task performance
analyze_multi_task_performance(multi_task_results)
```

### Example 7: Time-Series Model Evaluation

**Scenario**: Evaluating model performance over time with temporal data.

```python
def time_series_benchmark():
    """Benchmark model performance over time."""
    
    # Load time-series dataset
    time_series_data = pd.read_csv("data/time_series_data.csv")
    time_series_data['date'] = pd.to_datetime(time_series_data['date'])
    
    # Sort by time
    time_series_data = time_series_data.sort_values('date')
    
    # Create temporal splits
    total_days = (time_series_data['date'].max() - time_series_data['date'].min()).days
    interval_days = 30
    num_splits = total_days // interval_days
    
    temporal_results = []
    
    for i in range(num_splits):
        start_idx = i * len(time_series_data) // num_splits
        end_idx = (i + 1) * len(time_series_data) // num_splits
        
        # Test on future data
        test_data = time_series_data.iloc[end_idx:]
        if len(test_data) == 0:
            continue
        
        # Convert to DNALLM dataset format
        test_dataset = DNADataset.from_dataframe(
            test_data,
            seq_col="sequence",
            label_col="label",
            max_length=512
        )
        
        # Evaluate performance
        result = benchmark.evaluate_single_model(
            selected_model,
            selected_tokenizer,
            test_dataset,
            metrics=["accuracy", "f1_score"]
        )
        
        temporal_results.append({
            "time_period": i,
            "start_date": time_series_data.iloc[start_idx]['date'],
            "end_date": time_series_data.iloc[end_idx]['date'],
            "test_size": len(test_data),
            **result
        })
    
    return temporal_results

# Run time-series benchmark
temporal_results = time_series_benchmark()

# Analyze temporal performance
analyze_temporal_performance(temporal_results)
```

## Best Practices Summary

### 1. **Research Applications**
- Use cross-validation for robust evaluation
- Generate publication-ready plots and tables
- Include statistical significance testing
- Document all experimental conditions

### 2. **Production Use Cases**
- Define clear performance criteria
- Test on representative production data
- Monitor performance continuously
- Implement automated alerting

### 3. **Performance Analysis**
- Profile memory and time usage
- Test optimization strategies
- Document performance baselines
- Track performance over time

### 4. **Custom Scenarios**
- Adapt benchmarking to specific requirements
- Implement custom evaluation metrics
- Handle multi-task and time-series data
- Consider domain-specific constraints

## Next Steps

After exploring these examples:

1. **Adapt to Your Use Case**: Modify examples for your specific requirements
2. **Combine Techniques**: Use multiple approaches together
3. **Scale Up**: Apply to larger datasets and model collections
4. **Automate**: Build automated benchmarking pipelines

---

**Ready to implement?** Start with the examples that match your use case, or combine multiple approaches for comprehensive evaluation.

# Basic Inference with DNALLM

This tutorial walks you through the complete process of running inference using the `DNAInference` engine. We will cover loading a model, preparing data, and making predictions on both individual sequences and files.

## 1. The Core Workflow

The inference process in DNALLM follows these steps:
1.  **Load Configuration**: Read the `inference_config.yaml` file.
2.  **Load Model & Tokenizer**: Fetch a pre-trained model and its corresponding tokenizer.
3.  **Initialize `DNAInference`**: Create an inference engine instance with the model, tokenizer, and config.
4.  **Run Inference**: Use the engine's `infer()` method to get predictions.
5.  **Interpret Results**: Analyze the output.

## 2. A Complete Example

Let's put everything together in a Python script. This example demonstrates loading a promoter prediction model and using it to classify DNA sequences.

```python
import os
from dnallm import load_config, load_model_and_tokenizer, DNAInference

def main():
    # 1. Load Configuration
    # Assumes 'inference_config.yaml' is in the same directory
    try:
        configs = load_config("inference_config.yaml")
    except FileNotFoundError:
        print("Error: 'inference_config.yaml' not found. Please create it.")
        return

    # 2. Load Model and Tokenizer
    # This example uses a model from ModelScope. You can also use 'huggingface'.
    model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
    print(f"Loading model '{model_name}'...")
    model, tokenizer = load_model_and_tokenizer(
        model_name, 
        task_config=configs['task'], 
        source="modelscope"
    )

    # 3. Initialize DNAInference Engine
    print("Initializing inference engine...")
    inference_engine = DNAInference(
        model=model,
        tokenizer=tokenizer,
        config=configs
    )

    # --- 4. Run Inference ---

    # Example 1: Infer from a list of sequences
    print("\n--- Predicting from a list of sequences ---")
    seqs_list = [
        "GCACTTTACTTAAAGTAAAAAGAAAAAAACTGTGCGCTCTCCAACTACCGCAGCAACGTGTCGAGCACAGGAACACGTGTCACTTCAGTTCTTCCAATTGCTGGGGCCCACCACTGTTTACTTCTGTACAGGCAGGTGGCCATGCTGATGACACTCCACACTCCTCGACTTTCGTAGCAGCAAGCCACGCGTGACCGAGAAGCCTCGCG",
        "TTGTCATCACATTTGATCAACTACGATTTATGTTGTACTATTCATCTGTTTTCTCCTTTTTTTTTCCCTTATTGACAGGTTGTGGAGGTTCACAACGAACAGAATACAAGAAATTTTGGTAATCATTTGAGGACTTTCATGGGGTATGAATTGTGTGCTATAATAAATTAA"
    ]
    results_from_list = inference_engine.infer(sequences=seqs_list)
    print("Results:")
    print(results_from_list)

    # Example 2: Infer from a file
    print("\n--- Predicting from a file ---")
    # Create a dummy CSV file for demonstration
    seq_file = 'test_data.csv'
    with open(seq_file, 'w') as f:
        f.write("sequence,label\n")
        f.write(f"{seqs_list[0]},1\n")
        f.write(f"{seqs_list[1]},0\n")

    # Run inference and evaluation
    try:
        results_from_file, metrics = inference_engine.infer(
            file_path=seq_file,
            evaluate=True,  # Enable evaluation since the file has labels
            label_col='label' # Specify the column containing labels
        )
        print("\nResults from file (first 2 entries):")
        print({k: results_from_file[k] for k in list(results_from_file)[:2]})
        
        print("\nEvaluation Metrics:")
        print(metrics)

    except FileNotFoundError:
        print(f"Error: The file '{seq_file}' was not found.")
    finally:
        # Clean up the dummy file
        if os.path.exists(seq_file):
            os.remove(seq_file)

if __name__ == "__main__":
    main()
```

After create the `inference.py` script, run the following code to do inference:
```bash
python inference.py
```

A user-friendly Jupyter Notebook is also provided: example/notebooks/inference/inference.ipynb.

## 3. Understanding the Output

The `infer()` method returns a dictionary where each key is the index of a sequence and the value contains its prediction details.

```json
{
    "0": {
        "sequence": "GCACTTTACTTAAAGTA...",
        "label": "positive",
        "scores": {
            "negative": 0.02738,
            "positive": 0.97261
        }
    },
    "1": {
        "sequence": "TTGTCATCACATTTGAT...",
        "label": "negative",
        "scores": {
            "negative": 0.99983,
            "positive": 0.00016
        }
    }
}
```

- **`sequence`**: The input DNA sequence (if `keep_seqs` is `True` during data loading).
- **`label`**: The final predicted label, based on the `task.threshold` from your config. For a binary task, this would be one of the `label_names`.
- **`scores`**: The raw probabilities for each class. This gives you a measure of the model's confidence.

If `evaluate=True`, a second dictionary containing performance metrics (like accuracy, F1-score, AUROC) is also returned.

## 4. Best Practices and Performance

### Error Handling
- **FileNotFoundError**: Always wrap file-based inference in a `try...except` block to handle cases where the input file doesn't exist.
- **OutOfMemoryError**: If you get a CUDA out-of-memory error, the primary solution is to **reduce `batch_size`** in your `inference_config.yaml`.

### Performance Optimization
- **Use a GPU**: For any serious workload, a GPU is essential. Set `device: auto` or `device: cuda`.
- **Tune `batch_size`**: Find the largest `batch_size` that fits in your GPU memory to maximize throughput.
- **Enable FP16/BF16**: If you have a modern NVIDIA GPU (Ampere architecture or newer), setting `use_fp16: true` or `use_bf16: true` can provide a significant speedup with minimal impact on accuracy.
- **Increase `num_workers`**: If you notice your GPU is often waiting for data, increasing `num_workers` can help speed up data loading, especially for large files.

## 5. Common Questions (FAQ)

**Q: Why are my predictions all the same?**
A: This can happen if the model is not well-suited for your data or if the input sequences are too different from what it was trained on. Check that the model you loaded is appropriate for your task.

**Q: How do I get hidden states or attention weights for model interpretability?**
A: The `infer()` method has `output_hidden_states=True` and `output_attentions=True` flags. Setting these will return embeddings and attention scores, which can be accessed via `inference_engine.embeddings`. Be aware that this consumes a large amount of memory.

**Q: Can I run inference on a FASTA file?**
A: Yes. The `infer_file` method automatically handles `.fasta`, `.fa`, `.csv`, `.tsv`, and `.txt` files. For FASTA, the sequence is read directly. For CSV/TSV, you must specify the `seq_col`. Other structured formats such as pickle, arrow, parquet, etc. are also supported.

---

## Next Steps

- [Advanced Inference](advanced_inference.md) - Explore advanced inference features
- [Performance Optimization](performance_optimization.md) - Optimize inference performance
- [Visualization](visualization.md) - Learn about result visualization
- [Inference Troubleshooting](../../faq/inference_troubleshooting.md) - Common inference issues and solutions
# Model Quantization

Quantization is an advanced optimization technique that can dramatically reduce model size and accelerate inference speed, especially on CPUs. It involves converting the model's floating-point weights to lower-precision integers, such as 8-bit integers (INT8).

## 1. What is Quantization?

### The Problem

Large language models have millions or billions of parameters, typically stored as 32-bit floating-point numbers (FP32). This results in:
- **Large Model Size**: A 1-billion parameter model can take up 4 GB of storage.
- **High Memory Bandwidth**: Moving these large weights from RAM to the processor is slow.
- **Slow Computation**: Floating-point arithmetic is more complex than integer arithmetic.

### How to Optimize

Quantization maps the FP32 weights to a smaller set of INT8 values. This provides several benefits:
- **4x Smaller Model**: INT8 uses 4 times less space than FP32.
- **Faster Memory Access**: Smaller weights can be moved more quickly.
- **Faster Computation**: Integer arithmetic is much faster, especially on modern CPUs with specialized instructions (e.g., AVX2, AVX512).

There is a small trade-off in model accuracy, but for many tasks, this is negligible.

## 2. How to Use Quantization in DNALLM

The `transformers` library, which DNALLM is built on, provides easy-to-use tools for quantization through the `bitsandbytes` library.

### 8-bit Quantization for Inference

You can load a model directly in 8-bit precision. This is the easiest way to get started.

1.  **Install `bitsandbytes`:**
    ```bash
    pip install bitsandbytes
    ```

2.  **Load the model with `load_in_8bit=True`:**
    ```python
    from dnallm.utils.load import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        "zhihan1996/DNABERT-2-117M",
        model_type="bert",
        load_in_8bit=True, # Enable 8-bit quantization
        device_map="auto"
    )

    print(f"Model loaded on device: {model.device}")
    print("Model footprint:", model.get_memory_footprint())
    ```
    This will load the model onto the GPU (if available) with its `nn.Linear` layers converted to 8-bit.

### 4-bit Quantization (NF4)

For even greater memory savings, you can use 4-bit quantization (NF4 - NormalFloat 4), which is particularly effective for fine-tuning very large models on consumer GPUs (a technique known as QLoRA).

1.  **Load the model with `load_in_4bit=True`:**
    ```python
    model, tokenizer = load_model_and_tokenizer(
        "GenerTeam/GENERator-eukaryote-1.2b-base",
        model_type="llama",
        load_in_4bit=True, # Enable 4-bit quantization
        device_map="auto"
    )
    
    print("Model footprint:", model.get_memory_footprint())
    ```

**When to use Quantization:**
- **Inference on CPU**: Quantization provides the biggest speed benefits on CPU.
- **Fitting Large Models**: Use 8-bit or 4-bit loading to fit massive models into limited VRAM for inference or QLoRA fine-tuning.
- **Reducing Model Size**: If you need to deploy a model in a size-constrained environment.
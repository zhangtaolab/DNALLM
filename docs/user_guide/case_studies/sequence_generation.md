# Case Study: DNA Sequence Generation with Evo and Mamba Models

This case study demonstrates how to use DNALLM for DNA sequence generation, a powerful technique for designing novel sequences with desired properties. We will showcase this capability using advanced generative models like Evo and Mamba.

## 1. Case Background

Generative models can learn the underlying patterns of biological sequences from large datasets. Once trained, they can generate new sequences that resemble the training data but are entirely novel. This has applications in synthetic biology, such as designing new promoters, enhancers, or other functional elements.

DNALLM integrates state-of-the-art generative architectures, including:
-   **Evo**: A family of models (like Evo-1 and Evo-2) based on the StripedHyena architecture, designed for long-context sequence modeling.
-   **Mamba**: A State Space Model (SSM) architecture that offers efficient, high-quality sequence modeling without the quadratic complexity of traditional Transformers.

In this example, we will load a pre-trained Evo model and use it to generate new DNA sequences from a starting prompt. The same workflow can be applied to Mamba-based models.

## 2. Code

This section provides a complete script for loading a generative model and performing inference.

### Setup

First, ensure you have the necessary dependencies installed for the model you wish to use. For Evo models, this may include `evo-model` and `flash-attn`.

**`inference_evo_config.yaml`:**
Create a configuration file that specifies the generation parameters.

```yaml
# task configuration
task:
  task_type: "sequence_generation"

# inference configuration
inference:
  per_device_eval_batch_size: 1
  output_dir: "./outputs_generation"
  generation_max_length: 400
  temperature: 1.0
  top_k: 50
```

### Python Script

This script loads a pre-trained Evo-2 model and uses it for both sequence generation and scoring.

```python
from dnallm import load_config, load_model_and_tokenizer, DNAInference

# --- 1. Load Configuration and Model ---
# Load settings from the YAML file
configs = load_config("./inference_evo_config.yaml")

# Specify the pre-trained generative model. Here, we use Evo-2.
# You can replace this with other generative models like Mamba.
model_name = "lgq12697/evo2_1b_base"

# Load the model and its corresponding tokenizer
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="modelscope"
)

# --- 2. Initialize Inference Engine ---
inference_engine = DNAInference(
    model=model,
    tokenizer=tokenizer,
    config=configs
)

# --- 3. Generate Sequences ---
# Provide one or more prompts to start the generation process.
# A special token like "@" can be used to start generation from scratch.
prompts = ["@", "ATG"]
print(f"Generating sequences from prompts: {prompts}")
generated_output = inference_engine.generate(prompts)

print("\n--- Generated Sequences ---")
for seq in generated_output:
    print(f"Prompt: {seq['Prompt']}")
    print(f"Generated Sequence: {seq['Output']}")
    print(f"Score: {seq['Score']}\n")

# --- 4. Score Existing Sequences ---
# The same engine can be used to score the likelihood of existing sequences.
sequences_to_score = ["ATCCGCATG", "ATGCGCATG"]
print(f"Scoring sequences: {sequences_to_score}")
scores = inference_engine.scoring(sequences_to_score)

print("\n--- Scored Sequences ---")
for res in scores:
    print(f"Input Sequence: {res['Input']}")
    print(f"Score: {res['Score']}\n")
```

## 3. Expected Results

The script will produce two sets of outputs:

1.  **Generated Sequences**: For each prompt, the model will generate a new DNA sequence. The output will include the original prompt, the generated sequence, and a score (typically the average log-likelihood) indicating how probable the sequence is according to the model.
2.  **Scored Sequences**: For each input sequence provided to the `scoring` method, the model will return its score, which reflects how well the sequence fits the patterns learned by the model.

## 4. Tuning Strategies

-   **`generation_max_length`**: Controls the length of the generated sequences. Adjust this based on your application's needs.
-   **`temperature`**: This parameter controls the randomness of the generation. A higher temperature (e.g., 1.0) produces more diverse and creative outputs, while a lower temperature (e.g., 0.7) makes the output more deterministic and focused.
-   **`top_k`**: This parameter limits the sampling pool to the `k` most likely next tokens. It can prevent the model from picking highly improbable tokens, leading to more coherent sequences.
-   **Prompt Engineering**: The starting prompt can significantly influence the generated sequence. Experiment with different prompts, including biologically meaningful ones, to guide the generation process.

## 5. Troubleshooting

-   **Dependency Issues**: Generative models like Evo and Mamba may have specific dependencies (e.g., `flash-attn`, `causal-conv1d`). Ensure you have installed all required packages for the chosen model. The installation process may involve compilation, which can take time.
-   **`CUDA out of memory`**: Sequence generation can be memory-intensive, especially with long sequences. If you encounter this error, try reducing `per_device_eval_batch_size` or `generation_max_length`.
-   **Low-Quality Generations**: If the generated sequences are repetitive or nonsensical, try adjusting the `temperature` and `top_k` parameters. A very high temperature can lead to randomness, while a very low temperature can cause repetition.
-   **Slow Inference**: Generation is an auto-regressive process and can be slow. For large-scale generation, ensure you are using a GPU. The model's size and the sequence length will be the primary factors affecting speed.
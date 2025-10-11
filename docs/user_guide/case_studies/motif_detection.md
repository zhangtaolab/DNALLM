# Case Study: Motif Detection via In-Silico Saturation Mutagenesis

This case study illustrates an advanced application of DNALLM: identifying critical nucleotide positions and discovering DNA motifs by simulating saturation mutagenesis *in silico*.

## 1. Case Background

Transcription factor binding sites (TFBS), or motifs, are short, recurring patterns in DNA that have a biological function. Discovering these motifs is key to understanding gene regulation. A powerful method for pinpointing functional elements is saturation mutagenesis, where every possible mutation is tested to see its effect on a property, such as promoter strength.

Instead of performing costly and time-consuming lab experiments, we can use a fine-tuned DNALLM model to predict the outcome of these mutations. The workflow is as follows:

1.  **Train a Model**: Fine-tune a DNALLM model on a quantitative task, such as predicting promoter strength (a regression task).
2.  **In-Silico Mutagenesis**: For a given high-performing sequence (e.g., a strong promoter), systematically generate every possible single-nucleotide mutation.
3.  **Predict and Score**: Use the fine-tuned model to predict the strength of each mutated sequence. The change in predicted strength reveals the importance of each nucleotide position.
4.  **Identify Important Regions**: Analyze the mutation effect scores to identify regions where mutations have the largest impact.
5.  **Discover Motifs**: Submit these identified important regions to specialized motif discovery tools like MEME Suite or Homer to find conserved motifs.

## 2. Code

This section provides the code to perform steps 2-4. It assumes you already have a fine-tuned regression model (e.g., for promoter strength).

### Setup

Create a configuration file for the mutagenesis task.

**`inference_config.yaml`:**
```yaml
# task configuration
task:
  task_type: "regression"
  num_labels: 1 # For regression, this is typically 1

# inference configuration
inference:
  per_device_eval_batch_size: 64
  output_dir: "./outputs"
```

### Python Script

This script performs saturation mutagenesis on a single DNA sequence, evaluates the effect of each mutation, and visualizes the results.

```python
from dnallm import load_config, load_model_and_tokenizer, Mutagenesis

# --- 1. Load Configuration and Model ---
# Load settings from the YAML file
configs = load_config("./inference_config.yaml")

# Load a model fine-tuned for a regression task (e.g., promoter strength)
# This model should be able to predict a continuous value from a DNA sequence.
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast"
model, tokenizer = load_model_and_tokenizer(
    model_name, 
    task_config=configs['task'], 
    source="modelscope"
)

# --- 2. Perform In-Silico Mutagenesis ---

# Initialize the mutagenesis tool with the loaded model and tokenizer
mutagenesis = Mutagenesis(config=configs, model=model, tokenizer=tokenizer)

# Define the wild-type (original) sequence to analyze
# This should be a sequence for which your model gives a high score.
wt_sequence = "AATATATTTAATCGGTGTATAATTTCTGTGAAGATCCTCGATACTTCATATAAGAGATTTTGAGAGAGAGAGAGAACCAATTTTCGAATGGGTGAGTTGGCAAAGTATTCACTTTTCAGAACATAATTGGGAAACTAGTCACTTTACTATTCAAAATTTGCAAAGTAGTC"

# Generate all single-nucleotide mutations for the sequence
# This creates a list of mutated sequences in the `mutagenesis.sequences` attribute.
mutagenesis.mutate_sequence(wt_sequence, replace_mut=True)

print(f"Generated {len(mutagenesis.sequences['name'])} sequences (1 wild-type + mutations).")

# --- 3. Predict and Evaluate Mutation Effects ---

# Use the model to predict the output value for the wild-type and all mutated sequences
# The `evaluate` method calculates the difference between each mutant's score and the wild-type score.
# The `strategy` determines how to aggregate scores from the model's output logits.
print("Evaluating the effect of each mutation...")
preds = mutagenesis.evaluate(strategy="mean")

# The `preds` object now contains the raw predictions and the calculated mutation effects.
print("Evaluation complete.")

# --- 4. Visualize the Results ---

# Plot the results as a heatmap, where the color indicates the effect of mutating
# a position to a specific nucleotide.
# This helps visually identify critical positions.
print("Generating and saving mutation effect heatmap...")
mut_plot = mutagenesis.plot(preds, save_path="mut_effects_heatmap.pdf")

# The plot can also be displayed in a notebook
# mut_plot.show()

print("Heatmap saved to mut_effects_heatmap.pdf")
```

### 5. Identify Important Regions and Find Motifs

After running the script, you will have a matrix of importance scores. You can programmatically identify regions with high-impact mutations and extract them. These regions are strong candidates for containing functional motifs.

For example, you can find positions where mutations cause a significant drop in predicted promoter strength and extract the corresponding DNA sequences from the wild-type sequence.

Once you have a set of these important sequences (as a FASTA file, for example `important_regions.fasta`), you can use external motif discovery tools.

**Using MEME Suite:**

Upload your FASTA file to the [MEME Suite web server](https://meme-suite.org/meme/tools/meme) or use the command-line tool:

```bash
# meme <input_fasta_file> [options]
meme important_regions.fasta -dna -o meme_out
```

**Using Homer:**

```bash
# findMotifs.pl <input_fasta_file> <promoter-set> <output_dir> [options]
findMotifs.pl important_regions.fasta fasta homer_out/ -fasta-bg background_sequences.fasta
```

## 3. Expected Results

1.  **Mutation Effect Data**: The `preds` object from `mutagenesis.evaluate()` contains a detailed breakdown of the predicted score for the wild-type and each mutant.
2.  **Heatmap Visualization**: The `mutagenesis.plot()` function generates a PDF file (`mut_effects_heatmap.pdf`). This heatmap shows the original sequence on the x-axis and the four nucleotides (A, C, G, T) on the y-axis. The color of each cell indicates the predicted change in score if the original base is mutated to that nucleotide. Darker colors (e.g., blue or red, depending on the color scheme) highlight mutations that significantly decrease or increase the model's output, indicating critical positions.
3.  **Motif Logo**: After using a tool like MEME, the expected output is a set of discovered motifs, often visualized as sequence logos. A sequence logo provides a graphical representation of the conservation of nucleotides at each position in the motif.

## 4. Tuning Strategies

-   **Model Choice**: The quality of motif detection is highly dependent on the accuracy of the underlying regression model. A well-trained model that accurately predicts the quantitative trait (e.g., promoter strength) is essential.
-   **Selection of Sequences**: Instead of analyzing just one sequence, run the mutagenesis analysis on multiple, diverse, high-activity sequences. Finding conserved patterns of important nucleotides across these sequences will yield more robust motifs.
-   **Thresholding**: When extracting important regions to send to a motif finder, the threshold you use to decide whether a mutation's effect is "significant" is a key parameter. You may need to experiment with this threshold to reduce noise and focus on the most promising regions.

## 5. Troubleshooting

-   **Noisy Heatmap**: If the mutation effect heatmap looks random with no clear hotspots, it could indicate that the underlying model is not sensitive enough or that the chosen sequence does not contain strong functional elements. Try using a better model or a different set of sequences.
-   **No Motifs Found**: If MEME or Homer do not return any significant motifs, consider the following:
    -   The extracted "important regions" may be too long, too short, or too noisy. Try adjusting the threshold for significance.
    -   The background set of sequences used for statistical comparison might be inappropriate. Ensure your background set has a similar nucleotide composition.
    -   The motif may not be well-represented by a simple position weight matrix (PWM), or it might be a structural motif not easily found by these tools.
-   **`CUDA out of memory`**: The prediction step can be memory-intensive. Reduce the `per_device_eval_batch_size` in your configuration file.

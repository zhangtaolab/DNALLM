"""In Silico Mutagenesis Analysis Module.

This module provides tools for evaluating the impact of sequence mutations on
model predictions,
including single nucleotide polymorphisms (
    SNPs),
    deletions,
    insertions,
    and other sequence variations.
"""

import os
import numpy as np
import pandas as pd
from scipy.special import softmax, expit
from tqdm import tqdm
from typing import Any

import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from ..datahandling.data import DNADataset
from .inference import DNAInference
from .plot import plot_muts

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Mutagenesis:
    """Class for evaluating in silico mutagenesis.

    This class provides methods to analyze how sequence mutations affect model
        predictions,
            including single base substitutions, deletions, and
            insertions. It can be used to
            identify important positions in DNA sequences and
            understand model interpretability.

        Attributes:
            model: Fine-tuned model for prediction
            tokenizer: Tokenizer for the model
                    config: Configuration object containing task settings and
                inference parameters
            sequences: Dictionary containing original and mutated sequences
            dataloader: DataLoader for batch processing of sequences
    """

    def __init__(self, model: Any, tokenizer: Any, config: dict):
        """Initialize Mutagenesis class.

        Args:
            model: Fine-tuned model for making predictions
            tokenizer: Tokenizer for encoding DNA sequences
            config: Configuration object containing task settings and
                inference parameters
        """

        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.sequences = None

    def get_inference_engine(self, model: Any, tokenizer: Any) -> DNAInference:
        """Create an inference engine object for the model.

        Args:
            model: The model to be used for inference
            tokenizer: The tokenizer to be used for encoding sequences

        Returns:
            DNAInference: The inference engine object configured with the given
                model and tokenizer
        """

        inference_engine = DNAInference(
            model=model, tokenizer=tokenizer, config=self.config
        )

        return inference_engine

    def mutate_sequence(
        self,
        sequence: str,
        batch_size: int = 1,
        replace_mut: bool = True,
        include_n: bool = False,
        delete_size: int = 0,
        cut_size: int = 0,
        fill_gap: bool = False,
        insert_seq: str | None = None,
        lowercase: bool = False,
        do_encode: bool = True,
    ) -> None:
        """Generate dataset from sequences with various mutation types.

        This method creates mutated versions of the input sequence including:
        - Single base substitutions (A, C, G, T, optionally N)
        - Deletions of specified size
        - Insertions of specified sequences
        - Case transformations

        Args:
            sequence: Single sequence for mutagenesis
            batch_size: Batch size for DataLoader
            replace_mut: Whether to perform single base substitutions
            include_n: Whether to include N base in substitutions
            delete_size: Size of deletions to create (0 for no deletions)
            fill_gap: Whether to fill deletion gaps with N bases
            insert_seq: Sequence to insert at various positions
            lowercase: Whether to convert sequences to lowercase
            do_encode: Whether to encode sequences for the model

        Returns:
            None (modifies internal state)
        """
        # Get the inference config
        pred_config = self.config["inference"]
        # Define the dataset
        sequences = {"name": ["raw"], "sequence": [sequence]}
        # Create mutated sequences
        if replace_mut:
            if include_n:
                base_map = ["A", "C", "G", "T", "N"]
            else:
                base_map = ["A", "C", "G", "T"]
            # Mutate sequence
            for i, base in enumerate(sequence):
                for mut_base in base_map:
                    if base != mut_base:
                        name = f"mut_{i}_{base}_{mut_base}"
                        mutated_sequence = (
                            sequence[:i] + mut_base + sequence[i + 1 :]
                        )
                        sequences["name"].append(name)
                        sequences["sequence"].append(mutated_sequence)
        # Delete mutations
        if delete_size > 0:
            for i in range(len(sequence) - delete_size + 1):
                name = f"del_{i}_{delete_size}"
                if fill_gap:
                    mutated_sequence = (
                        sequence[:i]
                        + "N" * delete_size
                        + sequence[i + delete_size :]
                    )
                else:
                    mutated_sequence = (
                        sequence[:i] + sequence[i + delete_size :]
                    )
                sequences["name"].append(name)
                sequences["sequence"].append(mutated_sequence)
        # Insert mutations
        if insert_seq is not None:
            for i in range(len(sequence) + 1):
                name = f"ins_{i}_{insert_seq}"
                mutated_sequence = sequence[:i] + insert_seq + sequence[i:]
                sequences["name"].append(name)
                sequences["sequence"].append(mutated_sequence)
        # Cut mutations
        if cut_size != 0:
            step = abs(cut_size)
            for i in range(0, len(sequence) - step + 1, step):
                name = f"cut_{i}_{cut_size}"
                if cut_size > 0:
                    mutated_sequence = sequence[i:]
                else:
                    mutated_sequence = sequence[: len(sequence) - i]
                sequences["name"].append(name)
                sequences["sequence"].append(mutated_sequence)
        # Lowercase sequences
        if lowercase:
            sequences["sequence"] = [
                seq.lower() for seq in sequences["sequence"]
            ]
        # Create dataset
        if len(sequences["sequence"]) > 0:
            ds = Dataset.from_dict(sequences)
            dataset = DNADataset(
                ds, self.tokenizer, max_length=pred_config.max_length
            )
            self.sequences = sequences
        # Encode sequences
        if do_encode:
            dataset.encode_sequences(remove_unused_columns=True)
        # Create DataLoader
        if batch_size <= 1:
            batch_size = pred_config.batch_size
        self.dataloader: DataLoader = DataLoader(
            dataset, batch_size=batch_size, num_workers=pred_config.num_workers
        )

    def pred_comparison(self, raw_pred, mut_pred):
        """Compare raw and mutated predictions.

        This method calculates the difference between predictions on the
        original sequence and mutated sequences, providing insights into
        mutation effects.

                Args:
                    raw_pred: Raw predictions from the original sequence
                    mut_pred: Predictions from the mutated sequence

                Returns:
                    Tuple containing (raw_score, mut_score, logfc):
            - raw_score: Processed scores from original sequence
            - mut_score: Processed scores from mutated sequence
            - logfc: Log fold change between mutated and original scores

                Raises:
                    ValueError: If task type is not supported
        """
        # Get the task config
        task_config = self.config["task"]
        # Get the predictions
        if task_config.task_type == "binary":
            raw_score = expit(raw_pred)
            mut_score = expit(mut_pred)
        elif task_config.task_type == "multiclass":
            raw_score = softmax(raw_pred)
            mut_score = softmax(mut_pred)
        elif task_config.task_type == "multilabel":
            raw_score = expit(raw_pred)
            mut_score = expit(mut_pred)
        elif task_config.task_type == "regression":
            raw_score = raw_pred
            mut_score = mut_pred
        elif task_config.task_type == "token":
            raw_score = np.argmax(raw_pred, axis=-1)
            mut_score = np.argmax(mut_pred, axis=-1)
        elif task_config.task_type == "generation":
            raw_score = np.array([raw_pred])
            mut_score = np.array([mut_pred])
        elif task_config.task_type == "mask":
            raw_score = np.array([raw_pred])
            mut_score = np.array([mut_pred])
        elif task_config.task_type == "embedding":
            raw_score = np.array([raw_pred])
            mut_score = np.array([mut_pred])
        else:
            raise ValueError(f"Unknown task type: {task_config.task_type}")

        # eps = 1e-8
        # logfc = np.log2((mut_score + eps) / (raw_score + eps))
        logfc = np.log2(mut_score / raw_score)
        diff = mut_score - raw_score

        return raw_score, mut_score, logfc, diff

    def get_model_device(self, model) -> torch.device:
        """Get the device of the model.

        Returns:
            torch.device: The device on which the model is located
        """
        if hasattr(model, "device"):
            device = model.device
        elif hasattr(model, "parameters"):
            device = next(model.parameters()).device
        else:
            device = torch.device("cpu")

        return device

    @torch.no_grad()
    def mlm_evaluate(self, return_sum: bool = True) -> list[float]:
        """Calculate pseudo-log-likelihood score using masked token prediction.

        This method computes the pseudo-log-likelihood (PLL) score for each
        sequence by iteratively masking each token and predicting it using the
        model. The PLL score is the sum of the log probabilities of the true
        tokens given the masked context.

        Returns:
            List of pseudo-log-likelihood scores for each sequence
        """
        all_logprobs = []
        model = self.model
        tokenizer = self.tokenizer
        device = self.get_model_device(model)
        if len(self.sequences["sequence"]) > 1:
            input_data = tqdm(self.sequences["sequence"], desc="Inferring")
        else:
            input_data = self.sequences["sequence"]
        for seq in input_data:
            toks = tokenizer(
                seq, return_tensors="pt", add_special_tokens=True
            ).to(device)
            input_ids = toks["input_ids"].clone()
            seq_len = input_ids.size(1)
            total = 0.0
            p_values = []

            for i in range(seq_len):
                tok_id = input_ids[0, i].item()
                if tok_id in tokenizer.all_special_ids:
                    continue
                masked = input_ids.clone()
                masked[0, i] = tokenizer.mask_token_id
                masked_inputs = {
                    "input_ids": masked,
                    # "attention_mask": toks["attention_mask"],
                }
                outputs = model(**masked_inputs)
                logits = outputs.logits
                logp = torch.nn.functional.log_softmax(logits[0, i], dim=-1)
                if return_sum:
                    total += float(logp[tok_id].item())
                else:
                    try:
                        token = tokenizer.decode([tok_id])[0]
                    except KeyError:
                        token = tokenizer.convert_ids_to_tokens(tok_id)
                    p_values.append((token, float(logp[tok_id].item())))
            if return_sum:
                all_logprobs.append(total)
            else:
                all_logprobs.append(p_values)
        return all_logprobs

    @torch.no_grad()
    def clm_evaluate(self, return_sum: bool = True) -> list[float]:
        """Calculate sequence log-probability using causal language modeling.

        This method computes the log-probability of each sequence under a
        causal language model by summing the log probabilities of each token
        given its preceding context.

        Returns:
            List of log-probabilities for each sequence
        """
        all_logprobs = []
        model = self.model
        tokenizer = self.tokenizer
        device = self.get_model_device(model)
        if len(self.sequences["sequence"]) > 1:
            input_data = tqdm(self.sequences["sequence"], desc="Inferring")
        else:
            input_data = self.sequences["sequence"]
        for seq in input_data:
            toks = tokenizer(
                seq, return_tensors="pt", add_special_tokens=True
            ).to(device)
            input_ids = toks["input_ids"]
            outputs = model(**toks)
            logits = outputs.logits  # (1, L, V)

            # shift for causal LM: predict token t given tokens < t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_logps = log_probs.gather(
                -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)  # (1, L-1)
            if return_sum:
                seq_logp = float(token_logps.sum().item())
            else:
                # Get all token logp
                seq_logp = [
                    float(token_logps[0, i].item())
                    for i in range(len(token_logps[0]))
                ]
            all_logprobs.append(seq_logp)
        return all_logprobs

    def evaluate(
            self,
            score_type: str = "embedding",
            strategy: str | int = "last",
            do_pred: bool = False,
            reduce_hidden_states: bool = True,
    ) -> list[dict]:
        """Evaluate the impact of mutations on model predictions.

        This method runs predictions on all mutated sequences and compares them
        with the original sequence to calculate mutation effects.

        Args:
            score_type: Type of score to compute:
                "embedding": Use embedding-based scoring
                "logits": Use logits-based scoring
                "probability": Use probability-based scoring
            strategy: Strategy for selecting the score from the log fold\
                change:
                "first": Use the first log fold change
                "last": Use the last log fold change
                "sum": Use the sum of log fold changes
                "mean": Use the mean of log fold changes
                "max": Use the index of the maximum raw score to select\
                    the log fold change
                int: Use the log fold change at the specified index
            do_pred: Whether to perform prediction (if False, only data is\
                prepared)
            reduce_hidden_states: Whether to reduce hidden states when using\
                embedding task

        Returns:
            Dictionary containing predictions and metadata for all sequences:
            - 'raw': Original sequence predictions and metadata
            - mutation names: Individual mutation results with scores and log
                fold changes
        """
        # Load predictor
        inference_engine = self.get_inference_engine(
            self.model, self.tokenizer
        )
        task_type = self.config["task"].task_type
        # Do prediction
        all_predictions = {}
        if task_type == "embedding":
            scores = inference_engine.scoring(
                self.dataloader,
                score_type=score_type,
                reduce_hidden_states=reduce_hidden_states,
                reduce_method=strategy,
            )
            raw_pred = scores[0]["Score"]
            mut_preds = [score["Score"] for score in scores[1:]]
        else:
            if self.config["task"].task_type == "mask":
                scores = self.mlm_evaluate()
            elif self.config["task"].task_type == "generation":
                scores = self.clm_evaluate()
            else:
                outputs = inference_engine.batch_infer(
                    self.dataloader,
                    do_pred=do_pred,
                    return_dict=False,
                )
                scores = outputs[1] if do_pred else outputs[0]
            scores = scores[0] if isinstance(scores, tuple) else scores
            # Get the raw predictions
            raw_pred = (
                scores[0].numpy()
                if isinstance(scores, torch.Tensor)
                else scores[0]
            )
            # Get the mutated predictions
            mut_preds = (
                scores[1:].numpy()
                if isinstance(scores, torch.Tensor)
                else scores[1:]
            )

        # Calculate scores
        def get_score(values: np.ndarray, raw_score: np.ndarray = None):
            # Get final score
            if strategy == "first":
                score = values[0]
            elif strategy == "last":
                score = values[-1]
            elif strategy == "sum":
                score = np.sum(values)
            elif strategy == "mean":
                score = np.mean(values)
            elif strategy == "max":
                idx = raw_score.index(max(raw_score))
                score = values[idx]
            elif isinstance(strategy, int):
                score = values[strategy]
            else:
                score = np.mean(values)
            if np.isnan(score):
                score = 0.0
            return score

        for i, mut_pred in tqdm(
            enumerate(mut_preds), desc="Evaluating mutations"
        ):
            # Get the mutated name
            mut_name = self.sequences["name"][i + 1]
            # Get the mutated sequence
            mut_seq = self.sequences["sequence"][i + 1]
            # Compare the predictions
            raw_score, mut_score, logfc, diff = self.pred_comparison(
                raw_pred, mut_pred
            )
            # Store the results
            if "raw" not in all_predictions:
                all_predictions["raw"] = {
                    "sequence": self.sequences["sequence"][0],
                    "pred": raw_score,
                    "logfc": np.zeros(len(raw_score)),
                    "diff": np.zeros(len(raw_score)),
                    "score": 0.0,
                    "logits": get_score(raw_score),
                }
            all_predictions[mut_name] = {
                "sequence": mut_seq,
                "pred": mut_score,
                "logfc": logfc,
                "diff": diff,
            }
            all_predictions[mut_name]["score"] = get_score(logfc, raw_score)
            all_predictions[mut_name]["score2"] = get_score(diff, raw_score)
            all_predictions[mut_name]["logits"] = get_score(
                mut_score, raw_score
            )

        return all_predictions

    def process_ism_data(
        self,
        ism_results: dict[str, dict],
        strategy: str = "maxabs",
    ) -> np.ndarray:
        """
        Process raw ISM result dictionary to get a single importance score
        per base.

        Args:
            ism_results (Dict[str, Dict]): The raw output from
                the ISM experiment.
            strategy (str): Strategy to aggregate scores at each position.
                            'maxabs': Use the score of the mutation with
                                      the max absolute effect.
                            'mean': Use the mean of all mutation scores.

        Returns:
            np.ndarray: A 1D array of importance scores, one per base pair.
        """
        raw_seq = ism_results["raw"]["sequence"]
        seq_len = len(raw_seq)
        base_scores = np.zeros(seq_len)

        # Group mutations by position
        pos_muts = {}
        for key, value in ism_results.items():
            if key.startswith("mut_"):
                parts = key.split("_")
                pos = int(parts[1])
                if pos not in pos_muts:
                    pos_muts[pos] = []
                pos_muts[pos].append(value["score"])

        # Apply aggregation strategy
        for pos, scores in pos_muts.items():
            if not scores:
                continue
            if strategy in ["maxabs", "min", "max"]:
                max_abs_idx = np.argmax(np.abs(scores))
                base_scores[pos] = scores[max_abs_idx]
            elif strategy == "mean":
                base_scores[pos] = np.mean(scores)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        return base_scores

    def find_hotspots(
        self,
        preds: dict[str, dict],
        strategy="maxabs",
        window_size: int = 10,
        percentile_threshold: float = 90.0,
    ) -> list[tuple[int, int]]:
        """
        Identify hotspot regions from base-level importance scores.

        Args:
            preds (Dict[str, Dict]): The raw output from the ISM experiment.
            strategy (str): Strategy to aggregate scores at each position.
                            'maxabs': Use the score of the mutation with
                                      the max absolute effect.
                            'mean': Use the mean of all mutation scores.
            window_size (int): The size of the sliding window to find hotspots.
            percentile_threshold (float): The percentile of window scores to be
                                        considered a hotspot.

        Returns:
            List[Tuple[int, int]]: A list of (start, end) tuples
                                   for each hotspot.
        """
        # We care about the magnitude of change, so use absolute scores
        base_scores = self.process_ism_data(preds, strategy=strategy)
        abs_scores = pd.Series(np.abs(base_scores))

        # Calculate rolling average of scores
        rolling_mean = abs_scores.rolling(
            window=window_size, center=True, min_periods=1
        ).mean()

        # Determine the score threshold for a hotspot
        threshold = np.percentile(rolling_mean, percentile_threshold)

        # Find regions above the threshold
        hotspot_mask = rolling_mean >= threshold

        # Find contiguous blocks of 'True'
        hotspots = []
        start = -1
        for i, is_hot in enumerate(hotspot_mask):
            if is_hot and start == -1:
                start = i
            elif not is_hot and start != -1:
                hotspots.append((start, i))
                start = -1
        if start != -1:
            hotspots.append((start, len(hotspot_mask)))

        # Return list of hotspot regions with window size
        hotspots_regioned = []
        for i, (start, end) in enumerate(hotspots):
            mid = (start + end) // 2
            window_start = min(max(0, mid - window_size // 2), start)
            window_end = max(
                min(mid + window_size // 2, len(base_scores)), end
            )
            # if the window is within last detected hotspot,
            # skip the current one
            if i > 0:
                prev_start, prev_end = hotspots_regioned[-1]
                if window_start >= prev_start and window_end <= prev_end:
                    continue
            hotspots_regioned.append((window_start, window_end))
        self.hotspots = hotspots_regioned

        return hotspots_regioned

    def prepare_tfmodisco_inputs(
        self, ism_results_list: list[dict[str, dict]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares inputs required for a TF-MoDISco run from
        a list of ISM results.
        """
        print("Preparing inputs for TF-MoDISco...")
        acgt = ["A", "C", "G", "T"]
        acgt_to_idx = {base: i for i, base in enumerate(acgt)}

        all_one_hot, all_hyp_scores = [], []

        for ism_results in ism_results_list:
            raw_seq = ism_results["raw"]["sequence"].upper()
            seq_len = len(raw_seq)

            one_hot = np.zeros((seq_len, 4))
            for i, base in enumerate(raw_seq):
                idx = acgt_to_idx.get(base)
                if idx is not None:
                    one_hot[i, idx] = 1
            all_one_hot.append(one_hot)

            hyp_scores = np.zeros((seq_len, 4))
            for key, value in ism_results.items():
                if key.startswith("mut_"):
                    parts = key.split("_")
                    pos, _, mut_base = int(parts[1]), parts[2], parts[-1]
                    if mut_base in acgt_to_idx:
                        hyp_scores[pos, acgt_to_idx[mut_base]] = value["score"]
            all_hyp_scores.append(hyp_scores)

        one_hot_seqs = np.array(all_one_hot)
        hyp_scores = np.array(all_hyp_scores)
        contrib_scores = hyp_scores * one_hot_seqs
        return one_hot_seqs, hyp_scores, contrib_scores

    def plot(
        self,
        preds: dict,
        width: int | None = None,
        height: int = 400,
        show_score: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Plot the mutagenesis analysis results.

                This method generates visualizations of mutation effects,
            typically as heatmaps,
                bar charts and
            line plots showing how different mutations affect model predictions
        at various positions.

        Args:
            preds: Dictionary containing model predicted scores and metadata
            show_score: Whether to show the score values on the plot
                        save_path: Path to save the plot. If None,
                plot will be shown interactively

        Returns:
            Plot object
        """
        if save_path:
            suffix = os.path.splitext(save_path)[-1]
            if suffix:
                outfile = save_path
            else:
                outfile = os.path.join(save_path, ".pdf")
        else:
            outfile = None
        # Plot heatmap
        pmut = plot_muts(
            preds, width=width, height=height,
            show_score=show_score, save_path=outfile
        )
        return pmut

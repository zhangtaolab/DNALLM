"""DNA Language Model Trainer Module.

This module implements the training process management for DNA language models,
    with the following main features:

1. DNATrainer Class
   - Unified management of model training, evaluation, and prediction processes
      - Support for multiple task types (
       classification,
       regression,
       masked language modeling)
   - Integration of task-specific prediction heads
   - Training parameter configuration
   - Training process monitoring and model saving

2. Core Features:
   - Model initialization and device management
   - Training parameter configuration
   - Training loop control
   - Evaluation metrics calculation
   - Model saving and loading
   - Prediction result generation

3. Supported Training Features:
   - Automatic evaluation and best model saving
   - Training log recording
   - Flexible batch size settings
   - Learning rate and weight decay configuration
   - Distributed training support
   - LoRA (Low-Rank Adaptation) for efficient fine-tuning

Usage Example:
    ```python
    trainer = DNATrainer(
        model=model,
        config=config,
        datasets=datasets
    )
    metrics = trainer.train()
    ```
"""

from typing import Any
from collections.abc import Callable
import torch
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig

from ..datahandling.data import DNADataset
from ..tasks.metrics import compute_metrics


class DNATrainer:
    """DNA Language Model Trainer that supports multiple model types.

        This trainer class provides a unified interface for training, evaluating, and
        predicting
with DNA language models. It supports various task types including
    classification,
    regression, and masked language modeling.

    Attributes:
        model: The DNA language model to be trained
        task_config: Configuration for the specific task
        train_config: Configuration for training parameters
        datasets: Dataset for training and evaluation
        extra_args: Additional training arguments
        trainer: HuggingFace Trainer instance
        training_args: Training arguments configuration
        data_split: Available dataset splits

    Examples:
        ```python
        trainer = DNATrainer(
            model=model,
            config=config,
            datasets=datasets
        )
        metrics = trainer.train()
        ```
    """

    def __init__(
        self,
        model: Any,
        config: dict,
        datasets: DNADataset | None = None,
        extra_args: dict | None = None,
        use_lora: bool = False,
    ):
        """Initialize the DNA trainer.

        Args:
            model: The DNA language model to be trained
                        config: Configuration dictionary containing task and
                training settings
            datasets: Dataset for training and evaluation
            extra_args: Additional training arguments to override defaults
            use_lora: Whether to use LoRA for efficient fine-tuning
        """
        self.model = model
        self.task_config = config["task"]
        self.train_config = config["finetune"]
        self.datasets = datasets
        self.extra_args = extra_args

        # LoRA
        if use_lora:
            print("[Info] Applying LoRA to the model...")
            lora_config = LoraConfig(**config["lora"])
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"[Info] Using {torch.cuda.device_count()} GPUs.")
            self.model = torch.nn.DataParallel(self.model)

        self.set_up_trainer()

    def set_up_trainer(self):
        """Set up the HuggingFace Trainer with appropriate configurations.

        This method configures the training environment by:
        1. Setting up training arguments from configuration
        2. Configuring dataset splits (train/eval/test)
        3. Setting up task-specific metrics computation
        4. Configuring appropriate data collator for different task types
        5. Initializing the HuggingFace Trainer instance

        The method automatically handles:
        - Dataset split detection and validation
        - Task-specific data collator selection
        - Evaluation strategy configuration
        - Metrics computation setup
        """
        # Setup training arguments
        training_args = self.train_config.model_dump()
        if self.extra_args:
            training_args.update(self.extra_args)
        self.training_args = TrainingArguments(
            **training_args,
        )
        # Check if the dataset has been split
        if isinstance(self.datasets.dataset, DatasetDict):
            self.data_split = self.datasets.dataset.keys()
        else:
            self.data_split = []
        # Get datasets
        if "train" in self.data_split:
            train_dataset = self.datasets.dataset["train"]
        else:
            if len(self.data_split) == 0:
                train_dataset = self.datasets.dataset
            else:
                raise KeyError("Cannot find train data.")
        eval_key = [x for x in self.data_split if x not in ["train", "test"]]
        if eval_key:
            eval_dataset = self.datasets.dataset[eval_key[0]]
        elif "test" in self.data_split:
            eval_dataset = self.datasets.dataset["test"]
        else:
            eval_dataset = None
            self.training_args.eval_strategy = "no"

        # Get compute metrics
        compute_metrics = self.compute_task_metrics()
        # Set data collator
        if self.task_config.task_type == "mask":
            from transformers import DataCollatorForLanguageModeling

            mlm_probability = self.task_config.mlm_probability
            mlm_probability = mlm_probability if mlm_probability else 0.15
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.datasets.tokenizer,
                mlm=True,
                mlm_probability=mlm_probability,
            )
        elif self.task_config.task_type == "generation":
            from transformers import DataCollatorForLanguageModeling

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.datasets.tokenizer, mlm=False
            )
        else:
            data_collator = None
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

    def compute_task_metrics(self) -> Callable:
        """Compute task-specific evaluation metrics.

This method returns a callable function that computes appropriate metrics
        for the specific task type (classification, regression, etc.).

        Returns:
Callable: A function that computes metrics for the specific task type
        """
        return compute_metrics(self.task_config)

    def train(self, save_tokenizer: bool = True) -> dict[str, float]:
        """Train the model and return training metrics.

This method executes the training process using the configured HuggingFace
        Trainer,
        automatically saving the best model and optionally the tokenizer.

        Args:
                        save_tokenizer: Whether to save the tokenizer along with the model,
                default True

        Returns:
                        Dictionary containing training metrics including loss,
                learning rate, etc.
        """
        self.model.train()
        train_result = self.trainer.train()
        metrics = train_result.metrics
        # Save the model
        self.trainer.save_model()
        if save_tokenizer:
            self.datasets.tokenizer.save_pretrained(
                self.train_config.output_dir
            )
        return metrics

    def evaluate(self) -> dict[str, float]:
        """Evaluate the model on the evaluation dataset.

                This method runs evaluation on the configured evaluation dataset and
            returns
        task-specific metrics.

        Returns:
Dictionary containing evaluation metrics for the current model state
        """
        self.model.eval()
        result = self.trainer.evaluate()
        return result

    def infer(self) -> dict[str, float]:
        """Generate inference results on the test dataset.

                This method generates inference results on the test dataset if available and
            returns
        both predictions and evaluation metrics.

        Returns:
                        Dictionary containing inference results and
                metrics if test dataset exists,
            otherwise empty dictionary
        """
        self.model.eval()
        result = {}
        if "test" in self.data_split:
            test_dataset = self.datasets.dataset["test"]
            result = self.trainer.predict(test_dataset)
        return result

import os
from typing import Optional, Dict
from transformers import Trainer, TrainingArguments
from datasets import DatasetDict

from ..datasets.data import DNADataset
from ..tasks.metrics import compute_metrics

"""
DNA Language Model Trainer Module

This module implements the training process management for DNA language models, with the following main features:

1. DNALLMTrainer Class
   - Unified management of model training, evaluation, and prediction processes
   - Support for multiple task types (classification, regression)
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

Usage Example:
    trainer = DNATrainer(
        model=model,
        config=config,
        datasets=datasets
    )
    metrics = trainer.train()
"""


class DNATrainer:
    """DNA Language Model Trainer that supports multiple model types.

    This trainer class provides a unified interface for training, evaluating, and predicting
    with DNA language models. It supports various task types including classification,
    regression, and masked language modeling.

    Attributes:
        model: The DNA language model to be trained.
        task_config (dict): Configuration for the specific task.
        train_config (dict): Configuration for training parameters.
        datasets (DNADataset, optional): Dataset for training and evaluation.
        extra_args (Dict, optional): Additional training arguments.

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
        model,
        config: dict,
        datasets: Optional[DNADataset] = None,
        extra_args: Optional[Dict] = None,
    ):
        """Initialize the DNA trainer.

        Args:
            model: The DNA language model to be trained.
            config (dict): Configuration dictionary containing task and training settings.
            datasets (DNADataset, optional): Dataset for training and evaluation.
            extra_args (Dict, optional): Additional training arguments to override defaults.
        """
        self.model = model
        self.task_config = config['task']
        self.train_config = config['finetune']
        self.datasets = datasets
        self.extra_args = extra_args
        
        self.set_up_trainer()
    
    def set_up_trainer(self):
        """Set up the HuggingFace Trainer with appropriate configurations.
        
        This method:
        1. Configures training arguments
        2. Sets up dataset splits
        3. Configures task-specific metrics
        4. Sets up appropriate data collator
        5. Initializes the HuggingFace Trainer
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
            self.data_split = [None]
        # Get datasets
        if "train" in self.data_split:
            train_dataset = self.datasets.dataset["train"]
        else:
            if len(self.data_split) == 1:
                train_dataset = self.datasets.dataset
            else:
                raise KeyError("Cannot find train data.")
        eval_key = [x for x in self.data_split if x not in ['train', 'test']]
        if eval_key:
            eval_dataset = self.datasets.dataset[eval_key[0]]
        elif "test" in self.data_split:
            eval_dataset = self.datasets.dataset['test']
        else:
            eval_dataset = None
        
        # Get compute metrics
        compute_metrics = self.compute_task_metrics()
        # Set data collator
        if self.task_config.task_type == "mask":
            from transformers import DataCollatorForLanguageModeling
            mlm_probability = self.task_config.mlm_probability
            mlm_probability = mlm_probability if mlm_probability else 0.15
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.datasets.tokenizer,
                mlm=True, mlm_probability=mlm_probability
            )
        elif self.task_config.task_type == "generation":
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.datasets.tokenizer,
                mlm=False
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

    def compute_task_metrics(self):
        """Compute task-specific evaluation metrics.

        Returns:
            Callable: A function that computes metrics for the specific task type.
        """
        return compute_metrics(self.task_config)

    def train(self, save_tokenizer: bool = False) -> Dict[str, float]:
        """Train the model and return training metrics.

        Args:
            save_tokenizer (bool, optional): Whether to save the tokenizer along with the model.
                Defaults to False.

        Returns:
            Dict[str, float]: Dictionary containing training metrics.
        """
        self.model.train()
        train_result = self.trainer.train()
        metrics = train_result.metrics
        # Save the model
        self.trainer.save_model()
        if save_tokenizer:
            self.datasets.tokenizer.save_pretrained(self.train_config.output_dir)
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the evaluation dataset.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """
        self.model.eval()
        result = self.trainer.evaluate()
        return result
    
    def predict(self) -> Dict[str, float]:
        """Generate predictions on the test dataset.

        Returns:
            Dict[str, float]: Dictionary containing prediction results and metrics.
        """
        self.model.eval()
        result = {}
        if "test" in self.data_split:
            test_dataset = self.datasets.dataset['test']
            result = self.trainer.predict(test_dataset)
        return result

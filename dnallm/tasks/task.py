"""DNA Language Model Fine-tuning Task Definition Module.

This module defines various task types and
    related components supported by DNA language models
during fine-tuning, including:

1. TaskType: Task type enumeration
      - Binary classification (
       BINARY): e.g.,
       promoter prediction,
       enhancer identification
      - Multi-class classification (
       MULTICLASS): e.g.,
       protein family classification,
       functional region classification
      - Regression (
       REGRESSION): e.g.,
       expression level prediction,
       binding strength prediction
   - Token classification (NER): Named Entity Recognition tasks
   - Generation and embedding tasks for different model architectures

2. TaskConfig: Task configuration class
   - Configures task type, number of labels, label names, etc.
   - Provides threshold settings for binary classification tasks

3. TaskHead: Task-specific prediction heads
   - Provides specialized neural network layers for different task types
      - Supports feature dimensionality reduction and
       dropout to prevent overfitting
   - Automatically selects output dimensions based on task type

4. compute_metrics: Evaluation metric computation
   - Binary: accuracy, F1 score
   - Multi-class: accuracy, macro F1, weighted F1
   - Regression: mean squared error, R-squared value

Usage example:
    task_config = TaskConfig(
        task_type=TaskType.BINARY,
        num_labels=2,
        label_names=["negative", "positive"]
    )
"""

from enum import Enum
from pydantic import BaseModel, Field


class TaskType(Enum):
    """Enumeration of supported task types for DNA language models.

This enum defines all the task types that can be performed by DNA language
    models,
        ranging from basic embedding extraction to complex classification and
        generation tasks.

    Attributes:
                EMBEDDING: Extract embeddings, attention maps, and
            token probabilities for downstream analysis
        MASK: Masked language modeling task for pre-training
        GENERATION: Text generation task for causal language models
        BINARY: Binary classification with two possible labels
        MULTICLASS: Multi-class classification with more than two classes
MULTILABEL: Multi-label classification with multiple binary labels per sample
        REGRESSION: Regression task returning continuous scores
        NER: Named Entity Recognition using token-level classification
    """

        EMBEDDING = "embedding"  # Get embeddings, attention map and token probability for downstream analysis
    MASK = "mask"  # Mask task, for Masked Language Model
    GENERATION = "generation"  # Generation task, for Causal Language Model
    BINARY = (
        "binary_classification"  # Binary classification task with two labels
    )
        MULTICLASS = "multi_class_classification"  # Multi-class classification task that specific the input belongs to which class (more than two)
        MULTILABEL = "multi_label_classification"  # Multi-label classification task with multiple binary labels
    REGRESSION = (
        "regression"  # Regression task which return a score for the input
    )
        NER = "token_classification"  # Token classification task which is usually for Named Entity Recognition


class TaskConfig(BaseModel):
    """Configuration class for different fine-tuning tasks.

    This class provides a structured way to configure task-specific parameters
        including task type, number of labels, label names, and
        classification thresholds.

    Attributes:
        task_type: Type of task to perform (must match regex pattern)
        num_labels: Number of output labels/classes
        label_names: List of label names for classification tasks
        threshold: Classification threshold for binary and multi-label tasks
    """

    def __init__(
        self,
        task_type: str = Field(
            ...,
                        regex="^(embedding|mask|generation|binary|multiclass|multilabel|regression|token)$",
        ),
        num_labels: int = 2,
        label_names: list | None = None,
                threshold: float = 0.5,  # For binary classification and
            multi label classification
    ):
        """Initialize task configuration.

        Args:
                        task_type: Type of task to perform. Must be one of: embedding,
                mask, generation,
                      binary, multiclass, multilabel, regression, or token
            num_labels: Number of output labels/classes (default: 2)
                        label_names: Optional list of label names. If not provided,
                generates generic names
                        threshold: Classification threshold for binary and
                multi-label tasks (default: 0.5)
        """
        self.task_type = task_type
        self.label_names = label_names or [
            f"class_{i}" for i in range(num_labels)
        ]
        self.num_labels = len(self.label_names)
        self.threshold = threshold

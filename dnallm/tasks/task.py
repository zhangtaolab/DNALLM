"""DNA Language Model Fine-tuning Task Definition Module.

This module defines various task types and related components supported
by DNA language models during fine-tuning, including:

1. TaskType: Task type enumeration
   - Binary classification (BINARY): e.g., promoter prediction,
     enhancer identification
   - Multi-class classification (MULTICLASS): e.g., protein family
     classification, functional region classification
   - Regression (REGRESSION): e.g., expression level prediction,
     binding strength prediction
   - Token classification (NER): Named Entity Recognition tasks
   - Generation and embedding tasks for different model architectures

2. TaskConfig: Task configuration class
   - Configures task type, number of labels, label names, etc.
   - Provides threshold settings for binary classification tasks

3. TaskHead: Task-specific prediction heads
   - Provides specialized neural network layers for different task types
   - Supports feature dimensionality reduction and dropout to prevent
     overfitting
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
    """Enum for supported task types in DNALLM.

    This enum defines the various tasks that can be performed with DNA
    language models, including classification, regression, and generation
    tasks.
    """

    EMBEDDING = "embedding"  # Get embeddings, attention map and token
    # probability for downstream analysis
    MASK = "mask"  # Mask task, for Masked Language Model
    GENERATION = "generation"  # Generation task, for Causal Language Model
    BINARY = (
        "binary_classification"  # Binary classification task with two labels
    )
    MULTICLASS = "multi_class_classification"  # Multi-class classification
    # task that specific the input belongs to which class (more than two)
    MULTILABEL = "multi_label_classification"  # Multi-label classification
    # task with multiple binary labels
    REGRESSION = (
        "regression"  # Regression task which return a score for the input
    )
    NER = "token_classification"  # Token classification task which is usually
    # for Named Entity Recognition


class TaskConfig(BaseModel):
    """Configuration class for different fine-tuning tasks.

    This class provides a structured way to configure task-specific
    parameters including task type, number of labels, label names, and
    classification thresholds.

    Attributes:
        task_type: Type of task to perform (must match regex pattern)
        num_labels: Number of output labels/classes
        label_names: List of label names for classification tasks
        threshold: Classification threshold for binary and multi-label tasks
    """

    task_type: str = Field(
        ...,
        pattern="^(embedding|mask|generation|binary|multiclass|multilabel\
                |regression|token)$",
    )
    num_labels: int = 2
    label_names: list | None = None
    threshold: float = (
        0.5  # For binary classification and multi label classification
    )

    def model_post_init(self, __context):
        """Initialize task configuration after model validation.

        This method is called after Pydantic model validation and
        automatically sets appropriate default values based on task type.
        """
        if self.label_names is None:
            self.label_names = [f"class_{i}" for i in range(self.num_labels)]
        self.num_labels = len(self.label_names)

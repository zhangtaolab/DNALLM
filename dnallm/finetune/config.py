from dataclasses import dataclass

@dataclass
class TrainingConfig:
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 100 
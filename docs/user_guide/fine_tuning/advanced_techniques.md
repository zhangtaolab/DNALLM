# Advanced Fine-tuning Techniques

This guide covers advanced fine-tuning techniques including custom training strategies, optimization methods, monitoring, and deployment considerations.

## Overview

Advanced fine-tuning techniques help you:
- Implement custom training loops and loss functions
- Optimize training performance and memory usage
- Monitor and debug training progress effectively
- Deploy fine-tuned models in production environments

## Custom Training Strategies

### Custom Loss Functions

DNALLM allows you to implement custom loss functions for specific use cases.

#### Weighted Loss for Imbalanced Data

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss for imbalanced datasets."""
    
    def __init__(self, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets):
        if self.class_weights is not None:
            loss = F.cross_entropy(
                logits, 
                targets, 
                weight=self.class_weights,
                label_smoothing=self.label_smoothing
            )
        else:
            loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        
        return loss

# Usage in trainer
class CustomDNATrainer(DNATrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Calculate class weights
        labels = [item['label'] for item in self.train_dataset]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        
        # Set custom loss
        self.criterion = WeightedCrossEntropyLoss(class_weights=class_weights)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        
        loss = self.criterion(logits, labels)
        
        if return_outputs:
            return loss, outputs
        return loss
```

#### Focal Loss for Hard Examples

```python
class FocalLoss(nn.Module):
    """Focal loss for handling hard examples."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Usage
trainer = CustomDNATrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.train_data,
    eval_dataset=dataset.val_data,
    config=config
)
trainer.criterion = FocalLoss(alpha=1, gamma=2)
```

### Custom Training Loops

Implement custom training loops for advanced control over the training process.

#### Custom Training with Gradient Accumulation

```python
class CustomTrainer:
    """Custom trainer with advanced features."""
    
    def __init__(self, model, tokenizer, train_dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.config = config
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.finetune.learning_rate,
            weight_decay=config.finetune.weight_decay
        )
        
        self.scheduler = self._get_scheduler()
        self.scaler = torch.cuda.amp.GradScaler() if config.finetune.bf16 else None
    
    def _get_scheduler(self):
        """Get learning rate scheduler."""
        num_training_steps = len(self.train_dataset) // self.config.finetune.per_device_train_batch_size
        num_warmup_steps = int(num_training_steps * self.config.finetune.warmup_ratio)
        
        return torch.optim.lr_scheduler.get_scheduler(
            name=self.config.finetune.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.finetune.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        for step, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.finetune.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.finetune.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.finetune.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.finetune.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Logging
            if step % self.config.finetune.logging_steps == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader)
    
    def train(self, num_epochs):
        """Main training loop."""
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            loss = self.train_epoch(epoch)
            print(f"Epoch {epoch + 1} completed. Average loss: {loss:.4f}")

# Usage
custom_trainer = CustomTrainer(model, tokenizer, dataset.train_data, config)
custom_trainer.train(num_epochs=config.finetune.num_train_epochs)
```

## Advanced Optimization Techniques

### Learning Rate Scheduling

Implement advanced learning rate scheduling strategies.

#### Cosine Annealing with Warm Restarts

```python
class CosineAnnealingWarmRestarts:
    """Cosine annealing with warm restarts."""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self):
        if self.T_cur >= self.T_0:
            self.T_0 *= self.T_mult
            self.T_cur = 0
        
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.T_cur / self.T_0)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.T_cur += 1

# Usage
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
```

#### One Cycle Policy

```python
class OneCycleScheduler:
    """One cycle learning rate scheduler."""
    
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.step_count = 0
        
        # Calculate step counts
        self.warmup_steps = int(total_steps * pct_start)
        self.decay_steps = total_steps - self.warmup_steps
    
    def step(self):
        if self.step_count < self.warmup_steps:
            # Warmup phase
            lr = self.max_lr * (self.step_count / self.warmup_steps)
        else:
            # Decay phase
            decay_step = self.step_count - self.warmup_steps
            lr = self.max_lr * (1 - decay_step / self.decay_steps)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.step_count += 1

# Usage
scheduler = OneCycleScheduler(optimizer, max_lr=1e-3, total_steps=10000)
```

## Memory Optimization Techniques

### Gradient Checkpointing

Implement gradient checkpointing to reduce memory usage.

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Or in configuration
finetune:
  gradient_checkpointing: true
```

### Memory Efficient Attention

Use memory-efficient attention mechanisms.

```python
# Enable memory efficient attention
from transformers import AutoConfig

config = AutoConfig.from_pretrained("zhangtaolab/plant-dnabert-BPE")
config.use_memory_efficient_attention = True

model = AutoModel.from_pretrained("zhangtaolab/plant-dnabert-BPE", config=config)
```

### Dynamic Batching

Implement dynamic batching for variable-length sequences.

```python
class DynamicBatchSampler:
    """Dynamic batch sampler for variable-length sequences."""
    
    def __init__(self, dataset, max_tokens_per_batch=4096, max_batch_size=32):
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_batch_size = max_batch_size
        
        # Sort by length for efficient batching
        self.lengths = [len(item['sequence']) for item in dataset]
        self.indices = sorted(range(len(dataset)), key=lambda i: self.lengths[i])
    
    def __iter__(self):
        batch = []
        current_tokens = 0
        
        for idx in self.indices:
            sequence_length = self.lengths[idx]
            
            # Check if adding this sample would exceed limits
            if (len(batch) >= self.max_batch_size or 
                current_tokens + sequence_length > self.max_tokens_per_batch):
                if batch:
                    yield batch
                    batch = []
                    current_tokens = 0
            
            batch.append(idx)
            current_tokens += sequence_length
        
        if batch:
            yield batch

# Usage
sampler = DynamicBatchSampler(dataset.train_data, max_tokens_per_batch=4096)
dataloader = torch.utils.data.DataLoader(
    dataset.train_data,
    batch_sampler=sampler,
    collate_fn=custom_collate_fn
)
```

## Advanced Monitoring and Debugging

### Custom Callbacks

Implement custom callbacks for advanced monitoring.

```python
class CustomCallback:
    """Custom callback for advanced monitoring."""
    
    def __init__(self, model, tokenizer, eval_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.best_metric = float('inf')
        self.patience_counter = 0
    
    def on_step_end(self, step, logs=None):
        """Called at the end of each step."""
        if step % 100 == 0:
            # Log learning rate
            lr = self.model.optimizer.param_groups[0]['lr']
            print(f"Step {step}, Learning Rate: {lr:.2e}")
            
            # Log gradient norm
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            print(f"Step {step}, Gradient Norm: {total_norm:.4f}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        # Evaluate on validation set
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.eval_dataset:
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = outputs.logits.argmax(-1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        
        avg_loss = total_loss / len(self.eval_dataset)
        accuracy = correct / total
        
        print(f"Epoch {epoch}, Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Early stopping logic
        if avg_loss < self.best_metric:
            self.best_metric = avg_loss
            self.patience_counter = 0
            # Save best model
            self.model.save_pretrained(f"best_model_epoch_{epoch}")
        else:
            self.patience_counter += 1
            if self.patience_counter >= 3:
                print("Early stopping triggered!")
                return True
        
        return False

# Usage
callback = CustomCallback(model, tokenizer, dataset.val_data)
```

### Advanced Logging

Implement comprehensive logging for debugging.

```python
import logging
import json
from datetime import datetime

class AdvancedLogger:
    """Advanced logging for fine-tuning experiments."""
    
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.setup_logging()
        
        # Track metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'memory_usage': []
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = f"{self.log_dir}/{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def log_metrics(self, step, metrics):
        """Log training metrics."""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Log to file
        self.logger.info(f"Step {step}: {json.dumps(metrics, indent=2)}")
        
        # Save metrics to JSON
        with open(f"{self.log_dir}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_model_info(self, model):
        """Log model architecture information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

# Usage
logger = AdvancedLogger("./logs", "promoter_classification")
logger.log_model_info(model)
```

## Hyperparameter Optimization

### Grid Search

Implement grid search for hyperparameter optimization.

```python
def grid_search_hyperparameters():
    """Grid search for hyperparameter optimization."""
    
    # Define hyperparameter grid
    hyperparams = {
        'learning_rate': [1e-5, 2e-5, 5e-5, 1e-4],
        'batch_size': [8, 16, 32],
        'weight_decay': [0.01, 0.05, 0.1],
        'warmup_ratio': [0.1, 0.2, 0.3]
    }
    
    best_config = None
    best_score = float('inf')
    
    # Generate all combinations
    from itertools import product
    keys = hyperparams.keys()
    values = hyperparams.values()
    
    for combination in product(*values):
        config_dict = dict(zip(keys, combination))
        
        # Update configuration
        config.finetune.learning_rate = config_dict['learning_rate']
        config.finetune.per_device_train_batch_size = config_dict['batch_size']
        config.finetune.weight_decay = config_dict['weight_decay']
        config.finetune.warmup_ratio = config_dict['warmup_ratio']
        
        # Train and evaluate
        score = train_and_evaluate(config)
        
        print(f"Config: {config_dict}, Score: {score}")
        
        if score < best_score:
            best_score = score
            best_config = config_dict.copy()
    
    print(f"Best config: {best_config}")
    print(f"Best score: {best_score}")
    
    return best_config

def train_and_evaluate(config):
    """Train model and return validation score."""
    # Implementation of training and evaluation
    # Return validation loss or other metric
    pass
```

### Bayesian Optimization

Use Bayesian optimization for more efficient hyperparameter search.

```python
from skopt import gp_minimize
from skopt.space import Real, Integer

def objective(params):
    """Objective function for Bayesian optimization."""
    lr, batch_size, weight_decay, warmup_ratio = params
    
    # Update configuration
    config.finetune.learning_rate = lr
    config.finetune.per_device_train_batch_size = int(batch_size)
    config.finetune.weight_decay = weight_decay
    config.finetune.warmup_ratio = warmup_ratio
    
    # Train and evaluate
    score = train_and_evaluate(config)
    return score

def bayesian_optimization():
    """Bayesian optimization for hyperparameters."""
    
    # Define search space
    space = [
        Real(1e-5, 1e-4, name='learning_rate', prior='log-uniform'),
        Integer(4, 64, name='batch_size'),
        Real(0.001, 0.1, name='weight_decay'),
        Real(0.05, 0.5, name='warmup_ratio')
    ]
    
    # Run optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=20,
        random_state=42,
        n_initial_points=5
    )
    
    print(f"Best parameters: {result.x}")
    print(f"Best score: {result.fun}")
    
    return result.x
```

## Next Steps

After mastering these advanced techniques:

1. **Explore Real-world Examples**: See [Examples and Use Cases](examples.md)
2. **Configuration Options**: Check [detailed configuration](configuration.md) options
3. **Troubleshooting**: Visit [common issues and solutions](troubleshooting.md)
4. **Deployment**: Learn about model deployment and serving

---

**Ready for real-world examples?** Continue to [Examples and Use Cases](examples.md) to see these advanced techniques in action.

# API Reference

This section provides auto-generated API documentation for all DNALLM modules.

## Core Modules

- **[Configuration](configuration/configs.md)** - Configuration management classes and loaders
- **[Models](models/model.md)** - Model loading and utilities
  - [Automatic Loading](models/modeling_auto.md) - Auto model detection and loading
  - [Classification Heads](models/head.md) - Task-specific classification heads
  - [Tokenizers](models/tokenizer.md) - DNA tokenization utilities
  - [Special Models](models/special/) - Specialized model implementations
    - [Basenji2](models/special/basenji2.md) - Basenji-2 model support
    - [Borzoi](models/special/borzoi.md) - Borzoi model support
    - [DNABERT2](models/special/dnabert2.md) - DNABERT-2 model support
    - [EVO](models/special/evo.md) - EVO-1/EVO-2 model support
    - [Enformer](models/special/enformer.md) - Enformer model support
    - [GPN](models/special/gpn.md) - GPN model support
    - [LucaOne](models/special/lucaone.md) - LucaOne model support
    - [megaDNA](models/special/megadna.md) - megaDNA model support
    - [MutBERT](models/special/mutbert.md) - MutBERT model support
    - [Omni-DNA](models/special/omnidna.md) - Omni-DNA model support
    - [SPACE](models/special/space.md) - SPACE model support
- **[Data Handling](datahandling/data.md)** - Dataset management
  - [Automatic Builders](datahandling/dataset_auto.md) - Auto dataset creation
- **[Tasks](tasks/task.md)** - Task definitions and evaluation
  - [Metrics](tasks/metrics.md) - Evaluation metrics

## Training and Inference

- **[Fine-tuning](finetune/trainer.md)** - Training pipeline and utilities
- **[Inference](inference/inference.md)** - Prediction engine
  - [Benchmark](inference/benchmark.md) - Multi-model comparison
  - [Mutagenesis](inference/mutagenesis.md) - In-silico mutation analysis
  - [Interpretation](inference/interpret.md) - Model interpretation
  - [Visualization](inference/plot.md) - Plotting utilities

## MCP Server

- **[Server](mcp/server.md)** - Main server implementation
- **[Config Manager](mcp/config_manager.md)** - Configuration management
- **[Config Validators](mcp/config_validators.md)** - Input validation
- **[Model Manager](mcp/model_manager.md)** - Model lifecycle management

## Utilities

- **[Sequence Utils](utils/sequence.md)** - DNA sequence processing
- **[Logger](utils/logger.md)** - Logging configuration
- **[Support](utils/support.md)** - Helper functions

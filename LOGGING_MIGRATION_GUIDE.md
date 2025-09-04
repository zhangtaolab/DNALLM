# DNALLM Logging Migration Guide

This guide helps developers migrate from `print` statements to proper logging in the DNALLM project.

## Overview

The DNALLM project has been updated to use a centralized logging system instead of `print` statements. This improves:
- **Production readiness**: Proper log levels and formatting
- **Debugging**: Better error tracking and debugging information
- **Performance**: Configurable logging levels
- **Maintainability**: Centralized logging configuration

## Quick Migration

### 1. Import the Logger

```python
from dnallm.utils import get_logger

logger = get_logger("your.module.name")
```

### 2. Replace Print Statements

| Old Print Statement | New Logging Call |
|-------------------|------------------|
| `print("Info message")` | `logger.info("Info message")` |
| `print("Error occurred")` | `logger.error("Error occurred")` |
| `print("Warning message")` | `logger.warning("Warning message")` |
| `print("Debug info")` | `logger.debug("Debug info")` |
| `print("✅ Success!")` | `logger.success("Success!")` |
| `print("❌ Failed!")` | `logger.failure("Failed!")` |
| `print("🔄 Processing...")` | `logger.progress("Processing...")` |
| `print("⚠️ Warning!")` | `logger.warning_icon("Warning!")` |

## Detailed Migration Examples

### Basic Information Logging

```python
# Before
print("Loading model...")
print(f"Model loaded: {model_name}")

# After
logger.info("Loading model...")
logger.info(f"Model loaded: {model_name}")
```

### Success and Error Messages

```python
# Before
print("✅ Model loaded successfully!")
print("❌ Failed to load model")

# After
logger.success("Model loaded successfully!")
logger.failure("Failed to load model")
```

### Progress Updates

```python
# Before
print("🔄 Processing sequences...")
print("📊 Computing metrics...")

# After
logger.progress("Processing sequences...")
logger.info("📊 Computing metrics...")
```

### Error Handling

```python
# Before
try:
    result = some_operation()
    print("Operation completed")
except Exception as e:
    print(f"Error: {e}")

# After
try:
    result = some_operation()
    logger.success("Operation completed")
except Exception as e:
    logger.error(f"Operation failed: {e}")
```

### Debug Information

```python
# Before
print(f"Debug: Processing {len(sequences)} sequences")
print(f"Debug: Using device {device}")

# After
logger.debug(f"Processing {len(sequences)} sequences")
logger.debug(f"Using device {device}")
```

## Advanced Usage

### Custom Logger Names

```python
# Use descriptive logger names for better organization
logger = get_logger("dnallm.inference.predictor")
logger = get_logger("dnallm.mcp.model_manager")
logger = get_logger("dnallm.cli")
```

### Logging Context

```python
from dnallm.utils import LoggingContext

# Temporarily change logging level
with LoggingContext("DEBUG"):
    logger.debug("This will be shown even if global level is INFO")
```

### Function Call Logging

```python
from dnallm.utils import log_function_call

@log_function_call
def my_function(param1, param2):
    # Function calls and results will be automatically logged
    return param1 + param2
```

## Configuration

### Setting Log Level

```python
from dnallm.utils import setup_logging

# Set global logging level
setup_logging(level="DEBUG")  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Custom Log File

```python
from dnallm.utils import setup_logging

# Log to specific file
setup_logging(level="INFO", log_file="custom.log")
```

## Migration Checklist

### For Each Module:

- [ ] Import logger: `from dnallm.utils import get_logger`
- [ ] Create logger instance: `logger = get_logger("module.name")`
- [ ] Replace `print()` with appropriate logging calls
- [ ] Use semantic logging methods (`success`, `failure`, `progress`)
- [ ] Add debug logging for development
- [ ] Test logging output

### For CLI Modules:

- [ ] Replace user-facing `print()` with `logger.info()`
- [ ] Replace error `print()` with `logger.error()`
- [ ] Keep success messages with `logger.success()`

### For Core Modules:

- [ ] Add debug logging for troubleshooting
- [ ] Use appropriate log levels
- [ ] Include context in log messages

## Best Practices

### 1. Use Semantic Methods

```python
# Good
logger.success("Model loaded successfully")
logger.failure("Failed to load model")
logger.progress("Loading model...")

# Avoid
logger.info("✅ Model loaded successfully")
logger.error("❌ Failed to load model")
```

### 2. Include Context

```python
# Good
logger.info(f"Loading model: {model_name} from {source}")
logger.error(f"Failed to load model {model_name}: {error}")

# Avoid
logger.info("Loading model")
logger.error("Failed to load model")
```

### 3. Use Appropriate Levels

```python
# DEBUG: Detailed information for debugging
logger.debug(f"Processing batch {batch_idx} with {len(sequences)} sequences")

# INFO: General information about program execution
logger.info("Starting model inference")

# WARNING: Something unexpected happened but program continues
logger.warning("Using CPU fallback, GPU not available")

# ERROR: Error occurred but program can continue
logger.error("Failed to load model, using default")

# CRITICAL: Serious error, program may not continue
logger.critical("Out of memory, cannot continue")
```

### 4. Avoid Logging Sensitive Information

```python
# Good
logger.info("User authentication successful")

# Avoid
logger.info(f"User {username} authenticated with password {password}")
```

## Testing Logging

### Check Log Output

```python
import logging
from dnallm.utils import get_logger

# Test logging setup
logger = get_logger("test.module")
logger.info("Test message")
logger.success("Test success")
logger.failure("Test failure")
```

### Verify Log Files

```bash
# Check log files are created
ls -la logs/

# View log content
tail -f logs/dnallm.log
```

## Troubleshooting

### Common Issues

1. **Logger not found**: Make sure to import from `dnallm.utils`
2. **No log output**: Check logging level configuration
3. **Duplicate messages**: Ensure logger is created once per module
4. **Missing log files**: Check write permissions in project directory

### Debug Logging Setup

```python
from dnallm.utils import setup_logging

# Enable debug logging
setup_logging(level="DEBUG")
```

## Migration Status

### Completed Modules:
- [x] `dnallm.utils.logger` - New logging system
- [x] `dnallm.mcp.run_tests` - Test runner
- [x] `dnallm.mcp.model_manager` - Model management
- [x] `dnallm.cli.cli` - Main CLI
- [x] `dnallm.inference.predictor` - Core predictor

### Pending Modules:
- [ ] `dnallm.inference.benchmark` - Benchmarking
- [ ] `dnallm.inference.plot` - Plotting functions
- [ ] `dnallm.datahandling.data` - Data handling
- [ ] `dnallm.models.model` - Model loading
- [ ] `dnallm.finetune.trainer` - Training
- [ ] CLI modules (`predict.py`, `train.py`, etc.)

## Benefits After Migration

1. **Better Debugging**: Structured log messages with timestamps
2. **Production Ready**: Configurable log levels and file output
3. **Performance**: No performance impact when logging is disabled
4. **Maintainability**: Centralized logging configuration
5. **Monitoring**: Easy integration with log monitoring systems

## Next Steps

1. Complete migration of remaining modules
2. Add comprehensive debug logging
3. Set up log rotation for production
4. Integrate with monitoring systems
5. Create logging documentation for users

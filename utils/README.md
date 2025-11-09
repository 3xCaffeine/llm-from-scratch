# Utilities

This directory contains shared utilities and helper functions used across language model from scratch project. These modules provide common functionality for training, data handling, model management, and project organization.

## Overview

The utils module serves as foundation for entire project, providing reusable components that simplify development, ensure consistency, and reduce code duplication across different experiments and implementations.

## Files

### `import_helper.py`
Project import management utilities:

**Key Functions:**
- `setup_project_imports()` - Automatically configures Python path for cross-directory imports
- `get_project_root()` - Returns project root directory path

**Purpose:**
- Enables clean imports from any subdirectory
- Eliminates relative import issues
- Provides consistent project structure access

### `loss.py`
Loss calculation and evaluation utilities:

**Key Functions:**
- `calc_loss_batch()` - Compute cross-entropy loss for a single batch
- `calc_loss_loader()` - Calculate average loss over entire dataloader
- `plot_losses()` - Visualize training and validation losses

**Features:**
- Efficient batch-wise loss computation
- Flexible evaluation with configurable batch limits
- Professional loss visualization with dual x-axes (epochs and tokens)
- Device-aware tensor operations

### `train.py`
Training and generation utilities:

**Key Functions:**
- `generate()` - Advanced text generation with sampling strategies
- `train_model_simple()` - Basic training loop implementation
- `text_to_token_ids()` / `token_ids_to_text()` - Text-token conversion utilities

**Generation Features:**
- Temperature-controlled sampling
- Top-k filtering for diverse generation
- Early stopping with EOS tokens
- Greedy decoding fallback

**Training Features:**
- Simple yet effective training loop
- Progress tracking and validation
- Checkpoint saving capabilities

### `gpt2_download.py`
GPT-2 model weight downloading utilities:

**Features:**
- Downloads all GPT-2 model sizes (124M, 355M, 774M, 1558M)
- Uses HuggingFace's fast Rust-based downloader (hf_transfer)
- Saves models locally for offline access
- Organizes weights by model size

## Configuration

### Environment Variables
- `HF_HUB_ENABLE_HF_TRANSFER=1` - Enables fast Rust-based downloads

### Default Paths
- Models directory: `models/`
- Checkpoints: `checkpoints/`
- Logs: `logs/`

## Performance Optimizations

### Memory Efficiency
- In-place operations where possible
- Efficient tensor reshaping
- Minimal memory allocations

### Computational Efficiency
- Vectorized operations
- Batch processing
- Device-aware computations

### I/O Optimization
- Fast downloaders (hf_transfer)
- Efficient file handling
- Minimal disk access

## Future Extensions

Planned utilities include:
- Advanced learning rate schedulers
- Model checkpointing utilities
- Distributed training helpers
- Evaluation metrics beyond loss
- Data preprocessing pipelines
- Hyperparameter optimization tools
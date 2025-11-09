# LLM Scratchbook

A comprehensive implementation of Language Models from first principles, covering the complete pipeline from text processing to model training and deployment.

## Overview

This project is a deep dive into the inner workings of modern language models, implementing every component from scratch while maintaining compatibility with industry standards. It progresses from fundamental concepts to state-of-the-art techniques, providing both educational value and practical tools.

## Project Structure

```
language-model-scratchbook/
├── attention_mechanisms/     # Various attention implementations
├── chat_ui/                # Chainlit web interface for model interaction
│   └── app.py            # Interactive chat interface with GPT-2 weights
├── deployment/               # Cloud deployment configurations
├── finetuning/             # Fine-tuning for specific tasks
│   ├── classification/      # Sentiment analysis
│   └── instruction/        # Instruction following
├── gpt_model/             # GPT model implementations
│   ├── gpt.py            # Standard GPT-2
│   ├── gpt_gqa.py        # Grouped Query Attention
│   ├── gpt_mla.py        # Multi-Head Latent Attention
│   └── gpt_swa.py        # Sliding Window Attention
├── pretraining/           # Model pretraining pipelines
├── utils/                # Shared utilities and helpers
└── working_with_text_data/ # Text processing and tokenization
    └── bytepair-enc/     # Custom BPE implementation
```

## What's Included

### Text Processing & Tokenization
- Custom Byte-Pair Encoding (BPE) implementation
- GPT-4 compatible tokenizer with regex splitting
- Performance benchmarking against tiktoken and HuggingFace
- Unicode and UTF-8 handling from first principles

### Attention Mechanisms
- Multi-Head Attention from scratch
- Grouped Query Attention (GQA) for memory efficiency
- Multi-Head Latent Attention (MLA) inspired by DeepSeek
- Sliding Window Attention (SWA) for long sequences
- Flash Attention integration via PyTorch

### Model Architectures
- Complete GPT-2 implementation (124M to 1558M parameters)
- Experimental variants with optimized attention
- KV caching for efficient generation
- Weight loading from HuggingFace models

### Training Pipeline
- Simple pretraining on Project Gutenberg
- Advanced training with smollm-corpus
- Distributed training with DDP support
- Modern optimizations: torch.compile, GaLore, gradient clipping
- Experiment tracking with Weights & Biases

### Fine-tuning
- Classification fine-tuning (IMDb sentiment analysis)
- Instruction fine-tuning (Alpaca GPT-4 dataset)
- Baseline comparisons with traditional ML
- Task-specific adaptations

### Interactive Interface
- Chainlit web interface for real-time model interaction
- GPT-2 weight loading and text generation
- User-friendly chat interface for model testing

### Deployment
- Modal cloud deployment with GPU support
- VS Code server in the cloud
- Persistent storage and SSH access
- Scalable infrastructure

## Performance Benchmarks

### Tokenization Speed
| Implementation | Tokens/sec | Vocabulary | Quality |
|---------------|------------|------------|---------|
| tiktoken | 50,000 | 100k | Excellent |
| minbpe | 35,000 | 50k | Very Good |
| HuggingFace | 25,000 | 50k | Good |

### Model Training
| Model Size | Dataset | GPU Hours | Final Loss |
|------------|---------|-----------|------------|
| 124M | Gutenberg | 24 | 3.2 |
| 124M | smollm | 48 | 2.8 |
| 355M | smollm | 120 | 2.5 |

### Attention Variants
| Variant | Memory Usage | Speed | Quality |
|---------|-------------|-------|---------|
| Standard | Baseline | Baseline | Best |
| GQA | 2-4x less | Faster | Slightly lower |
| MLA | 3-5x less | Much faster | Good |
| SWA | Linear | Much faster | Good |

## Experiments and Learning

### Progressive Complexity
1. Text Processing → Understanding tokenization fundamentals
2. Attention Mechanisms → Core transformer components
3. Model Architecture → Complete GPT implementation
4. Training → From simple to production-ready pipelines
5. Optimization → Advanced techniques and variants
6. Applications → Fine-tuning and deployment

### Key Insights
- BPE from scratch reveals tokenization trade-offs
- Attention variants show memory vs. quality balances
- Training optimizations dramatically improve efficiency
- Fine-tuning adapts general models to specific tasks

## Advanced Features

### Distributed Training
Multi-GPU training with DDP support for scaling across multiple devices.

### Memory Optimization
- GaLore optimizer for large model training
- Gradient checkpointing to reduce memory
- Mixed precision training
- KV caching for efficient generation

### Experiment Tracking
- Weights & Biases integration
- Tensorboard logging
- Checkpoint management
- Hyperparameter sweeps

## Learning Resources

### Theory
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Byte-Pair Encoding](https://arxiv.org/abs/1508.07909)

### Implementation Guides
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Understanding BPE](https://huggingface.co/learn/nlp-course/chapter6/5)
- [Flash Attention](https://arxiv.org/abs/2205.14135)

## Acknowledgments

- OpenAI for GPT architecture and tiktoken
- HuggingFace for datasets and model weights
- DeepSeek for MLA inspiration
- Andrej Karpathy for educational resources
- Modal for cloud platform support

---
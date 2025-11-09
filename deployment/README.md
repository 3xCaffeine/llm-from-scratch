# Deployment

This directory contains deployment configurations for running the LLM models in various environments, particularly cloud-based GPU setups.

## Overview

The deployment module provides infrastructure-as-code solutions for running LLM experiments and development environments with GPU support. Currently focused on Modal cloud platform for scalable, on-demand GPU computing.

## Files

### `modal_vscode.py`
A complete Modal application that launches a VS Code Server with GPU support for LLM development.

**Features:**
- GPU Support: Configured with NVIDIA L40S GPU and CUDA 12.9.1
- VS Code Server: Full-featured web-based IDE accessible via browser
- SSH Access: Secure shell access for command-line operations
- Persistent Storage: Uses Modal volumes for workspace persistence
- Repository Integration: Automatically clones the `language-model-scratchbook` repository
- Security: Token-based authentication and SSH key support

**Configuration:**
- CPU: 8 cores
- Memory: 32 GB
- GPU: NVIDIA L40S
- Timeout: 1 hour
- Volume: Persistent workspace storage

## Dependencies

- Modal Python SDK
- NVIDIA CUDA runtime
- VS Code Server
- OpenSSH Server

## Future Extensions

- Support for other cloud platforms (AWS, GCP, Azure)
- Docker containerization for local deployment
- Kubernetes manifests for orchestration
- CI/CD pipeline integration
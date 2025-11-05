"""
Import helper utilities for the LLM from Scratch project.

This module provides utilities to help scripts import modules from the project
regardless of where they are run from.
"""

import sys
from pathlib import Path


def setup_project_imports():
    """
    Add the project root directory to sys.path if not already present.
    
    This allows scripts in subdirectories (like pretraining/) to import
    modules from the project root (like gpt_model).
    
    Usage:
        from utils import setup_project_imports
        setup_project_imports()
        
        # Now you can import project modules
        from gpt_model import GPTModel
    
    Returns:
        Path: The project root directory
    """
    # Get the directory containing this file (utils/)
    utils_dir = Path(__file__).parent.absolute()
    
    # Get the project root (parent of utils/)
    project_root = utils_dir.parent
    
    # Add to sys.path if not already present
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: The project root directory
    """
    utils_dir = Path(__file__).parent.absolute()
    return utils_dir.parent

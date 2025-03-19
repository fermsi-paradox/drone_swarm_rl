"""
Common utility functions for the drone swarm RL project.

This module provides common utility functions used across training and visualization.
"""

import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def set_random_seed(seed: int):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate Euclidean distance between two 3D positions.
    
    Args:
        pos1: First position (x, y, z)
        pos2: Second position (x, y, z)
        
    Returns:
        Euclidean distance
    """
    return np.sqrt(np.sum((pos1 - pos2) ** 2))


def calculate_distances(positions: np.ndarray) -> np.ndarray:
    """Calculate pairwise distances between all positions.
    
    Args:
        positions: Array of positions with shape (n, 3)
        
    Returns:
        Distance matrix with shape (n, n)
    """
    n = positions.shape[0]
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            distances[i, j] = calculate_distance(positions[i], positions[j])
            distances[j, i] = distances[i, j]
    
    return distances


def exponential_moving_average(values: List[float], alpha: float = 0.1) -> List[float]:
    """Calculate exponential moving average.
    
    Args:
        values: List of values
        alpha: Smoothing factor
        
    Returns:
        Smoothed values
    """
    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i-1])
    return smoothed


def save_metrics(metrics: Dict, filepath: str):
    """Save metrics to a file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the metrics
    """
    import json
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save metrics to file
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: str) -> Dict:
    """Load metrics from a file.
    
    Args:
        filepath: Path to the metrics file
        
    Returns:
        Dictionary of metrics
    """
    import json
    
    # Load metrics from file
    with open(filepath, "r") as f:
        metrics = json.load(f)
    
    return metrics


def get_device() -> torch.device:
    """Get available device for PyTorch.
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s" 
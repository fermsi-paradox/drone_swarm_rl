#!/usr/bin/env python3
"""
Main script for running the drone swarm RL pipeline.

This script provides a command-line interface for training and visualizing
drone swarm reinforcement learning agents.
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional

from src.utils.common import set_random_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Drone Swarm RL Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a drone swarm RL agent")
    train_parser.add_argument(
        "--config", 
        type=str, 
        default="src/configs/default.yaml",
        help="Path to the configuration file"
    )
    train_parser.add_argument(
        "--output", 
        type=str, 
        default="src/models",
        help="Directory to save trained models and logs"
    )
    train_parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to model to resume training from"
    )
    train_parser.add_argument(
        "--timesteps", 
        type=int, 
        default=None,
        help="Total timesteps to train (overrides config)"
    )
    train_parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    # Visualize command
    vis_parser = subparsers.add_parser("visualize", help="Visualize a trained agent")
    vis_parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to the trained model file"
    )
    vis_parser.add_argument(
        "--scenario", 
        type=str, 
        default="src/configs/combat_scenario.yaml",
        help="Path to the scenario configuration file"
    )
    vis_parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    vis_parser.add_argument(
        "--episodes", 
        type=int, 
        default=5,
        help="Number of episodes to render"
    )
    vis_parser.add_argument(
        "--record", 
        action="store_true",
        help="Record video of the visualization"
    )
    vis_parser.add_argument(
        "--output", 
        type=str, 
        default="videos",
        help="Directory to save recorded videos"
    )
    
    return parser.parse_args()


def train_agent(args):
    """Train a drone swarm RL agent.
    
    Args:
        args: Command line arguments
    """
    from src.training.train import main as train_main
    
    # Update sys.argv to pass args to the training script
    orig_argv = sys.argv
    sys.argv = [
        sys.argv[0],
        "--config", args.config,
        "--output", args.output,
        "--seed", str(args.seed)
    ]
    
    if args.resume:
        sys.argv.extend(["--resume", args.resume])
    
    if args.timesteps:
        sys.argv.extend(["--timesteps", str(args.timesteps)])
    
    # Run training
    train_main()
    
    # Restore original argv
    sys.argv = orig_argv


def visualize_agent(args):
    """Visualize a trained agent.
    
    Args:
        args: Command line arguments
    """
    from src.visualization.render import main as render_main
    
    # Update sys.argv to pass args to the visualization script
    orig_argv = sys.argv
    sys.argv = [
        sys.argv[0],
        "--model", args.model,
        "--scenario", args.scenario,
        "--seed", str(args.seed),
        "--episodes", str(args.episodes)
    ]
    
    if args.record:
        sys.argv.append("--record")
        sys.argv.extend(["--output", args.output])
    
    # Run visualization
    render_main()
    
    # Restore original argv
    sys.argv = orig_argv


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Run selected command
    if args.command == "train":
        train_agent(args)
    elif args.command == "visualize":
        visualize_agent(args)
    else:
        print("Please specify a command: train or visualize")
        print("Run with --help for more information.")


if __name__ == "__main__":
    main() 
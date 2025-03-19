#!/usr/bin/env python3
"""
Military Drone Swarm Training and Visualization

This script provides a simple command-line interface to train, analyze, and
visualize military-grade drone swarm models in a single command.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("military_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("military_pipeline")

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def parse_args():
    """Parse command line arguments for the military training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train, analyze, and visualize military drone swarm models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/configs/military_combat.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/military",
        help="Directory to save outputs"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,  # Reduced default for demonstration
        help="Number of timesteps to train for (0 to use config value)"
    )
    
    # Mode selection
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train a new model"
    )
    
    mode_group.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze model performance"
    )
    
    mode_group.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize model performance"
    )
    
    mode_group.add_argument(
        "--full_pipeline",
        action="store_true",
        help="Run full pipeline (train, analyze, visualize)"
    )
    
    # Model specification for analysis/visualization
    model_group = parser.add_argument_group("Model Specification")
    model_group.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to existing model (for analyze/visualize without training)"
    )
    
    # Training parameters
    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    training_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use for training"
    )
    
    training_group.add_argument(
        "--checkpoint_freq",
        type=int,
        default=0,
        help="Override checkpoint frequency (0 to use config value)"
    )
    
    training_group.add_argument(
        "--eval_freq",
        type=int,
        default=0,
        help="Override evaluation frequency (0 to use config value)"
    )
    
    # Visualization parameters
    viz_group = parser.add_argument_group("Visualization Parameters")
    viz_group.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to visualize"
    )
    
    viz_group.add_argument(
        "--record",
        action="store_true",
        help="Record video of visualization"
    )
    
    return parser.parse_args()

def run_training(args):
    """Run the training process."""
    logger.info("Starting training process")
    
    from train_military_model import run_training
    
    success, model_path = run_training(
        config_path=args.config,
        time_steps=args.timesteps if args.timesteps > 0 else None,
        eval_freq=args.eval_freq if args.eval_freq > 0 else None,
        checkpoint_freq=args.checkpoint_freq if args.checkpoint_freq > 0 else None,
        seed=args.seed,
        device=args.device
    )
    
    if success:
        logger.info(f"Training completed successfully. Model saved at: {model_path}")
        return model_path
    else:
        logger.error("Training failed")
        return None

def run_analysis(model_path, args):
    """Run model analysis."""
    logger.info(f"Analyzing model: {model_path}")
    
    from train_military_model import analyze_model
    
    results = analyze_model(model_path)
    
    # Log results summary
    logger.info("=== Analysis Results ===")
    for key, value in results.items():
        logger.info(f"{key}: {value}")
    
    return results

def run_visualization(model_path, args):
    """Run model visualization."""
    logger.info(f"Visualizing model: {model_path}")
    
    from visualize_combat import CombatVisualizer
    
    # Create and run visualizer
    visualizer = CombatVisualizer(
        model_path=model_path,
        config_path=args.config,
        record_video=args.record
    )
    
    visualizer.run_multiple_episodes(num_episodes=args.episodes)
    
    logger.info("Visualization complete")

def main():
    """Main function to run the military training pipeline."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine model path
    model_path = args.model_path
    
    # Run selected components
    if args.train or args.full_pipeline:
        model_path = run_training(args)
        if not model_path:
            logger.error("Cannot proceed without a trained model")
            return
    
    if (args.analyze or args.full_pipeline) and model_path:
        run_analysis(model_path, args)
    
    if (args.visualize or args.full_pipeline) and model_path:
        run_visualization(model_path, args)
    
    # If no modes were selected, show help
    if not (args.train or args.analyze or args.visualize or args.full_pipeline):
        logger.error("No operation mode selected. Please specify --train, --analyze, --visualize, or --full_pipeline")
        parser = argparse.ArgumentParser()
        parse_args(["--help"])

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Military Drone Swarm Combat Training

This script provides a specialized training pipeline for military-grade drone swarm
simulations, with a focus on tactical combat, reconnaissance, and mission execution.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("military_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("military_training")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train military-grade drone swarm combat model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/configs/military_combat.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=0,
        help="Override number of timesteps in config. 0 means use config value."
    )
    
    parser.add_argument(
        "--eval_freq", 
        type=int, 
        default=0,
        help="Override evaluation frequency. 0 means use config value."
    )
    
    parser.add_argument(
        "--checkpoint_freq", 
        type=int, 
        default=0,
        help="Override checkpoint frequency. 0 means use config value."
    )
    
    parser.add_argument(
        "--analyze", 
        action="store_true",
        help="Run analysis on the trained model after training"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Visualize the trained model after training"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to use for training (cpu, cuda, auto)"
    )
    
    return parser.parse_args()

def run_training(config_path, time_steps=None, eval_freq=None, checkpoint_freq=None, 
                seed=42, device="cpu"):
    """Run the training process with enhanced military parameters."""
    logger.info(f"Starting military-grade training with config: {config_path}")
    
    # Import here to avoid importing tensorflow/torch before setting device
    from src.training.train import train_model, load_config
    
    # Load configuration
    config = load_config(config_path)
    
    # Override parameters if provided
    if time_steps:
        config["total_timesteps"] = time_steps
        logger.info(f"Overriding total_timesteps: {time_steps}")
    
    if eval_freq:
        config["eval_freq"] = eval_freq
        logger.info(f"Overriding eval_freq: {eval_freq}")
    
    if checkpoint_freq:
        config["checkpoint_freq"] = checkpoint_freq
        logger.info(f"Overriding checkpoint_freq: {checkpoint_freq}")
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else "0"
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    np.random.seed(seed)
    logger.info(f"Using seed: {seed}")
    
    # Print training parameters for record-keeping
    logger.info("=== Training Parameters ===")
    logger.info(f"Model: {config['run_name']}")
    logger.info(f"Environment: {config['environment']['type']}")
    logger.info(f"Agent: {config['agent']['type']}")
    logger.info(f"Network architecture: {config['agent']['net_arch']}")
    logger.info(f"Timesteps: {config['total_timesteps']}")
    logger.info(f"Num environments: {config['environment']['num_envs']}")
    logger.info(f"Batch size: {config['agent']['batch_size']}")
    logger.info("=========================")
    
    # Start training timer
    start_time = time.time()
    
    # Run training
    try:
        final_model_path = train_model(config, seed=seed)
        training_success = True
        logger.info(f"Training completed successfully. Model saved at: {final_model_path}")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        training_success = False
        final_model_path = None
    
    # Log training time
    training_time = time.time() - start_time
    logger.info(f"Total training time: {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    
    return training_success, final_model_path

def analyze_model(model_path):
    """Run a comprehensive analysis on the trained model."""
    logger.info(f"Analyzing model: {model_path}")
    
    # Import here to avoid importing tensorflow/torch before setting device
    import stable_baselines3
    from src.environments.drone_env import DroneSwarmCombatEnv
    from src.training.train import load_config
    
    # Create analysis directory
    analysis_dir = os.path.join(os.path.dirname(model_path), "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load the model
    model = stable_baselines3.PPO.load(model_path)
    
    # Create environment for evaluation
    config_path = "src/configs/military_combat.yaml"
    config = load_config(config_path)
    
    env_params = config["environment"]["params"].copy()
    env_params["render_mode"] = None
    env_params["headless"] = True
    
    env = DroneSwarmCombatEnv(**env_params)
    
    # Run model for multiple episodes and collect metrics
    num_episodes = 20
    episode_lengths = []
    episode_rewards = []
    mission_success_rate = 0
    enemy_elimination_rate = []
    friendly_survival_rate = []
    
    logger.info(f"Running {num_episodes} evaluation episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
        
        # Record metrics
        episode_lengths.append(step)
        episode_rewards.append(episode_reward)
        
        # Calculate mission success (all enemies destroyed)
        if "enemy_drones_remaining" in info and info["enemy_drones_remaining"] == 0:
            mission_success_rate += 1
        
        # Calculate enemy elimination rate
        if "enemy_drones_remaining" in info and "destroyed_enemies" in info:
            total_enemies = len(env.enemy_drone_ids)
            eliminated = info.get("destroyed_enemies", 0)
            enemy_elimination_rate.append(eliminated / total_enemies if total_enemies > 0 else 0)
        
        # Calculate friendly survival rate
        if "friendly_drones_remaining" in info:
            total_friendly = len(env.drone_ids)
            survived = info.get("friendly_drones_remaining", 0)
            friendly_survival_rate.append(survived / total_friendly if total_friendly > 0 else 0)
            
        logger.info(f"Episode {episode+1}/{num_episodes}: "
                   f"Reward={episode_reward:.2f}, "
                   f"Length={step}, "
                   f"Success={'Yes' if info.get('enemy_drones_remaining', 1) == 0 else 'No'}")
    
    # Calculate overall metrics
    mission_success_rate = mission_success_rate / num_episodes * 100
    avg_enemy_elimination = np.mean(enemy_elimination_rate) * 100 if enemy_elimination_rate else 0
    avg_friendly_survival = np.mean(friendly_survival_rate) * 100 if friendly_survival_rate else 0
    
    # Log summary statistics
    logger.info("=== Model Performance Summary ===")
    logger.info(f"Average Episode Reward: {np.mean(episode_rewards):.2f}")
    logger.info(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
    logger.info(f"Mission Success Rate: {mission_success_rate:.2f}%")
    logger.info(f"Average Enemy Elimination Rate: {avg_enemy_elimination:.2f}%")
    logger.info(f"Average Friendly Survival Rate: {avg_friendly_survival:.2f}%")
    logger.info("===============================")
    
    # Generate plots
    plt.figure(figsize=(12, 8))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot episode lengths
    plt.subplot(2, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot enemy elimination rate
    plt.subplot(2, 2, 3)
    plt.plot(enemy_elimination_rate)
    plt.title('Enemy Elimination Rate')
    plt.xlabel('Episode')
    plt.ylabel('Elimination %')
    plt.ylim(0, 1.1)
    
    # Plot friendly survival rate
    plt.subplot(2, 2, 4)
    plt.plot(friendly_survival_rate)
    plt.title('Friendly Survival Rate')
    plt.xlabel('Episode')
    plt.ylabel('Survival %')
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "performance_metrics.png"))
    
    return {
        "avg_reward": np.mean(episode_rewards),
        "mission_success_rate": mission_success_rate,
        "avg_enemy_elimination": avg_enemy_elimination,
        "avg_friendly_survival": avg_friendly_survival
    }

def visualize_model(model_path, episodes=3):
    """Visualize the trained model with tactical overlay."""
    logger.info(f"Visualizing model: {model_path}")
    
    # Build command to run visualization
    cmd = f"python visualize_combat.py --model {model_path} --episodes {episodes}"
    logger.info(f"Running visualization command: {cmd}")
    
    # Execute visualization script
    os.system(cmd)

def main():
    """Main function to run training and analysis."""
    args = parse_args()
    
    # Ensure src directory is in Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Run training
    training_success, model_path = run_training(
        args.config,
        args.timesteps if args.timesteps > 0 else None,
        args.eval_freq if args.eval_freq > 0 else None,
        args.checkpoint_freq if args.checkpoint_freq > 0 else None,
        args.seed,
        args.device
    )
    
    # If training was successful and analysis requested
    if training_success and model_path and args.analyze:
        # Small delay to ensure all files are written
        time.sleep(1)
        analysis_results = analyze_model(model_path)
        
        # Log analysis results
        logger.info("=== Analysis Results ===")
        for key, value in analysis_results.items():
            logger.info(f"{key}: {value}")
    
    # If visualization requested
    if training_success and model_path and args.visualize:
        # Small delay to ensure all files are written
        time.sleep(1)
        visualize_model(model_path)
    
    logger.info("Military training workflow completed")

if __name__ == "__main__":
    main() 
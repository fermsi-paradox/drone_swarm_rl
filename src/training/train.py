#!/usr/bin/env python3
"""
Core training module for drone swarm reinforcement learning.

This module provides the core functionality for training drone swarm models,
loading configurations, and setting up the training environment.
"""

import os
import time
import yaml
import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env

# Configure logging
logger = logging.getLogger("drone_training")

def load_config(config_path):
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Configuration parameters
    """
    logger.info(f"Loading configuration from {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def make_env(config, rank=0, seed=0):
    """
    Factory function for creating an environment with proper configuration.
    
    Args:
        config: Environment configuration
        rank: Process rank for seeding
        seed: Random seed base
        
    Returns:
        function: Environment creation function
    """
    env_config = config["environment"]
    env_type = env_config["type"]
    
    def _init():
        # Import environment class based on type
        if env_type == "DroneSwarmCombatEnv":
            from src.environments.drone_env import DroneSwarmCombatEnv
            env = DroneSwarmCombatEnv(**env_config["params"])
        else:
            # Fallback to gym environments
            env = gym.make(env_config["name"], **env_config.get("params", {}))
        
        # Set environment seed for reproducibility
        env.reset(seed=seed + rank)
        
        # Wrap environment with Monitor for logging
        log_dir = f"logs/{config['run_name']}/rank_{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        
        return env
    
    return _init

def create_agent(config, env, seed=0):
    """
    Create a reinforcement learning agent based on configuration.
    
    Args:
        config: Agent configuration
        env: Training environment
        seed: Random seed
        
    Returns:
        agent: Instantiated agent
    """
    agent_config = config["agent"]
    agent_type = agent_config["type"]
    
    # Extract agent parameters
    agent_kwargs = {
        "policy": agent_config.get("policy", "MlpPolicy"),
        "learning_rate": agent_config.get("learning_rate", 3e-4),
        "n_steps": agent_config.get("n_steps", 2048),
        "batch_size": agent_config.get("batch_size", 64),
        "n_epochs": agent_config.get("n_epochs", 10),
        "gamma": agent_config.get("gamma", 0.99),
        "gae_lambda": agent_config.get("gae_lambda", 0.95),
        "clip_range": agent_config.get("clip_range", 0.2),
        "clip_range_vf": agent_config.get("clip_range_vf", None),
        "normalize_advantage": agent_config.get("normalize_advantage", True),
        "ent_coef": agent_config.get("ent_coef", 0.0),
        "vf_coef": agent_config.get("vf_coef", 0.5),
        "max_grad_norm": agent_config.get("max_grad_norm", 0.5),
        "use_sde": agent_config.get("use_sde", False),
        "sde_sample_freq": agent_config.get("sde_sample_freq", -1),
        "target_kl": agent_config.get("target_kl", None),
        "verbose": 1,
        "seed": seed
    }
    
    # Handle net_arch if specified
    if "net_arch" in agent_config:
        agent_kwargs["policy_kwargs"] = {"net_arch": agent_config["net_arch"]}
    
    # Create the agent based on type
    if agent_type == "PPO":
        return stable_baselines3.PPO(env=env, **agent_kwargs)
    elif agent_type == "A2C":
        return stable_baselines3.A2C(env=env, **agent_kwargs)
    elif agent_type == "SAC":
        return stable_baselines3.SAC(env=env, **agent_kwargs)
    elif agent_type == "TD3":
        return stable_baselines3.TD3(env=env, **agent_kwargs)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

def setup_callbacks(config, env=None):
    """
    Set up training callbacks based on configuration.
    
    Args:
        config: Training configuration
        env: Optional evaluation environment
        
    Returns:
        CallbackList: List of callbacks
    """
    callbacks = []
    
    # Create output directories
    run_name = config["run_name"]
    models_dir = f"models/{run_name}"
    os.makedirs(models_dir, exist_ok=True)
    
    # Checkpoint callback
    checkpoint_freq = config.get("checkpoint_freq", 100000)
    if checkpoint_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=models_dir,
            name_prefix=f"{run_name}_checkpoint",
            save_replay_buffer=config.get("save_buffer", False),
            save_vecnormalize=True,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_freq = config.get("eval_freq", 25000)
    if eval_freq > 0 and env is not None:
        # Create evaluation environment
        eval_env = env
        
        # Create evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=models_dir,
            log_path=f"logs/{run_name}/eval",
            eval_freq=eval_freq,
            deterministic=True,
            render=config.get("render_training", False),
            n_eval_episodes=config.get("eval_episodes", 5),
            verbose=1
        )
        callbacks.append(eval_callback)
    
    return CallbackList(callbacks)

def train_model(config, seed=0):
    """
    Train a model using the specified configuration.
    
    Args:
        config: Training configuration dictionary
        seed: Random seed
        
    Returns:
        str: Path to the trained model
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    
    # Extract parameters
    run_name = config["run_name"]
    total_timesteps = config["total_timesteps"]
    num_envs = config["environment"].get("num_envs", 1)
    
    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"Number of environments: {num_envs}")
    
    # Create output directories
    models_dir = f"models/{run_name}"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create vectorized environment
    if num_envs > 1:
        logger.info(f"Creating {num_envs} parallel environments")
        env = make_vec_env(
            make_env(config, seed=seed),
            n_envs=num_envs,
            seed=seed,
            vec_env_cls=SubprocVecEnv if num_envs > 1 else DummyVecEnv
        )
    else:
        env = DummyVecEnv([make_env(config, seed=seed)])
    
    # Apply observation normalization if configured
    if config["environment"].get("normalize", False):
        logger.info("Applying observation normalization")
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=config["environment"].get("normalize_reward", True),
            clip_obs=config["environment"].get("clip_obs", 10.0),
            clip_reward=config["environment"].get("clip_reward", 10.0),
            gamma=config["agent"].get("gamma", 0.99),
            epsilon=1e-8
        )
    
    # Create agent
    logger.info(f"Creating {config['agent']['type']} agent")
    agent = create_agent(config, env, seed=seed)
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    
    # Train the agent
    start_time = time.time()
    logger.info("Starting training")
    
    agent.learn(
        total_timesteps=total_timesteps, 
        callback=callbacks,
        progress_bar=True
    )
    
    # Log training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f"{models_dir}/{run_name}_final_{timestamp}.zip"
    agent.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save normalized environment stats if applicable
    if isinstance(env, VecNormalize):
        stats_path = f"{models_dir}/{run_name}_final_{timestamp}_env_stats.pkl"
        env.save(stats_path)
        logger.info(f"Environment stats saved to {stats_path}")
    
    return final_model_path

def evaluate_model(model_path, config, episodes=10, render=False, seed=0):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the trained model
        config: Environment configuration
        episodes: Number of evaluation episodes
        render: Whether to render the evaluation
        seed: Random seed
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Evaluating model: {model_path}")
    
    # Load the model
    model = stable_baselines3.PPO.load(model_path)
    
    # Create environment
    eval_env = make_env(config, seed=seed)()
    
    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if render:
                eval_env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        logger.info(f"Episode {episode+1}/{episodes}: "
                   f"Reward={episode_reward:.2f}, "
                   f"Length={episode_length}")
    
    # Calculate metrics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths)
    }
    
    # Log results
    logger.info("=== Evaluation Results ===")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.2f}")
    
    return metrics 
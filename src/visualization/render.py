Visualization script for rendering trained drone swarm agents.

This module provides functionality to render and visualize trained drone swarm
agents in a realistic combat simulation environment.
"""

import argparse
import os
import time
from typing import Dict, Optional

import numpy as np
import pygame
import yaml
from pygame.locals import *
from stable_baselines3 import PPO, SAC

# Register custom environments
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.environments.drone_env import DroneSwarmEnv, DroneSwarmCombatEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize a trained drone swarm RL agent")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to the trained model file"
    )
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="src/configs/combat_scenario.yaml",
        help="Path to the scenario configuration file"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=5,
        help="Number of episodes to render"
    )
    parser.add_argument(
        "--record", 
        action="store_true",
        help="Record video of the visualization"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="videos",
        help="Directory to save recorded videos"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def create_environment(config: Dict, seed: int = 42):
    """Create the visualization environment.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
        
    Returns:
        Environment for visualization
    """
    # Extract environment parameters
    env_config = config["environment"]
    env_type = env_config.get("type", "combat")
    
    # Force rendering for visualization
    env_params = env_config.get("params", {}).copy()
    env_params["render_mode"] = "rgb_array"
    env_params["headless"] = False
    
    # Select environment class based on type
    if env_type == "combat":
        env = DroneSwarmCombatEnv(**env_params)
    else:
        env = DroneSwarmEnv(**env_params)
    
    return env


def load_model(model_path: str, env, config: Dict):
    """Load a trained model.
    
    Args:
        model_path: Path to the model file
        env: Environment to run the model in
        config: Configuration dictionary
        
    Returns:
        Loaded model
    """
    # Determine model type from config
    agent_type = config.get("agent", {}).get("type", "PPO")
    
    # Load the appropriate model type
    if agent_type == "SAC":
        model = SAC.load(model_path, env=env)
    else:  # Default to PPO
        model = PPO.load(model_path, env=env)
    
    return model


def setup_pygame(width: int = 1280, height: int = 720):
    """Set up Pygame for visualization.
    
    Args:
        width: Screen width
        height: Screen height
        
    Returns:
        Pygame screen
    """
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Drone Swarm Combat Simulation")
    
    return screen


def setup_recording(output_dir: str, width: int = 1280, height: int = 720):
    """Set up video recording.
    
    Args:
        output_dir: Directory to save videos
        width: Video width
        height: Video height
        
    Returns:
        Video writer
    """
    try:
        import cv2
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(output_dir, f"drone_swarm_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            video_path, 
            fourcc, 
            30.0, 
            (width, height)
        )
        
        print(f"Recording video to {video_path}")
        return video_writer
    except ImportError:
        print("OpenCV not found. Video recording disabled.")
        return None


def process_pybullet_frame(frame, width: int = 1280, height: int = 720):
    """Process a PyBullet frame for visualization.
    
    Args:
        frame: Raw frame from PyBullet
        width: Target width
        height: Target height
        
    Returns:
        Processed frame
    """
    # Resize if needed
    if frame.shape[0] != height or frame.shape[1] != width:
        import cv2
        frame = cv2.resize(frame, (width, height))
    
    # Convert to Pygame surface
    frame = np.transpose(frame, (1, 0, 2))
    frame = pygame.surfarray.make_surface(frame)
    
    return frame


def add_information_overlay(screen, episode: int, step: int, rewards: float, fps: float):
    """Add information overlay to the visualization.
    
    Args:
        screen: Pygame screen
        episode: Current episode
        step: Current step
        rewards: Current rewards
        fps: Current FPS
    """
    font = pygame.font.SysFont("Arial", 24)
    
    # Render text
    episode_text = font.render(f"Episode: {episode}", True, (255, 255, 255))
    step_text = font.render(f"Step: {step}", True, (255, 255, 255))
    reward_text = font.render(f"Reward: {rewards:.2f}", True, (255, 255, 255))
    fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
    
    # Add text to screen
    screen.blit(episode_text, (10, 10))
    screen.blit(step_text, (10, 40))
    screen.blit(reward_text, (10, 70))
    screen.blit(fps_text, (10, 100))


def visualize_agent(
    env, 
    model, 
    config: Dict, 
    episodes: int = 5, 
    record: bool = False, 
    output_dir: str = "videos"
):
    """Visualize a trained agent.
    
    Args:
        env: Environment to run the agent in
        model: Trained agent model
        config: Configuration dictionary
        episodes: Number of episodes to render
        record: Whether to record video
        output_dir: Directory to save recorded videos
    """
    # Set up visualization
    width, height = 1280, 720
    screen = setup_pygame(width, height)
    clock = pygame.time.Clock()
    
    # Set up recording if enabled
    video_writer = None
    if record:
        video_writer = setup_recording(output_dir, width, height)
    
    # Run visualization for specified number of episodes
    for episode in range(episodes):
        observation, info = env.reset()
        
        cumulative_reward = 0
        step = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Check for exit events
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    if video_writer:
                        video_writer.release()
                    pygame.quit()
                    return
            
            # Get action from model
            action, _ = model.predict(observation, deterministic=True)
            
            # Step the environment
            observation, reward, done, truncated, info = env.step(action)
            cumulative_reward += reward.mean() if hasattr(reward, "mean") else reward
            
            # Render the environment
            frame = env.render()
            
            # Convert frame to Pygame surface
            frame_surface = process_pybullet_frame(frame, width, height)
            
            # Display the frame
            screen.blit(frame_surface, (0, 0))
            
            # Add information overlay
            add_information_overlay(
                screen, 
                episode + 1, 
                step, 
                cumulative_reward, 
                clock.get_fps()
            )
            
            # Update the display
            pygame.display.flip()
            
            # Record frame if enabled
            if video_writer:
                # Convert Pygame surface to numpy array for OpenCV
                frame_array = pygame.surfarray.array3d(screen)
                frame_array = np.transpose(frame_array, (1, 0, 2))
                video_writer.write(frame_array)
            
            # Control frame rate
            clock.tick(30)
            
            step += 1
        
        print(f"Episode {episode + 1} finished with reward {cumulative_reward:.2f}")
    
    # Clean up
    if video_writer:
        video_writer.release()
    pygame.quit()


def main():
    """Main visualization function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.scenario)
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    # Create environment for visualization
    env = create_environment(config, args.seed)
    
    # Load trained model
    model = load_model(args.model, env, config)
    
    # Run visualization
    visualize_agent(
        env=env,
        model=model,
        config=config,
        episodes=args.episodes,
        record=args.record,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
 
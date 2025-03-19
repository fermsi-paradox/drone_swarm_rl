#!/usr/bin/env python3
"""
Visualization script for the trained drone swarm agent.

This script directly implements visualization functionality for watching the
performance of the trained reinforcement learning agent in the drone swarm environment.
"""

import os
import sys
import numpy as np

# Add the project root to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
from src.environments.drone_env import DroneSwarmEnv, DroneSwarmCombatEnv
from stable_baselines3 import PPO, SAC
import yaml
import pygame
from pygame.locals import *

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def create_environment(config, seed=42):
    """Create the visualization environment."""
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

def load_model(model_path, env, config):
    """Load a trained model."""
    # Determine model type from config
    agent_type = config.get("agent", {}).get("type", "PPO")
    
    # Load the appropriate model type
    if agent_type == "SAC":
        model = SAC.load(model_path, env=env)
    else:  # Default to PPO
        model = PPO.load(model_path, env=env)
    
    return model

def setup_pygame(width=1280, height=720):
    """Set up Pygame for visualization."""
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Drone Swarm Combat Simulation")
    
    return screen

def process_pybullet_frame(frame, width=1280, height=720):
    """Process a PyBullet frame for visualization."""
    # Resize if needed
    if frame.shape[0] != height or frame.shape[1] != width:
        import cv2
        frame = cv2.resize(frame, (width, height))
    
    # Convert to Pygame surface
    frame = np.transpose(frame, (1, 0, 2))
    frame = pygame.surfarray.make_surface(frame)
    
    return frame

def add_information_overlay(screen, episode, step, rewards, fps, env=None, info=None):
    """Add information overlay to the visualization."""
    font = pygame.font.SysFont("Arial", 24)
    small_font = pygame.font.SysFont("Arial", 18)
    title_font = pygame.font.SysFont("Arial", 28, bold=True)
    
    # Render text for basic info
    episode_text = font.render(f"Episode: {episode}", True, (255, 255, 255))
    step_text = font.render(f"Step: {step}", True, (255, 255, 255))
    reward_text = font.render(f"Reward: {rewards:.2f}", True, (255, 255, 255))
    fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
    
    # Add text to screen
    screen.blit(episode_text, (10, 10))
    screen.blit(step_text, (10, 40))
    screen.blit(reward_text, (10, 70))
    screen.blit(fps_text, (10, 100))
    
    # Add mission status title
    mission_title = title_font.render("MISSION STATUS", True, (220, 220, 220))
    screen.blit(mission_title, (10, 140))
    
    # Add combat info if available
    if env is not None and hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'drone_ids'):
        env_unwrapped = env.unwrapped
        
        # If we have the info dict and it contains relevant info
        if info is not None:
            y_offset = 180
            
            if "friendly_drones_remaining" in info and "enemy_drones_remaining" in info:
                friendly_count = info["friendly_drones_remaining"]
                enemy_count = info["enemy_drones_remaining"]
                total_friendly = len(env_unwrapped.drone_ids)
                total_enemy = len(env_unwrapped.enemy_drone_ids)
                
                # Status message based on remaining drones
                status_color = (220, 220, 100)  # Default yellow
                status_msg = "MISSION ACTIVE"
                
                if enemy_count == 0:
                    status_msg = "MISSION ACCOMPLISHED"
                    status_color = (100, 255, 100)  # Green
                elif friendly_count == 0:
                    status_msg = "MISSION FAILED"
                    status_color = (255, 100, 100)  # Red
                
                status_text = font.render(status_msg, True, status_color)
                screen.blit(status_text, (screen.get_width() - 300, 20))
                
                # Format as fractions with percentage
                friendly_percent = (friendly_count / total_friendly) * 100
                enemy_percent = (enemy_count / total_enemy) * 100
                
                friendly_text = font.render(f"Friendly: {friendly_count}/{total_friendly} ({friendly_percent:.0f}%)", 
                                           True, (100, 255, 100))
                enemy_text = font.render(f"Enemy: {enemy_count}/{total_enemy} ({enemy_percent:.0f}%)", 
                                         True, (255, 100, 100))
                
                screen.blit(friendly_text, (10, y_offset))
                screen.blit(enemy_text, (10, y_offset + 30))
                y_offset += 70
            
            # Display drone health if available
            if "drone_health" in info:
                health_title = font.render("Drone Health:", True, (220, 220, 220))
                screen.blit(health_title, (10, y_offset))
                y_offset += 30
                
                for i, health in enumerate(info["drone_health"]):
                    # Set color based on health
                    if health > 75:
                        health_color = (100, 255, 100)  # Green for good health
                    elif health > 40:
                        health_color = (255, 255, 100)  # Yellow for medium health
                    else:
                        health_color = (255, 100, 100)  # Red for low health
                    
                    # Draw health bar
                    bar_width = 150
                    bar_height = 15
                    outline_rect = pygame.Rect(130, y_offset, bar_width, bar_height)
                    health_rect = pygame.Rect(130, y_offset, int(bar_width * health/100), bar_height)
                    
                    pygame.draw.rect(screen, (70, 70, 70), outline_rect)  # Background
                    pygame.draw.rect(screen, health_color, health_rect)   # Health level
                    pygame.draw.rect(screen, (200, 200, 200), outline_rect, 1)  # Border
                    
                    health_text = small_font.render(f"Drone {i+1}: {health:.0f}%", True, health_color)
                    screen.blit(health_text, (10, y_offset - 2))
                    y_offset += 25
                
                y_offset += 20
            
            # Display detection statistics if we can calculate them
            if hasattr(env_unwrapped, "enemy_drone_ids") and hasattr(env_unwrapped, "destroyed_enemies"):
                detected_title = font.render("Enemy Detection:", True, (220, 220, 220))
                screen.blit(detected_title, (10, y_offset))
                y_offset += 30
                
                # Try to estimate detected enemies from our reward function logic
                if hasattr(env_unwrapped, "step_count") and env_unwrapped.step_count > 0:
                    remaining_enemies = len(env_unwrapped.enemy_drone_ids) - len(env_unwrapped.destroyed_enemies)
                    destroyed_enemies = len(env_unwrapped.destroyed_enemies)
                    
                    # Show destroyed enemies
                    eliminated_text = small_font.render(
                        f"Eliminated: {destroyed_enemies}/{len(env_unwrapped.enemy_drone_ids)}", 
                        True, (255, 150, 100))
                    screen.blit(eliminated_text, (10, y_offset))
                    y_offset += 25
                    
                    # Try to render detected enemies if we can access the detection info
                    # This is a rough approximation since the actual calculation happens in _compute_reward
                    if "drone_rewards" in info and len(info["drone_rewards"]) > 0:
                        detected_text = small_font.render(
                            "Detection status displayed during training", 
                            True, (200, 200, 200))
                        screen.blit(detected_text, (10, y_offset))
    
    # Add controls hint at the bottom
    controls_text = small_font.render("Press ESC to exit", True, (200, 200, 200))
    screen.blit(controls_text, (10, screen.get_height() - 30))

    # Add legend for rewards
    legend_title = small_font.render("Reward Components:", True, (220, 220, 220))
    screen.blit(legend_title, (screen.get_width() - 240, screen.get_height() - 150))
    
    reward_items = [
        ("Enemy elimination", (255, 150, 100)),
        ("Detection without damage", (100, 255, 255)),
        ("Maintaining formation", (200, 200, 255)),
        ("Avoiding obstacles", (200, 255, 200)),
        ("Maintaining health", (255, 255, 100))
    ]
    
    for i, (text, color) in enumerate(reward_items):
        item_text = small_font.render(text, True, color)
        screen.blit(item_text, (screen.get_width() - 230, screen.get_height() - 130 + i*20))

def visualize_agent(env, model, config, episodes=3):
    """Visualize a trained agent."""
    # Set up visualization
    width, height = 1280, 720
    screen = setup_pygame(width, height)
    clock = pygame.time.Clock()
    
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
                clock.get_fps(),
                env,  # Pass the environment to access its internal state
                info   # Pass the info dict from step
            )
            
            # Update the display
            pygame.display.flip()
            
            # Control frame rate
            clock.tick(30)
            
            step += 1
        
        print(f"Episode {episode + 1} finished with reward {cumulative_reward:.2f}")
    
    # Clean up
    pygame.quit()

def main():
    """Main visualization function."""
    # Configuration
    model_path = "src/models/drone_swarm_ppo_20250314_151357/best_model/best_model.zip"
    config_path = "src/configs/default.yaml"  # Using default.yaml to match training configuration
    seed = 42
    episodes = 3
    
    # Load configuration
    config = load_config(config_path)
    
    # Override headless parameter to enable visualization
    if "environment" in config and "params" in config["environment"]:
        config["environment"]["params"]["headless"] = False
        
        # Enable all enhanced features
        config["environment"]["params"]["use_obstacles"] = True
        config["environment"]["params"]["num_obstacles"] = 10
        config["environment"]["params"]["enable_firing"] = True
        config["environment"]["params"]["enable_destruction"] = True
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create environment for visualization
    env = create_environment(config, seed)
    
    # Load trained model
    model = load_model(model_path, env, config)
    
    # Run visualization
    visualize_agent(env=env, model=model, config=config, episodes=episodes)

if __name__ == "__main__":
    main() 
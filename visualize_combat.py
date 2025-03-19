#!/usr/bin/env python3
"""
Military Drone Swarm Combat Visualization

This script loads a trained model and visualizes the drone swarm combat simulation
with tactical overlays and combat statistics.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import gymnasium as gym
import stable_baselines3
from pathlib import Path
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("combat_visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("combat_visualization")

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.environments.drone_env import DroneSwarmCombatEnv
from src.training.train import load_config

class CombatVisualizer:
    """Class to visualize military drone swarm combat."""
    
    def __init__(self, model_path, config_path=None, record_video=False):
        """Initialize the visualizer with the trained model."""
        self.model_path = model_path
        self.config_path = config_path or "src/configs/military_combat.yaml"
        self.record_video = record_video
        self.output_dir = "output/visualization"
        self.frame_count = 0
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = stable_baselines3.PPO.load(model_path)
        
        # Load configuration
        self.config = load_config(self.config_path)
        
        # Setup environment
        env_params = self.config["environment"]["params"].copy()
        env_params["render_mode"] = "human"
        env_params["headless"] = False
        
        logger.info("Creating visualization environment")
        self.env = DroneSwarmCombatEnv(**env_params)
        
        # Initialize statistics
        self.stats = {
            "friendly_health": [],
            "enemy_health": [],
            "distances": [],
            "hits_given": 0,
            "hits_taken": 0,
            "drones_lost": 0,
            "enemies_destroyed": 0,
            "mission_status": "In Progress"
        }
        
        # Initialize Pygame and OpenGL for visualization
        self._init_visualization()
    
    def _init_visualization(self):
        """Initialize the visualization components."""
        logger.info("Initializing visualization")
        
        # Initialize Pygame
        pygame.init()
        display = (1280, 720)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Military Drone Swarm Combat Simulation")
        
        # Setup OpenGL
        glClearColor(0.1, 0.1, 0.1, 1.0)  # Dark background
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Setup perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (display[0] / display[1]), 0.1, 500.0)
        
        # Initial camera position
        self.camera_distance = 100.0
        self.camera_angle_x = 45.0
        self.camera_angle_y = 30.0
        
        # Font for text overlay
        self.font = pygame.font.SysFont('Arial', 18)
        
        # Colors
        self.colors = {
            "friendly": (0.0, 0.5, 1.0),  # Blue
            "enemy": (1.0, 0.2, 0.2),     # Red
            "target": (1.0, 0.8, 0.0),    # Yellow
            "warning": (1.0, 0.5, 0.0),   # Orange
            "hit": (1.0, 0.0, 0.0),       # Bright red
            "text": (1.0, 1.0, 1.0),      # White
            "background": (0.1, 0.1, 0.1) # Dark gray
        }
        
        # Initialize frame timestamp for FPS calculation
        self.last_frame_time = time.time()
        self.fps = 0
    
    def _update_camera(self):
        """Update the camera position based on user input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Mouse control for camera
            if event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                dx, dy = event.rel
                self.camera_angle_x += dx * 0.5
                self.camera_angle_y = max(min(self.camera_angle_y + dy * 0.5, 89), -89)
        
        # Keyboard controls
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.camera_angle_x -= 2
        if keys[pygame.K_RIGHT]:
            self.camera_angle_x += 2
        if keys[pygame.K_UP]:
            self.camera_angle_y = max(self.camera_angle_y - 2, -89)
        if keys[pygame.K_DOWN]:
            self.camera_angle_y = min(self.camera_angle_y + 2, 89)
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            self.camera_distance = min(self.camera_distance + 2, 200)
        if keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]:
            self.camera_distance = max(self.camera_distance - 2, 10)
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()
            
        # Update the camera position
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Convert angles to radians
        angle_x_rad = np.radians(self.camera_angle_x)
        angle_y_rad = np.radians(self.camera_angle_y)
        
        # Calculate camera position
        cam_x = self.camera_distance * np.cos(angle_y_rad) * np.sin(angle_x_rad)
        cam_y = self.camera_distance * np.sin(angle_y_rad)
        cam_z = self.camera_distance * np.cos(angle_y_rad) * np.cos(angle_x_rad)
        
        # Set the camera position and target
        env_center = np.array(self.env.env_size) / 2
        gluLookAt(
            cam_x + env_center[0], cam_y + env_center[1], cam_z + env_center[2],  # Camera position
            env_center[0], env_center[1], env_center[2],  # Target (center of environment)
            0, 1, 0  # Up vector
        )
    
    def _draw_drone(self, position, orientation, color, health=100, radius=1.0, is_enemy=False):
        """Draw a drone at the specified position with the given orientation."""
        x, y, z = position
        
        # Push the current matrix to the stack
        glPushMatrix()
        
        # Move to the drone's position
        glTranslatef(x, y, z)
        
        # Draw the drone body (a sphere)
        glColor4f(color[0], color[1], color[2], 0.8)
        quad = gluNewQuadric()
        gluSphere(quad, radius, 16, 16)
        
        # Draw orientation indicator (a small cylinder pointing in drone's direction)
        if orientation is not None:
            glColor4f(color[0], color[1], color[2], 1.0)
            heading = np.arctan2(orientation[0], orientation[2])
            pitch = np.arcsin(orientation[1])
            
            glRotatef(np.degrees(heading), 0, 1, 0)
            glRotatef(-np.degrees(pitch), 1, 0, 0)
            
            cylinder = gluNewQuadric()
            gluCylinder(cylinder, 0.2, 0.0, 2.0, 8, 1)
        
        # Draw health bar above the drone
        self._draw_health_bar(health, is_enemy)
        
        # Pop the matrix off the stack
        glPopMatrix()
    
    def _draw_health_bar(self, health, is_enemy):
        """Draw a health bar above the drone."""
        max_health = 100  # Assuming max health is 100
        health_ratio = max(0, min(health / max_health, 1.0))
        
        # Health bar position (above the drone)
        bar_y = 2.0
        bar_width = 2.0
        bar_height = 0.3
        
        glPushMatrix()
        glTranslatef(-bar_width/2, bar_y, 0)
        
        # Health bar background (grey)
        glColor4f(0.3, 0.3, 0.3, 0.8)
        glBegin(GL_QUADS)
        glVertex3f(0, 0, 0)
        glVertex3f(bar_width, 0, 0)
        glVertex3f(bar_width, bar_height, 0)
        glVertex3f(0, bar_height, 0)
        glEnd()
        
        # Health bar fill (color based on health)
        if is_enemy:
            r, g, b = 1.0, 0.2, 0.2  # Red for enemies
        else:
            r, g, b = 0.2, 1.0, 0.2  # Green for friendly
        
        # Adjust color based on health (yellow when low)
        if health_ratio < 0.3:
            g = 0.6
            r = 1.0
        
        glColor4f(r, g, b, 0.8)
        glBegin(GL_QUADS)
        glVertex3f(0, 0, 0)
        glVertex3f(bar_width * health_ratio, 0, 0)
        glVertex3f(bar_width * health_ratio, bar_height, 0)
        glVertex3f(0, bar_height, 0)
        glEnd()
        
        glPopMatrix()
    
    def _draw_environment(self):
        """Draw the environment boundaries and obstacles."""
        env_size = self.env.env_size
        
        # Draw environment boundaries
        glColor4f(0.5, 0.5, 0.5, 0.2)
        glBegin(GL_LINES)
        
        # Bottom face
        glVertex3f(0, 0, 0)
        glVertex3f(env_size[0], 0, 0)
        
        glVertex3f(env_size[0], 0, 0)
        glVertex3f(env_size[0], 0, env_size[2])
        
        glVertex3f(env_size[0], 0, env_size[2])
        glVertex3f(0, 0, env_size[2])
        
        glVertex3f(0, 0, env_size[2])
        glVertex3f(0, 0, 0)
        
        # Top face
        glVertex3f(0, env_size[1], 0)
        glVertex3f(env_size[0], env_size[1], 0)
        
        glVertex3f(env_size[0], env_size[1], 0)
        glVertex3f(env_size[0], env_size[1], env_size[2])
        
        glVertex3f(env_size[0], env_size[1], env_size[2])
        glVertex3f(0, env_size[1], env_size[2])
        
        glVertex3f(0, env_size[1], env_size[2])
        glVertex3f(0, env_size[1], 0)
        
        # Connecting edges
        glVertex3f(0, 0, 0)
        glVertex3f(0, env_size[1], 0)
        
        glVertex3f(env_size[0], 0, 0)
        glVertex3f(env_size[0], env_size[1], 0)
        
        glVertex3f(env_size[0], 0, env_size[2])
        glVertex3f(env_size[0], env_size[1], env_size[2])
        
        glVertex3f(0, 0, env_size[2])
        glVertex3f(0, env_size[1], env_size[2])
        
        glEnd()
        
        # Draw ground grid
        glColor4f(0.3, 0.3, 0.3, 0.2)
        glBegin(GL_LINES)
        grid_size = 10
        for i in range(0, int(env_size[0])+1, grid_size):
            glVertex3f(i, 0, 0)
            glVertex3f(i, 0, env_size[2])
        
        for i in range(0, int(env_size[2])+1, grid_size):
            glVertex3f(0, 0, i)
            glVertex3f(env_size[0], 0, i)
        glEnd()
        
        # Draw obstacles if they exist
        if hasattr(self.env, 'obstacles'):
            glColor4f(0.6, 0.6, 0.6, 0.7)
            for obstacle in self.env.obstacles:
                position = obstacle['position']
                radius = obstacle.get('radius', 2.0)
                
                glPushMatrix()
                glTranslatef(position[0], position[1], position[2])
                quad = gluNewQuadric()
                gluSphere(quad, radius, 16, 16)
                glPopMatrix()
    
    def _draw_weapons_fire(self, origin, target):
        """Draw weapons fire between origin and target."""
        glColor4f(1.0, 0.8, 0.0, 0.8)  # Bright yellow
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(origin[0], origin[1], origin[2])
        glVertex3f(target[0], target[1], target[2])
        glEnd()
        glLineWidth(1.0)
        
        # Draw impact flash at target
        glPushMatrix()
        glTranslatef(target[0], target[1], target[2])
        glColor4f(1.0, 0.5, 0.0, 0.7)  # Orange
        quad = gluNewQuadric()
        gluSphere(quad, 0.8, 8, 8)
        glPopMatrix()
    
    def _draw_sensor_range(self, position, orientation, range_val, fov, is_enemy=False):
        """Draw sensor cone visualization."""
        # Only show sensor range for some drones to avoid visual clutter
        if np.random.random() > 0.3:  # 30% chance to show sensor range
            return
            
        x, y, z = position
        
        # Set color based on drone type
        if is_enemy:
            glColor4f(1.0, 0.2, 0.2, 0.1)  # Red with low alpha
        else:
            glColor4f(0.0, 0.5, 1.0, 0.1)  # Blue with low alpha
        
        # Push matrix
        glPushMatrix()
        glTranslatef(x, y, z)
        
        # Calculate cone orientation
        if orientation is not None:
            heading = np.arctan2(orientation[0], orientation[2])
            pitch = np.arcsin(orientation[1])
            
            glRotatef(np.degrees(heading), 0, 1, 0)
            glRotatef(-np.degrees(pitch), 1, 0, 0)
        
        # Draw sensor cone
        quad = gluNewQuadric()
        # Convert FOV from degrees to a fraction of a circle
        slice_angle = fov / 360.0 * 2 * np.pi
        gluPartialDisk(quad, 0, range_val, 32, 1, 0, np.degrees(slice_angle))
        
        glPopMatrix()
    
    def _draw_tactical_info(self):
        """Draw tactical information overlay."""
        # Create a Pygame surface with transparent background for text overlay
        text_surface = pygame.Surface((640, 720), pygame.SRCALPHA)
        
        # Draw tactical information
        y = 10
        line_height = 24
        
        # Mission status
        pygame.draw.rect(text_surface, (0, 0, 0, 180), (10, y, 280, 35))
        status_text = f"MISSION STATUS: {self.stats['mission_status']}"
        text = self.font.render(status_text, True, (255, 255, 255))
        text_surface.blit(text, (20, y + 5))
        y += 45
        
        # Combat statistics box
        pygame.draw.rect(text_surface, (0, 0, 0, 180), (10, y, 280, 150))
        
        # Draw heading
        text = self.font.render("COMBAT STATISTICS", True, (255, 255, 255))
        text_surface.blit(text, (20, y + 5))
        y += 30
        
        # Draw stats
        stat_items = [
            f"Friendly Drones: {self.env.num_drones - self.stats['drones_lost']}/{self.env.num_drones}",
            f"Enemy Drones: {self.env.num_enemy_drones - self.stats['enemies_destroyed']}/{self.env.num_enemy_drones}",
            f"Hits Given: {self.stats['hits_given']}",
            f"Hits Taken: {self.stats['hits_taken']}",
            f"FPS: {self.fps:.1f}"
        ]
        
        for stat in stat_items:
            text = self.font.render(stat, True, (255, 255, 255))
            text_surface.blit(text, (20, y + 5))
            y += line_height
        
        # Draw swarm health summary at bottom right
        y = 560
        pygame.draw.rect(text_surface, (0, 0, 0, 180), (10, y, 280, 150))
        
        # Title
        text = self.font.render("SWARM HEALTH", True, (255, 255, 255))
        text_surface.blit(text, (20, y + 5))
        y += 30
        
        # Average health
        if self.stats["friendly_health"]:
            avg_health = sum(self.stats["friendly_health"]) / len(self.stats["friendly_health"])
            health_text = f"Average Health: {avg_health:.1f}%"
            text = self.font.render(health_text, True, (255, 255, 255))
            text_surface.blit(text, (20, y + 5))
        
        # Convert the Pygame surface to an OpenGL texture
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        glEnable(GL_TEXTURE_2D)
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 640, 720, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        # Draw the texture as a 2D overlay
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1280, 720, 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glEnable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(640, 0)
        glTexCoord2f(1, 1); glVertex2f(640, 720)
        glTexCoord2f(0, 1); glVertex2f(0, 720)
        glEnd()
        
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        # Delete the texture to avoid memory leaks
        glDeleteTextures(1, [texture_id])
    
    def _update_stats(self, obs, info):
        """Update the visualization statistics."""
        # Update health statistics
        self.stats["friendly_health"] = []
        self.stats["enemy_health"] = []
        
        for i in range(self.env.num_drones):
            if hasattr(self.env, 'drone_health') and i < len(self.env.drone_health):
                self.stats["friendly_health"].append(self.env.drone_health[i])
        
        for i in range(self.env.num_enemy_drones):
            if hasattr(self.env, 'enemy_drone_health') and i < len(self.env.enemy_drone_health):
                self.stats["enemy_health"].append(self.env.enemy_drone_health[i])
        
        # Update hits given/taken and drones destroyed
        if "destroyed_enemies" in info:
            self.stats["enemies_destroyed"] = info["destroyed_enemies"]
        
        if "drones_lost" in info:
            self.stats["drones_lost"] = info["drones_lost"]
        
        if "hits_given" in info:
            self.stats["hits_given"] = info["hits_given"]
        
        if "hits_taken" in info:
            self.stats["hits_taken"] = info["hits_taken"]
        
        # Update mission status
        if self.env.num_enemy_drones - self.stats["enemies_destroyed"] <= 0:
            self.stats["mission_status"] = "SUCCESS - All enemies eliminated"
        elif self.env.num_drones - self.stats["drones_lost"] <= 0:
            self.stats["mission_status"] = "FAILURE - All drones lost"
    
    def _calculate_fps(self):
        """Calculate and update the frames per second."""
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        self.fps = 1.0 / time_diff if time_diff > 0 else 0
        self.last_frame_time = current_time
    
    def _capture_frame(self):
        """Capture the current frame if recording is enabled."""
        if self.record_video:
            # Create frame directory if it doesn't exist
            frames_dir = os.path.join(self.output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Read the framebuffer and save as an image
            x, y, width, height = glGetIntegerv(GL_VIEWPORT)
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            
            # Convert to pygame surface
            surface = pygame.image.fromstring(data, (width, height), "RGB", True)
            
            # Save the image
            frame_path = os.path.join(frames_dir, f"frame_{self.frame_count:05d}.png")
            pygame.image.save(surface, frame_path)
            
            self.frame_count += 1
    
    def run_episode(self):
        """Run a complete episode with visualization."""
        logger.info("Starting visualization episode")
        
        # Reset the environment
        obs, _ = self.env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        combat_events = []  # Track combat events for visualization
        
        # Main simulation loop
        while not (done or truncated):
            # Clear buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Update camera based on user input
            self._update_camera()
            
            # Draw environment
            self._draw_environment()
            
            # Get policy action
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Apply action and get new observation
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            step += 1
            
            # Update statistics
            self._update_stats(obs, info)
            
            # Track combat events (if they occurred)
            if "combat_events" in info:
                for event in info["combat_events"]:
                    event["ttl"] = 10  # Time to live (frames)
                    combat_events.append(event)
            
            # Draw drones and tactical information
            if hasattr(self.env, 'drone_positions'):
                # Draw friendly drones
                for i, pos in enumerate(self.env.drone_positions):
                    if i < len(self.env.drone_orientations):
                        orientation = self.env.drone_orientations[i]
                    else:
                        orientation = np.array([0, 0, 1])  # Default forward orientation
                    
                    health = 100
                    if hasattr(self.env, 'drone_health') and i < len(self.env.drone_health):
                        health = self.env.drone_health[i]
                    
                    self._draw_drone(pos, orientation, self.colors["friendly"], health, radius=1.0, is_enemy=False)
                    
                    # Draw sensor range for some drones
                    sensor_range = self.config["environment"]["params"]["sensor_range"]
                    sensor_fov = self.config["environment"]["params"]["sensor_fov"]
                    self._draw_sensor_range(pos, orientation, sensor_range, sensor_fov, is_enemy=False)
            
            if hasattr(self.env, 'enemy_drone_positions'):
                # Draw enemy drones
                for i, pos in enumerate(self.env.enemy_drone_positions):
                    if hasattr(self.env, 'enemy_drone_orientations') and i < len(self.env.enemy_drone_orientations):
                        orientation = self.env.enemy_drone_orientations[i]
                    else:
                        orientation = np.array([0, 0, 1])  # Default forward orientation
                    
                    health = 100
                    if hasattr(self.env, 'enemy_drone_health') and i < len(self.env.enemy_drone_health):
                        health = self.env.enemy_drone_health[i]
                    
                    self._draw_drone(pos, orientation, self.colors["enemy"], health, radius=1.0, is_enemy=True)
                    
                    # Draw sensor range for some drones
                    sensor_range = self.config["environment"]["params"]["sensor_range"]
                    sensor_fov = self.config["environment"]["params"]["sensor_fov"]
                    self._draw_sensor_range(pos, orientation, sensor_range, sensor_fov, is_enemy=True)
            
            # Draw combat events (weapons fire, hits, etc.)
            for event in combat_events[:]:
                if event["type"] == "weapon_fire" and "origin" in event and "target" in event:
                    self._draw_weapons_fire(event["origin"], event["target"])
                
                event["ttl"] -= 1
                if event["ttl"] <= 0:
                    combat_events.remove(event)
            
            # Draw tactical info overlay
            self._draw_tactical_info()
            
            # Calculate FPS
            self._calculate_fps()
            
            # Capture frame if recording
            self._capture_frame()
            
            # Update display
            pygame.display.flip()
            
            # Maintain frame rate
            pygame.time.wait(25)  # Approximately 40 FPS
            
            # Log step info every 50 steps
            if step % 50 == 0:
                logger.info(f"Step {step}: Reward={reward:.2f}, "
                            f"Friendly drones={self.env.num_drones - self.stats['drones_lost']}, "
                            f"Enemy drones={self.env.num_enemy_drones - self.stats['enemies_destroyed']}")
        
        # Episode summary
        logger.info(f"Episode complete: Steps={step}, Total reward={total_reward:.2f}")
        logger.info(f"Final status: {self.stats['mission_status']}")
        logger.info(f"Friendly drones remaining: {self.env.num_drones - self.stats['drones_lost']}")
        logger.info(f"Enemy drones destroyed: {self.stats['enemies_destroyed']}")
        
        # Create video from frames if recording
        if self.record_video:
            self._create_video()
    
    def _create_video(self):
        """Create a video from captured frames using ffmpeg."""
        if self.frame_count == 0:
            logger.warning("No frames captured, skipping video creation")
            return
        
        logger.info("Creating video from captured frames")
        
        frames_dir = os.path.join(self.output_dir, "frames")
        video_path = os.path.join(self.output_dir, f"combat_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        
        try:
            import subprocess
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate", "30",
                "-i", os.path.join(frames_dir, "frame_%05d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                video_path
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"Video saved to: {video_path}")
        except Exception as e:
            logger.error(f"Failed to create video: {e}")
    
    def run_multiple_episodes(self, num_episodes=3):
        """Run multiple visualization episodes."""
        logger.info(f"Running {num_episodes} visualization episodes")
        
        for episode in range(num_episodes):
            logger.info(f"Starting episode {episode+1}/{num_episodes}")
            self.run_episode()
            
            # Small delay between episodes
            time.sleep(1)
        
        # Close pygame
        pygame.quit()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize trained military drone swarm model")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to the trained model"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file. If not provided, will use default military config."
    )
    
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=3,
        help="Number of episodes to visualize"
    )
    
    parser.add_argument(
        "--record", 
        action="store_true",
        help="Record video of the visualization"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the visualization."""
    args = parse_args()
    
    # Create visualizer
    visualizer = CombatVisualizer(
        model_path=args.model,
        config_path=args.config,
        record_video=args.record
    )
    
    # Run visualization
    visualizer.run_multiple_episodes(num_episodes=args.episodes)

if __name__ == "__main__":
    main() 
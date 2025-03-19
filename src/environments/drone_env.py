"""
Drone Swarm Environment for Reinforcement Learning.

This module defines a custom environment for training drone swarm agents.
The environment supports multiple drones interacting in a 3D space with physics
simulation provided by PyBullet.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class DroneSwarmEnv(gym.Env):
    """
    A Gymnasium environment for drone swarm combat simulations.
    
    This environment allows for training multiple drone agents in a simulated
    3D space with realistic physics. It can be run headless for efficient training
    or with rendering for visualization.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], 
                "render_fps": 30}
    
    def __init__(
        self,
        num_drones: int = 5,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        headless: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize the drone swarm environment.
        
        Args:
            num_drones: Number of drones in the swarm
            max_steps: Maximum number of steps per episode
            render_mode: Rendering mode ("human", "rgb_array", or None)
            headless: Whether to run in headless mode (no visualization)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.num_drones = num_drones
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.headless = headless
        self.seed_val = seed
        
        # Initialize step counter
        self.step_count = 0
        
        # Define action and observation spaces
        # Each drone can control thrust and 3 rotation angles
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0, 
            shape=(num_drones, 4),
            dtype=np.float32
        )
        
        # Observations include position, orientation, velocity, and sensor data
        # Position (3), Orientation (3), Linear Velocity (3), Angular Velocity (3)
        # Plus sensor readings of nearest objects (10)
        # Total: 12 for drone state + 10 for sensors = 22
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_drones, 22),
            dtype=np.float32
        )
        
        # Initialize physics client
        self._init_simulation()
        
        # Create terrain and drones
        self._create_terrain()
        self.drone_ids = self._spawn_drones()
        
        # Enemy drones (if in combat scenario)
        self.enemy_drone_ids = []
        
    def _init_simulation(self):
        """Initialize the PyBullet physics simulation."""
        if self.headless:
            self.physics_client = p.connect(p.DIRECT)
        else:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1.0 / 240.0)
        
    def _create_terrain(self):
        """Create the terrain for the simulation."""
        self.terrain_id = p.loadURDF("plane.urdf")
        
        # Optional: Add obstacles and structures
        # self.obstacles = []
        # for i in range(10):
        #     pos = [np.random.uniform(-10, 10), np.random.uniform(-10, 10), 1]
        #     self.obstacles.append(p.loadURDF("cube.urdf", pos, globalScaling=0.5))
    
    def _spawn_drones(self) -> List[int]:
        """Spawn drones in the environment.
        
        Returns:
            List of drone IDs in PyBullet
        """
        # In a real implementation, you would load a URDF file for the drone model
        # For now, we'll use a simple shape as a placeholder
        drone_ids = []
        
        # Create a basic drone shape for each agent
        for i in range(self.num_drones):
            # Position drones in a formation
            x = np.cos(2 * np.pi * i / self.num_drones) * 5
            y = np.sin(2 * np.pi * i / self.num_drones) * 5
            z = 1.0
            
            # In reality, you would load a proper URDF model
            # drone_id = p.loadURDF("drone.urdf", [x, y, z])
            
            # For now, use a cylinder as a placeholder
            visual_id = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=0.2,
                length=0.1,
                rgbaColor=[0, 0.5, 0.8, 1]
            )
            
            collision_id = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=0.2,
                height=0.1
            )
            
            drone_id = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=[x, y, z],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
            )
            
            drone_ids.append(drone_id)
            
        return drone_ids
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options for customizing reset
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Set seed if provided
        if seed is not None:
            self.seed_val = seed
            np.random.seed(seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # Recreate environment
        self._create_terrain()
        self.drone_ids = self._spawn_drones()
        
        # Get initial observation
        observation = self._get_observation()
        
        # Additional info
        info = {}
        
        return observation, info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        """Take a step in the environment.
        
        Args:
            action: Actions for all drones in the swarm
            
        Returns:
            observation: New observation after action
            reward: Reward for the action
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated (e.g., max steps)
            info: Additional information
        """
        self.step_count += 1
        
        # Apply actions to drones
        self._apply_action(action)
        
        # Step the simulation
        for _ in range(10):  # Simulate physics for stability
            p.stepSimulation()
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate rewards
        rewards = self._compute_reward()
        
        # Use mean reward as scalar reward for RL training
        reward = float(np.mean(rewards))
        
        # Store individual rewards in info dict
        info = {"drone_rewards": rewards}
        
        # Check if done
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray) -> None:
        """Apply actions to the drones.
        
        Args:
            action: Array of actions for each drone
        """
        for i, drone_id in enumerate(self.drone_ids):
            # Extract actions for this drone
            thrust = action[i, 0]  # Vertical thrust
            roll = action[i, 1]    # Roll control
            pitch = action[i, 2]   # Pitch control
            yaw = action[i, 3]     # Yaw control
            
            # Scale actions to appropriate ranges
            thrust_force = 10.0 * (thrust + 1.0)  # Scale to 0-20N
            
            # In a real implementation, you would apply proper aerodynamic forces
            # For this simplified version, we apply direct forces and torques
            
            # Apply thrust (always in local z-direction)
            p.applyExternalForce(
                objectUniqueId=drone_id,
                linkIndex=-1,
                forceObj=[0, 0, thrust_force],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME
            )
            
            # Apply torques for rotation control
            roll_torque = 0.5 * roll
            pitch_torque = 0.5 * pitch
            yaw_torque = 0.5 * yaw
            
            p.applyExternalTorque(
                objectUniqueId=drone_id,
                linkIndex=-1,
                torqueObj=[roll_torque, pitch_torque, yaw_torque],
                flags=p.LINK_FRAME
            )
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation from the environment.
        
        Returns:
            Observations for all drones
        """
        observations = np.zeros((self.num_drones, 22), dtype=np.float32)
        
        for i, drone_id in enumerate(self.drone_ids):
            # Get position and orientation
            pos, orn = p.getBasePositionAndOrientation(drone_id)
            euler = p.getEulerFromQuaternion(orn)
            
            # Get velocities
            lin_vel, ang_vel = p.getBaseVelocity(drone_id)
            
            # Combine basic state information
            drone_state = list(pos) + list(euler) + list(lin_vel) + list(ang_vel)
            
            # Get sensor readings - in a real implementation, 
            # these would be raycasts or other sensors
            # For now, just fill with zeros
            sensor_readings = np.zeros(10, dtype=np.float32)
            
            # Combine all observations
            observations[i, :12] = drone_state
            observations[i, 12:] = sensor_readings
            
        return observations
    
    def _compute_reward(self) -> np.ndarray:
        """Compute rewards for all drones.
        
        Returns:
            Array of rewards for each drone
        """
        rewards = np.zeros(self.num_drones, dtype=np.float32)
        
        # In a real implementation, you would define more sophisticated 
        # reward functions based on mission objectives
        
        # Example: reward for altitude maintenance around z=5
        for i, drone_id in enumerate(self.drone_ids):
            pos, _ = p.getBasePositionAndOrientation(drone_id)
            
            # Altitude maintenance reward
            target_altitude = 5.0
            altitude_error = abs(pos[2] - target_altitude)
            altitude_reward = np.exp(-altitude_error)
            
            # Staying within bounds reward
            distance_from_center = np.sqrt(pos[0]**2 + pos[1]**2)
            boundary_reward = np.exp(-max(0, distance_from_center - 20) / 10)
            
            # Combine rewards
            rewards[i] = 0.5 * altitude_reward + 0.5 * boundary_reward
            
        return rewards
    
    def _check_terminated(self) -> bool:
        """Check if the episode should be terminated.
        
        Returns:
            True if the episode should terminate, False otherwise
        """
        for drone_id in self.drone_ids:
            pos, _ = p.getBasePositionAndOrientation(drone_id)
            
            # Terminate if drone flies too high or crashes
            if pos[2] < 0.2 or pos[2] > 100:
                return True
            
            # Terminate if drone goes out of bounds
            if abs(pos[0]) > 50 or abs(pos[1]) > 50:
                return True
        
        return False
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # For human rendering, ensure the GUI is enabled
            if self.headless:
                print("Cannot render in human mode when headless=True")
                return None
                
            # The PyBullet GUI is already showing the simulation
            time.sleep(1.0 / self.metadata["render_fps"])
            return None
            
        elif self.render_mode == "rgb_array":
            # Set up camera for rendering
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=20,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0
            )
            
            # Get camera image
            width, height = 640, 480
            _, _, rgb_img, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Convert to proper RGB array
            rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)
            rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
            
            return rgb_array
        
        return None
    
    def close(self):
        """Close the environment and clean up resources."""
        if p.isConnected():
            p.disconnect()


# Example combat scenario subclass
class DroneSwarmCombatEnv(DroneSwarmEnv):
    """Drone swarm environment with combat objectives."""
    
    def __init__(
        self,
        num_friendly_drones: int = 5,
        num_enemy_drones: int = 5,
        use_obstacles: bool = True,  # Enable obstacles
        num_obstacles: int = 10,     # Number of obstacles to create
        enable_firing: bool = True,  # Enable firing mechanics
        enable_destruction: bool = True,  # Enable drone destruction
        **kwargs
    ):
        """Initialize combat environment.
        
        Args:
            num_friendly_drones: Number of friendly drones (controlled by agent)
            num_enemy_drones: Number of enemy drones
            use_obstacles: Whether to include obstacles in the environment
            num_obstacles: Number of obstacles to create
            enable_firing: Whether drones can fire weapons
            enable_destruction: Whether drones can be destroyed
            **kwargs: Additional arguments for the base environment
        """
        self.num_friendly_drones = num_friendly_drones
        self.num_enemy_drones = num_enemy_drones
        self.use_obstacles = use_obstacles
        self.num_obstacles = num_obstacles
        self.enable_firing = enable_firing
        self.enable_destruction = enable_destruction
        
        # Get environment size from kwargs (default to [100, 100, 50] if not provided)
        self.env_size = kwargs.get('env_size', [100, 100, 50])
        
        # Tracking for drone and enemy health/status
        self.drone_health = np.ones(num_friendly_drones) * 100.0  # Full health
        self.enemy_health = np.ones(num_enemy_drones) * 100.0     # Full health
        self.destroyed_drones = []  # Track destroyed friendly drones
        self.destroyed_enemies = [] # Track destroyed enemy drones
        
        # Weapon cooldown for all drones (friendly and enemy)
        self.weapon_cooldown = np.zeros(num_friendly_drones)
        self.enemy_cooldown = np.zeros(num_enemy_drones)
        self.cooldown_period = 10  # Steps between weapon firing
        
        # Obstacle IDs for tracking
        self.obstacle_ids = []
        
        # Enemy behavior tracking
        self.enemy_hiding_status = np.zeros(num_enemy_drones, dtype=bool)  # Track if enemies are hiding
        
        # Deployment location (will be set by agent action)
        self.deployment_location = np.zeros((3,))  # Default center deployment
        
        # Initialize base class with total number of drones
        # Expand action space for additional actions: firing, proximity control, aggression, deployment location
        self.use_extended_action_space = True
        
        # Filter kwargs to only include parameters accepted by DroneSwarmEnv.__init__
        parent_params = {
            'max_steps': kwargs.get('max_steps', 1000),
            'render_mode': kwargs.get('render_mode', None),
            'headless': kwargs.get('headless', True),
            'seed': kwargs.get('seed', None)
        }
            
        super().__init__(num_drones=num_friendly_drones, **parent_params)
        
        # Store any additional parameters for use in this class
        self.step_time = kwargs.get('step_time', 0.1)
        
        # Override action space to include:
        # - Original 4 controls per drone (thrust and rotation)
        # - Firing control (1 per drone)
        # - Proximity preference (1 value for swarm)
        # - Aggression level (1 value for swarm)
        # - Obstacle avoidance distance (1 value for swarm)
        # - Deployment location (3 values for swarm: x, y, z)
        if self.use_extended_action_space:
            action_size = num_friendly_drones * 4  # Base drone controls
            action_size += num_friendly_drones     # Firing control for each drone
            action_size += 4                       # Proximity, aggression, obstacle distance, deployment
            
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0, 
                shape=(action_size,),  # Ensure shape is a tuple
                dtype=np.float32
            )
    
    def reset(self, seed=None, options=None):
        """Reset the combat environment."""
        # Reset health and status tracking
        self.drone_health = np.ones(self.num_friendly_drones) * 100.0
        self.enemy_health = np.ones(self.num_enemy_drones) * 100.0
        self.destroyed_drones = []
        self.destroyed_enemies = []
        self.weapon_cooldown = np.zeros(self.num_friendly_drones)
        self.enemy_cooldown = np.zeros(self.num_enemy_drones)
        self.enemy_hiding_status = np.zeros(self.num_enemy_drones, dtype=bool)
        
        # Reset deployment location
        self.deployment_location = np.zeros((3,))
        
        # Reset base environment
        observation, info = super().reset(seed=seed, options=options)
        
        # Create obstacles
        if self.use_obstacles:
            self._create_obstacles()
        
        # Spawn enemy drones
        self.enemy_drone_ids = self._spawn_enemy_drones()
        
        # Hide some enemies
        self._update_enemy_hiding()
        
        return observation, info
    
    def _create_obstacles(self):
        """Create obstacles in the environment."""
        # Remove any existing obstacles
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id)
        self.obstacle_ids = []
        
        # Create new obstacles
        for i in range(self.num_obstacles):
            # Random position for obstacles avoiding center area
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(5, 20)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = np.random.uniform(1, 10)
            
            # Random size for obstacles
            size_x = np.random.uniform(0.5, 3.0)
            size_y = np.random.uniform(0.5, 3.0)
            size_z = np.random.uniform(0.5, 3.0)
            
            # Create obstacle
            visual_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[size_x/2, size_y/2, size_z/2],
                rgbaColor=[0.5, 0.5, 0.5, 0.8]  # Gray, semi-transparent
            )
            
            collision_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[size_x/2, size_y/2, size_z/2]
            )
            
            obstacle_id = p.createMultiBody(
                baseMass=0,  # Static obstacle
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=[x, y, z]
            )
            
            self.obstacle_ids.append(obstacle_id)
    
    def _spawn_enemy_drones(self):
        """Spawn enemy drones in the environment."""
        enemy_ids = []
        
        # Reduced radius to make enemy drones closer together (15 -> 5)
        spawn_radius = 5.0  # Decreased from 15.0 to make drones closer together
        
        for i in range(self.num_enemy_drones):
            # Position enemy drones opposite to friendly drones but closer together
            x = np.cos(2 * np.pi * i / self.num_enemy_drones) * spawn_radius
            y = np.sin(2 * np.pi * i / self.num_enemy_drones) * spawn_radius
            z = 5.0
            
            # Use a different color for enemy drones
            visual_id = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=0.2,
                length=0.1,
                rgbaColor=[0.8, 0.1, 0.1, 1]  # Red color for enemies
            )
            
            collision_id = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=0.2,
                height=0.1
            )
            
            drone_id = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=[x, y, z],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
            )
            
            enemy_ids.append(drone_id)
            
        return enemy_ids
    
    def _update_enemy_hiding(self):
        """Update which enemies are hiding."""
        # Randomly decide which enemies should hide
        self.enemy_hiding_status = np.random.rand(self.num_enemy_drones) > 0.7  # 30% chance of hiding
        
        # Move hiding enemies behind obstacles
        for i, (drone_id, is_hiding) in enumerate(zip(self.enemy_drone_ids, self.enemy_hiding_status)):
            if is_hiding and self.obstacle_ids:
                # Pick a random obstacle to hide behind
                obstacle_id = np.random.choice(self.obstacle_ids)
                obstacle_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
                
                # Position enemy behind obstacle from perspective of center
                direction = np.array(obstacle_pos) / np.linalg.norm(obstacle_pos)
                hide_pos = np.array(obstacle_pos) + direction * 2.0  # 2 units behind obstacle
                hide_pos[2] = np.random.uniform(1, 10)  # Random height
                
                # Move enemy to hiding position
                p.resetBasePositionAndOrientation(
                    drone_id,
                    hide_pos,
                    p.getQuaternionFromEuler([0, 0, 0])
                )
    
    def step(self, action):
        """Execute one step in the environment."""
        if self.use_extended_action_space:
            # Extract the different action components
            num_basic_actions = self.num_drones * 4
            num_fire_actions = self.num_drones
            
            standard_actions = action[:num_basic_actions].reshape((self.num_drones, 4))
            fire_actions = action[num_basic_actions:num_basic_actions + num_fire_actions]
            
            # Get formation and combat parameters (last 4 values)
            extra_params_start = num_basic_actions + num_fire_actions
            proximity_control = action[extra_params_start]     # Formation spacing control
            aggression_level = action[extra_params_start + 1]  # Combat aggression
            obstacle_distance = action[extra_params_start + 2] # Obstacle avoidance distance
            deployment_factor = action[extra_params_start + 3] # Deployment location control
            
            # Last value controls deployment if at start of episode
            if self.step_count == 0:
                # Scale deployment values to reasonable range (-20, 20) for x,y and (0, 10) for z
                # Use one value to control general positioning (near/far from center)
                angle = np.random.uniform(0, 2 * np.pi)  # Random angle for deployment
                radius = (deployment_factor + 1) * 10.0   # 0-20 units from center
                
                deployment_x = radius * np.cos(angle)
                deployment_y = radius * np.sin(angle)
                deployment_z = (deployment_factor * 0.5 + 0.5) * 10.0  # Scale to (0, 10)
                self.deployment_location = np.array([deployment_x, deployment_y, deployment_z])
                
                # Move drones to deployment location
                for i, drone_id in enumerate(self.drone_ids):
                    # Add some spread around deployment location
                    offset_x = np.cos(2 * np.pi * i / self.num_drones) * (2.0 + proximity_control * 3.0)
                    offset_y = np.sin(2 * np.pi * i / self.num_drones) * (2.0 + proximity_control * 3.0)
                    
                    p.resetBasePositionAndOrientation(
                        drone_id,
                        [
                            self.deployment_location[0] + offset_x,
                            self.deployment_location[1] + offset_y,
                            self.deployment_location[2]
                        ],
                        p.getQuaternionFromEuler([0, 0, 0])
                    )
            
            # Process firing actions
            if self.enable_firing:
                self._process_firing(fire_actions, aggression_level)
                
            # Apply standard drone control actions
            self._apply_action(standard_actions)
        else:
            # Just use the standard action handling
            self._apply_action(action)
        
        # Update physics simulation
        for _ in range(10):  # 10 simulation steps per environment step
            # Check for collisions with obstacles
            self._check_collisions()
            
            # Update enemy behavior
            self._update_enemies(aggression_level if self.use_extended_action_space else 0.0)
            
            # Step simulation
            p.stepSimulation()
        
        # Get updated observation
        observation = self._get_observation()
        
        # Compute reward
        rewards, reward_components = self._compute_reward(
            proximity_control if self.use_extended_action_space else 0.0,
            obstacle_distance if self.use_extended_action_space else 1.0
        )
        
        # Convert rewards to scalar (mean across all drones)
        reward = float(np.mean(rewards))
        
        # Check if episode is done
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps
        
        # Increment step counter
        self.step_count += 1
        
        # Gather detailed information for visualization
        friendly_drones_remaining = len(self.drone_ids) - len(self.destroyed_drones)
        enemy_drones_remaining = len(self.enemy_drone_ids) - len(self.destroyed_enemies)
        
        # Get current health status of all drones
        drone_health = []
        for i, drone_id in enumerate(self.drone_ids):
            if drone_id in self.destroyed_drones:
                drone_health.append(0.0)  # Destroyed drones have 0 health
            else:
                # Get health percentage (0-100)
                health_pct = (self.drone_health[i] / 100.0) * 100.0
                drone_health.append(float(health_pct))
        
        # Count detected enemies (those within detection range of any friendly drone)
        detected_enemies = self._count_detected_enemies()
        
        # Additional info dictionary with detailed metrics for visualization
        info = {
            "drone_rewards": rewards.tolist(),  # Per-drone rewards
            "reward_components": reward_components,  # Components that make up the reward
            "friendly_drones_remaining": friendly_drones_remaining,
            "enemy_drones_remaining": enemy_drones_remaining,
            "drone_health": drone_health,
            "detected_enemies": detected_enemies,
            "destroyed_enemies": len(self.destroyed_enemies),
            "step_count": self.step_count
        }
        
        return observation, reward, terminated, truncated, info
    
    def _process_firing(self, fire_actions, aggression_level):
        """Process drone firing actions."""
        for i, (drone_id, fire_action) in enumerate(zip(self.drone_ids, fire_actions)):
            # Skip destroyed drones
            if i in self.destroyed_drones:
                continue
                
            # Check if drone wants to fire and cooldown has elapsed
            if fire_action > 0.5 and self.weapon_cooldown[i] <= 0:
                drone_pos, drone_orient = p.getBasePositionAndOrientation(drone_id)
                
                # Get direction drone is facing
                rotation_matrix = p.getMatrixFromQuaternion(drone_orient)
                forward_vector = [rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]]
                
                # Find closest enemy
                closest_enemy_idx = -1
                closest_distance = float('inf')
                
                for j, enemy_id in enumerate(self.enemy_drone_ids):
                    if j in self.destroyed_enemies:
                        continue
                    
                    try:
                        enemy_pos, _ = p.getBasePositionAndOrientation(enemy_id)
                        
                        # Calculate distance to enemy
                        distance = np.sqrt(
                            (drone_pos[0] - enemy_pos[0])**2 +
                            (drone_pos[1] - enemy_pos[1])**2 +
                            (drone_pos[2] - enemy_pos[2])**2
                        )
                        
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_enemy_idx = j
                    except Exception as e:
                        print(f"Warning: Failed to get position of enemy {enemy_id} in firing calculation. Error: {e}")
                        continue
                
                # Fire at closest enemy if within range
                # Range is influenced by aggression level
                max_range = 10.0 + aggression_level * 5.0  # 10-15 units range based on aggression
                
                if closest_enemy_idx >= 0 and closest_distance < max_range:
                    try:
                        # Create visual effect for firing (temporary object)
                        enemy_pos, _ = p.getBasePositionAndOrientation(self.enemy_drone_ids[closest_enemy_idx])
                        p.addUserDebugLine(
                            drone_pos,
                            enemy_pos,
                            [1, 1, 0],  # Yellow laser
                            2.0,        # Line width
                            lifeTime=0.1  # Disappears after 0.1 seconds
                        )
                        
                        # Apply damage to enemy (higher damage with higher aggression)
                        hit_chance = 0.7 + aggression_level * 0.3  # 70-100% chance to hit based on aggression
                        
                        if np.random.random() < hit_chance:
                            damage = 20 + aggression_level * 10  # 20-30 damage based on aggression
                            self.enemy_health[closest_enemy_idx] -= damage
                            
                            # Check if enemy is destroyed
                            if self.enemy_health[closest_enemy_idx] <= 0 and self.enable_destruction:
                                self.destroyed_enemies.append(closest_enemy_idx)
                                
                                # Visual effect for destruction
                                p.addUserDebugLine(
                                    enemy_pos,
                                    [enemy_pos[0], enemy_pos[1], enemy_pos[2] + 3],
                                    [1, 0.5, 0],  # Orange explosion
                                    10.0,        # Line width
                                    lifeTime=0.5  # Disappears after 0.5 seconds
                                )
                                
                                # Hide destroyed enemy
                                p.resetBasePositionAndOrientation(
                                    self.enemy_drone_ids[closest_enemy_idx],
                                    [0, 0, -100],  # Move far below ground
                                    [0, 0, 0, 1]
                                )
                    except Exception as e:
                        print(f"Warning: Failed to get position of enemy {self.enemy_drone_ids[closest_enemy_idx]} in firing calculation. Error: {e}")
                        continue
                
                # Reset cooldown
                self.weapon_cooldown[i] = self.cooldown_period
            
            # Decrease cooldown
            if self.weapon_cooldown[i] > 0:
                self.weapon_cooldown[i] -= 1
    
    def _update_enemies(self, player_aggression):
        """Update enemy behavior."""
        for i, enemy_id in enumerate(self.enemy_drone_ids):
            # Skip destroyed enemies
            if i in self.destroyed_enemies:
                continue
                
            # Get enemy position
            enemy_pos, enemy_orient = p.getBasePositionAndOrientation(enemy_id)
            
            # 1. Update hiding behavior
            if np.random.random() < 0.01:  # 1% chance per step to change hiding status
                self.enemy_hiding_status[i] = not self.enemy_hiding_status[i]
                
                if self.enemy_hiding_status[i] and self.obstacle_ids:
                    # Find a new hiding spot
                    obstacle_id = np.random.choice(self.obstacle_ids)
                    obstacle_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
                    
                    # Position enemy behind obstacle from perspective of center
                    direction = np.array(obstacle_pos) / np.linalg.norm(obstacle_pos)
                    hide_pos = np.array(obstacle_pos) + direction * 2.0
                    hide_pos[2] = np.random.uniform(1, 10)
                    
                    # Move gradually to hiding position
                    target_pos = hide_pos
                else:
                    # Move to random patrol position
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(10, 20)
                    target_pos = [
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                        np.random.uniform(1, 10)
                    ]
            else:
                # Continue current behavior
                if self.enemy_hiding_status[i]:
                    # Stay hidden
                    target_pos = enemy_pos
                else:
                    # Find closest drone to attack
                    closest_drone_idx = -1
                    closest_distance = float('inf')
                    
                    for j, drone_id in enumerate(self.drone_ids):
                        if j in self.destroyed_drones:
                            continue
                            
                        drone_pos, _ = p.getBasePositionAndOrientation(drone_id)
                        
                        # Calculate distance
                        distance = np.sqrt(
                            (enemy_pos[0] - drone_pos[0])**2 +
                            (enemy_pos[1] - drone_pos[1])**2 +
                            (enemy_pos[2] - drone_pos[2])**2
                        )
                        
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_drone_idx = j
                    
                    if closest_drone_idx >= 0:
                        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_ids[closest_drone_idx])
                        
                        # Optimal attack distance (influenced by player aggression)
                        optimal_distance = 8.0 - player_aggression * 3.0  # 5-8 units depending on player aggression
                        
                        # Calculate direction to drone
                        direction = np.array([
                            drone_pos[0] - enemy_pos[0],
                            drone_pos[1] - enemy_pos[1],
                            drone_pos[2] - enemy_pos[2]
                        ])
                        
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                        
                        # Calculate target position
                        current_distance = np.sqrt(
                            (enemy_pos[0] - drone_pos[0])**2 +
                            (enemy_pos[1] - drone_pos[1])**2 +
                            (enemy_pos[2] - drone_pos[2])**2
                        )
                        
                        if current_distance < optimal_distance - 1.0:
                            # Move away from drone
                            target_pos = [
                                enemy_pos[0] - direction[0] * 0.2,
                                enemy_pos[1] - direction[1] * 0.2,
                                enemy_pos[2] - direction[2] * 0.2
                            ]
                        elif current_distance > optimal_distance + 1.0:
                            # Move toward drone
                            target_pos = [
                                enemy_pos[0] + direction[0] * 0.2,
                                enemy_pos[1] + direction[1] * 0.2,
                                enemy_pos[2] + direction[2] * 0.2
                            ]
                        else:
                            # Stay at optimal distance and fire
                            target_pos = enemy_pos
                            
                            # Try to fire at drone
                            if self.enable_firing and self.enemy_cooldown[i] <= 0:
                                # Fire at drone
                                p.addUserDebugLine(
                                    enemy_pos,
                                    drone_pos,
                                    [1, 0, 0],  # Red laser
                                    2.0,        # Line width
                                    lifeTime=0.1  # Disappears after 0.1 seconds
                                )
                                
                                # Apply damage to drone
                                hit_chance = 0.5  # Base hit chance
                                
                                if np.random.random() < hit_chance:
                                    damage = 10  # Base damage
                                    self.drone_health[closest_drone_idx] -= damage
                                    
                                    # Check if drone is destroyed
                                    if self.drone_health[closest_drone_idx] <= 0 and self.enable_destruction:
                                        self.destroyed_drones.append(closest_drone_idx)
                                        
                                        # Visual effect for destruction
                                        p.addUserDebugLine(
                                            drone_pos,
                                            [drone_pos[0], drone_pos[1], drone_pos[2] + 3],
                                            [1, 0.5, 0],  # Orange explosion
                                            10.0,        # Line width
                                            lifeTime=0.5  # Disappears after 0.5 seconds
                                        )
                                        
                                        # Hide destroyed drone
                                        p.resetBasePositionAndOrientation(
                                            self.drone_ids[closest_drone_idx],
                                            [0, 0, -100],  # Move far below ground
                                            [0, 0, 0, 1]
                                        )
                                
                                # Reset cooldown
                                self.enemy_cooldown[i] = self.cooldown_period
                    else:
                        # No drones to attack, just hover
                        target_pos = enemy_pos
            
            # Decrease enemy cooldown
            if self.enemy_cooldown[i] > 0:
                self.enemy_cooldown[i] -= 1
            
            # Simple enemy movement toward target
            if not np.array_equal(target_pos, enemy_pos):
                # Apply force toward target
                direction = np.array([
                    target_pos[0] - enemy_pos[0],
                    target_pos[1] - enemy_pos[1],
                    target_pos[2] - enemy_pos[2]
                ])
                
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    
                    p.applyExternalForce(
                        enemy_id,
                        -1,  # Apply to base link
                        [direction[0] * 5.0, direction[1] * 5.0, direction[2] * 5.0],
                        [0, 0, 0],  # Apply at center of mass
                        p.WORLD_FRAME
                    )
    
    def _check_collisions(self):
        """Check for collisions with obstacles."""
        if not self.use_obstacles:
            return
            
        # Check each drone
        for i, drone_id in enumerate(self.drone_ids):
            if i in self.destroyed_drones:
                continue
                
            drone_pos, _ = p.getBasePositionAndOrientation(drone_id)
            
            # Check collision with each obstacle
            for obstacle_id in self.obstacle_ids:
                contact_points = p.getContactPoints(drone_id, obstacle_id)
                
                if len(contact_points) > 0:
                    # Collision detected, apply damage
                    self.drone_health[i] -= 5.0  # 5 damage per collision
                    
                    # Visual effect for collision
                    p.addUserDebugLine(
                        drone_pos,
                        [drone_pos[0] + 1, drone_pos[1] + 1, drone_pos[2] + 1],
                        [1, 1, 1],  # White spark
                        5.0,        # Line width
                        lifeTime=0.1  # Disappears after 0.1 seconds
                    )
                    
                    # Check if drone is destroyed
                    if self.drone_health[i] <= 0 and self.enable_destruction:
                        self.destroyed_drones.append(i)
                        
                        # Visual effect for destruction
                        p.addUserDebugLine(
                            drone_pos,
                            [drone_pos[0], drone_pos[1], drone_pos[2] + 3],
                            [1, 0.5, 0],  # Orange explosion
                            10.0,        # Line width
                            lifeTime=0.5  # Disappears after 0.5 seconds
                        )
                        
                        # Hide destroyed drone
                        p.resetBasePositionAndOrientation(
                            drone_id,
                            [0, 0, -100],  # Move far below ground
                            [0, 0, 0, 1]
                        )
                    
                    break  # Only count one collision per obstacle per step
    
    def _compute_reward(self, proximity_preference=0.0, obstacle_distance_preference=1.0):
        """Compute rewards for combat scenario."""
        rewards = np.zeros(self.num_drones, dtype=np.float32)
        
        # Initialize reward component tracking for visualization
        reward_components = {
            "basic": 0.0,
            "formation": 0.0,
            "combat": 0.0,
            "obstacle": 0.0,
            "health": 0.0,
            "mission_completion": 0.0,
            "enemy_elimination": 0.0,
            "enemy_detection": 0.0,
            "detection_without_damage": 0.0
        }
        
        # For destroyed drones, award large negative reward
        for i in range(self.num_drones):
            if i in self.destroyed_drones:
                rewards[i] = -5.0  # Larger penalty for being destroyed
                continue
                
            # Position of current drone
            drone_pos, _ = p.getBasePositionAndOrientation(self.drone_ids[i])
            
            # 1. Basic flight and stability reward (smaller weight)
            basic_reward = 0.0
            
            # Reward for height (being in the air)
            height_reward = min(drone_pos[2] / 5.0, 1.0)  # Max reward at 5 units height
            basic_reward += height_reward * 0.5
            
            # 2. Formation and proximity reward
            formation_reward = 0.0
            
            for j, other_id in enumerate(self.drone_ids):
                if i == j or j in self.destroyed_drones:
                    continue
                    
                other_pos, _ = p.getBasePositionAndOrientation(other_id)
                
                # Distance to other drone
                distance = np.sqrt(
                    (drone_pos[0] - other_pos[0])**2 +
                    (drone_pos[1] - other_pos[1])**2 +
                    (drone_pos[2] - other_pos[2])**2
                )
                
                # Optimal distance depends on proximity preference
                optimal_distance = 2.0 + proximity_preference * 6.0  # 2-8 units
                
                # Reward for maintaining optimal distance
                distance_reward = np.exp(-(distance - optimal_distance)**2 / (2.0 * 4.0))
                formation_reward += distance_reward
            
            # Normalize by number of other drones
            if self.num_drones > 1:
                formation_reward /= (self.num_drones - 1)
            
            # 3. Combat reward - MAJOR FOCUS
            combat_reward = 0.0
            enemy_elimination_reward = 0.0
            enemy_detection_reward = 0.0
            detection_without_damage_reward = 0.0
            
            # *** ENEMY ELIMINATION REWARD (MAJOR) ***
            # Large reward for destroying enemies
            num_destroyed_enemies = len(self.destroyed_enemies)
            enemy_elimination_reward = num_destroyed_enemies * 3.0  # Increased from 1.0 to 3.0
            
            # Reward for damaging enemies
            total_enemy_damage = sum([100.0 - health for health in self.enemy_health])
            enemy_elimination_reward += total_enemy_damage * 0.03  # Increased from 0.01 to 0.03
            
            # Add to combat reward
            combat_reward += enemy_elimination_reward
            
            # *** ENEMY DETECTION REWARD ***
            # Track how many enemies have been detected
            enemies_detected = 0
            enemies_detected_without_damage = 0
            
            for j, enemy_id in enumerate(self.enemy_drone_ids):
                if j in self.destroyed_enemies:
                    continue
                
                try:
                    enemy_pos, _ = p.getBasePositionAndOrientation(enemy_id)
                    
                    # Calculate distance to enemy
                    distance = np.sqrt(
                        (drone_pos[0] - enemy_pos[0])**2 +
                        (drone_pos[1] - enemy_pos[1])**2 +
                        (drone_pos[2] - enemy_pos[2])**2
                    )
                    
                    # If enemy is within detection range (20 units)
                    if distance < 20.0:
                        enemies_detected += 1
                        # Extra reward if enemy is detected and drone is at full health
                        if self.drone_health[i] > 95.0:  # Almost full health
                            enemies_detected_without_damage += 1
                except Exception as e:
                    print(f"Warning: Failed to get position of enemy {enemy_id} in reward calculation. Error: {e}")
                    continue
            
            # Reward for detecting enemies
            detection_ratio = enemies_detected / max(1, len(self.enemy_drone_ids) - len(self.destroyed_enemies))
            enemy_detection_reward = detection_ratio * 1.0
            
            # Extra reward for detecting enemies without taking damage
            if enemies_detected > 0:
                detection_without_damage_ratio = enemies_detected_without_damage / enemies_detected
                detection_without_damage_reward = detection_without_damage_ratio * 2.0
                enemy_detection_reward += detection_without_damage_reward
            
            # Add to combat reward
            combat_reward += enemy_detection_reward
            
            # 4. Obstacle avoidance reward
            obstacle_reward = 0.0
            
            if self.use_obstacles:
                for obstacle_id in self.obstacle_ids:
                    try:
                        obstacle_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
                        
                        # Calculate distance
                        distance = np.sqrt(
                            (drone_pos[0] - obstacle_pos[0])**2 +
                            (drone_pos[1] - obstacle_pos[1])**2 +
                            (drone_pos[2] - obstacle_pos[2])**2
                        )
                        
                        # Minimum safe distance depends on preference
                        min_safe_distance = 1.0 + obstacle_distance_preference * 2.0  # 1-3 units
                        
                        # Penalty for being too close
                        if distance < min_safe_distance:
                            obstacle_reward -= (1.0 - distance / min_safe_distance) * 0.5
                        else:
                            # Small reward for successful avoidance
                            obstacle_reward += 0.1
                    except Exception as e:
                        print(f"Warning: Failed to get position of obstacle {obstacle_id}. Error: {e}")
                        continue
                
                # Normalize by number of obstacles
                if self.num_obstacles > 0:
                    obstacle_reward /= self.num_obstacles
            
            # 5. Health reward - Increased importance to encourage avoiding damage
            health_reward = self.drone_health[i] / 100.0  # Reward for maintaining health
            
            # Additional reward for perfect health (no damage taken)
            if self.drone_health[i] >= 95.0:
                health_reward *= 1.5  # 50% bonus for staying at high health
            
            # Mission completion bonus
            mission_completion_reward = 0.0
            if len(self.destroyed_enemies) == self.num_enemy_drones:
                mission_completion_reward = 5.0  # Large bonus for mission success
            
            # Combine rewards with appropriate weights - Adjusted to prioritize mission objectives
            rewards[i] = (
                0.1 * basic_reward +      # Reduced from 0.2
                0.1 * formation_reward +  # Reduced from 0.2
                0.5 * combat_reward +     # Increased from 0.3
                0.1 * obstacle_reward +   # Same
                0.2 * health_reward +     # Same
                mission_completion_reward # Added separately
            )
            
            # Update reward components (add each drone's contribution)
            reward_components["basic"] += basic_reward * 0.1 / self.num_drones
            reward_components["formation"] += formation_reward * 0.1 / self.num_drones
            reward_components["combat"] += combat_reward * 0.5 / self.num_drones
            reward_components["obstacle"] += obstacle_reward * 0.1 / self.num_drones
            reward_components["health"] += health_reward * 0.2 / self.num_drones
            reward_components["mission_completion"] += mission_completion_reward / self.num_drones
            reward_components["enemy_elimination"] += enemy_elimination_reward * 0.5 / self.num_drones
            reward_components["enemy_detection"] += enemy_detection_reward * 0.5 / self.num_drones
            reward_components["detection_without_damage"] += detection_without_damage_reward * 0.5 / self.num_drones
            
        return rewards, reward_components
    
    def _check_terminated(self):
        """Check if the episode should terminate."""
        # Terminate if all friendly drones are destroyed
        if len(self.destroyed_drones) == self.num_drones:
            return True
            
        # Terminate if all enemy drones are destroyed
        if len(self.destroyed_enemies) == self.num_enemy_drones:
            return True
            
        return False
    
    def _get_observation(self):
        """Get the current observation."""
        observation = np.zeros((self.num_drones, 22), dtype=np.float32)
        
        for i, drone_id in enumerate(self.drone_ids):
            if i in self.destroyed_drones:
                # For destroyed drones, provide zero observation
                continue
                
            # Get drone state
            pos, orient = p.getBasePositionAndOrientation(drone_id)
            linear_vel, angular_vel = p.getBaseVelocity(drone_id)
            
            # Convert quaternion to Euler angles
            euler = p.getEulerFromQuaternion(orient)
            
            # Basic drone state: position (3), orientation (3), velocity (6)
            observation[i, 0:3] = pos
            observation[i, 3:6] = euler
            observation[i, 6:9] = linear_vel
            observation[i, 9:12] = angular_vel
            
            # Sensor readings for nearby objects (10 values)
            sensor_readings = np.zeros(10)
            
            # 1-3: Nearest enemy data
            nearest_enemy_dist = float('inf')
            nearest_enemy_dir = np.zeros(3)
            
            for j, enemy_id in enumerate(self.enemy_drone_ids):
                if j in self.destroyed_enemies:
                    continue
                
                try:
                    enemy_pos, _ = p.getBasePositionAndOrientation(enemy_id)
                    
                    # Calculate distance
                    distance = np.sqrt(
                        (pos[0] - enemy_pos[0])**2 +
                        (pos[1] - enemy_pos[1])**2 +
                        (pos[2] - enemy_pos[2])**2
                    )
                    
                    if distance < nearest_enemy_dist:
                        nearest_enemy_dist = distance
                        nearest_enemy_dir = np.array([
                            enemy_pos[0] - pos[0],
                            enemy_pos[1] - pos[1],
                            enemy_pos[2] - pos[2]
                        ])
                        if np.linalg.norm(nearest_enemy_dir) > 0:
                            nearest_enemy_dir = nearest_enemy_dir / np.linalg.norm(nearest_enemy_dir)
                except Exception as e:
                    print(f"Warning: Failed to get position of enemy {enemy_id} in reward calculation. Error: {e}")
                    continue
            
            # Store nearest enemy data
            sensor_readings[0] = nearest_enemy_dist if nearest_enemy_dist != float('inf') else 100.0
            sensor_readings[1:4] = nearest_enemy_dir
            
            # 4-6: Nearest obstacle data
            nearest_obstacle_dist = float('inf')
            nearest_obstacle_dir = np.zeros(3)
            
            if self.use_obstacles:
                for obstacle_id in self.obstacle_ids:
                    try:
                        obstacle_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
                        
                        # Calculate distance
                        distance = np.sqrt(
                            (pos[0] - obstacle_pos[0])**2 +
                            (pos[1] - obstacle_pos[1])**2 +
                            (pos[2] - obstacle_pos[2])**2
                        )
                        
                        if distance < nearest_obstacle_dist:
                            nearest_obstacle_dist = distance
                            nearest_obstacle_dir = np.array([
                                obstacle_pos[0] - pos[0],
                                obstacle_pos[1] - pos[1],
                                obstacle_pos[2] - pos[2]
                            ])
                            if np.linalg.norm(nearest_obstacle_dir) > 0:
                                nearest_obstacle_dir = nearest_obstacle_dir / np.linalg.norm(nearest_obstacle_dir)
                    except Exception as e:
                        print(f"Warning: Failed to get position of obstacle {obstacle_id}. Error: {e}")
                        continue
            
            # Store nearest obstacle data
            sensor_readings[4] = nearest_obstacle_dist if nearest_obstacle_dist != float('inf') else 100.0
            sensor_readings[5:8] = nearest_obstacle_dir
            
            # 7: Drone health
            sensor_readings[8] = self.drone_health[i] / 100.0
            
            # 8: Weapon cooldown
            sensor_readings[9] = self.weapon_cooldown[i] / self.cooldown_period
            
            # Store sensor readings
            observation[i, 12:22] = sensor_readings
        
        return observation
    
    def _count_detected_enemies(self):
        """Count how many enemies are currently detected by friendly drones."""
        detected = 0
        detection_range = 15.0  # Detection range in world units
        
        # Count enemies that are within detection range of any friendly drone
        for enemy_id in self.enemy_drone_ids:
            if enemy_id in self.destroyed_enemies:
                continue  # Skip destroyed enemies
            
            try:
                enemy_pos, _ = p.getBasePositionAndOrientation(enemy_id)
                
                # Check if this enemy is within detection range of any friendly drone
                for i, drone_id in enumerate(self.drone_ids):
                    if drone_id in self.destroyed_drones:
                        continue  # Skip destroyed drones
                    
                    try:
                        drone_pos, _ = p.getBasePositionAndOrientation(drone_id)
                        distance = np.linalg.norm(np.array(drone_pos) - np.array(enemy_pos))
                        
                        if distance < detection_range:
                            detected += 1
                            break  # This enemy is detected, no need to check other drones
                    except Exception as e:
                        print(f"Warning: Failed to get position of friendly drone {drone_id}. Error: {e}")
                        continue
            except Exception as e:
                print(f"Warning: Failed to get position of enemy {enemy_id}. Error: {e}")
                continue
                    
        return detected 
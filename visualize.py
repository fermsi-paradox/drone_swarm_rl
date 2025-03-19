#!/usr/bin/env python3
"""
Visualization script for the trained drone swarm agent.

This script uses the visualization module to render and watch the performance
of the trained reinforcement learning agent in the drone swarm environment.
"""

import os
import sys

# Add the project root to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.visualization.render import main

if __name__ == "__main__":
    # Use the existing render.py functionality with the latest model
    sys.argv = [
        sys.argv[0],
        "--model", "src/models/drone_swarm_ppo_20250314_151357/best_model/best_model.zip",
        "--scenario", "src/configs/combat_scenario.yaml",
        "--episodes", "3",
    ]
    main() 
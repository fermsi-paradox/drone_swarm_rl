#!/bin/bash
# Directory setup script for drone_swarm_rl project
# This script creates the necessary directory structure and ensures files are in the right place

echo "Setting up directory structure for drone_swarm_rl project..."

# Create main directory structure
mkdir -p src/environments src/training src/configs models logs output/visualization

# Ensure src/environments directory exists
if [ ! -d "src/environments" ]; then
    echo "Creating src/environments directory"
    mkdir -p src/environments
fi

# Ensure src/training directory exists
if [ ! -d "src/training" ]; then
    echo "Creating src/training directory"
    mkdir -p src/training
fi

# Ensure src/configs directory exists
if [ ! -d "src/configs" ]; then
    echo "Creating src/configs directory"
    mkdir -p src/configs
fi

# Ensure models directory exists
if [ ! -d "models" ]; then
    echo "Creating models directory"
    mkdir -p models
fi

# Ensure logs directory exists
if [ ! -d "logs" ]; then
    echo "Creating logs directory"
    mkdir -p logs
fi

# Ensure output directory exists
if [ ! -d "output/visualization" ]; then
    echo "Creating output/visualization directory"
    mkdir -p output/visualization
fi

# Create empty __init__.py files for Python packages
touch src/__init__.py
touch src/environments/__init__.py
touch src/training/__init__.py

echo "Directory structure setup complete."
echo ""
echo "Your project structure should now look like this:"
echo "drone_swarm_rl/"
echo "├── requirements.txt"
echo "├── train_military_model.py"
echo "├── visualize_combat.py"
echo "├── train_and_visualize.py"
echo "├── models/"
echo "├── logs/"
echo "├── output/"
echo "└── src/"
echo "    ├── __init__.py"
echo "    ├── environments/"
echo "    │   ├── __init__.py"
echo "    │   └── drone_env.py"
echo "    ├── training/"
echo "    │   ├── __init__.py"
echo "    │   └── train.py"
echo "    └── configs/"
echo "        └── military_combat.yaml"
echo ""
echo "You can now run the following commands from the drone_swarm_rl directory:"
echo "python train_and_visualize.py --full_pipeline --timesteps 10000"
echo "or"
echo "python train_and_visualize.py --train --timesteps 10000" 
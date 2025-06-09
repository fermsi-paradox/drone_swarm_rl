# Military Drone Swarm Reinforcement Learning

This project implements a comprehensive reinforcement learning framework for training and evaluating military-grade drone swarm combat tactics. The system uses state-of-the-art RL algorithms to train drone swarms for tactical operations, reconnaissance, formation control, and engagement strategies.

**Optimized for Linux/WSL2 environments.**

## Key Features

- **Advanced Tactical Training**: Train drone swarms to develop sophisticated combat and reconnaissance strategies
- **Multi-Agent Coordination**: Enables emergent collective behaviors and swarm intelligence
- **Combat Simulation**: Realistic combat environment with weapons, damage modeling, and tactical objectives
- **Performance Analytics**: Comprehensive metrics for evaluating mission success, tactical effectiveness, and survivability
- **3D Visualization**: Interactive visualization of trained models with tactical overlays and combat statistics

## Quick Start Guide (5 Minutes)

For the fastest way to get started:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/drone_swarm_rl.git

# 2. Navigate to the project directory (IMPORTANT!)
cd drone_swarm_rl

# 3. Run the automated setup script
./setup_and_run.sh
```

Alternatively, you can run the interactive quick start:

```bash
python quick_start.py
```

The setup script will:
- Set up all necessary directories
- Create and activate a virtual environment
- Install all dependencies
- Check system requirements
- Offer to run a demo

## Manual Installation

### Prerequisites

- Python 3.8-3.12
- pip (Python package installer)
- For visualization: OpenGL and ffmpeg (for recording)
- Linux or WSL2 environment

### Step-by-Step Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/drone_swarm_rl.git
cd drone_swarm_rl  # IMPORTANT! All commands must be run from this directory
```

2. Set up the directory structure:
```bash
./setup_directories.sh
```

3. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you encounter issues with numpy installation on Python 3.12, you can try:
```bash
pip install numpy --pre
pip install -r requirements.txt
```

### System Dependencies (Linux/WSL2)

For OpenGL support on Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install python3-opengl freeglut3-dev mesa-utils
```

For recording videos:
```bash
sudo apt-get install ffmpeg
```

### Troubleshooting Common Issues

- **"File not found" errors**: Make sure you're running commands from the `drone_swarm_rl` directory
- **Import errors**: Check that the directory structure is correct (run `./setup_directories.sh`)
- **Permission denied on scripts**: Run `chmod +x setup_and_run.sh setup_directories.sh`
- **OpenGL-related errors**: Install the OpenGL packages listed above

## Project Structure

```
drone_swarm_rl/
├── src/                        # Core source code
│   ├── environments/           # RL environments
│   ├── training/               # Training algorithms
│   └── configs/                # Configuration files
│       └── military_combat.yaml # Military combat configuration
├── train_military_model.py     # Military-specific training script
├── visualize_combat.py         # Combat visualization
├── train_and_visualize.py      # Combined training/visualization
├── setup_directories.sh        # Directory setup script
├── setup_and_run.sh            # Automated setup and run script
├── quick_start.py              # Interactive setup and demo
└── README.md                   # This file
```

## How to Use

### Full Training Pipeline

To train, analyze, and visualize a military drone swarm model:

```bash
# Make sure you're in the project directory and virtual environment is activated
source venv/bin/activate
python train_and_visualize.py --full_pipeline --timesteps 100000
```

### Training Only

To only train a model:

```bash
python train_and_visualize.py --train --timesteps 1000000 --device cuda
```

### Visualization

To visualize an existing model:

```bash
python train_and_visualize.py --visualize --model_path path/to/model.zip --episodes 5 --record
```

### Model Analysis

To analyze an existing model's performance:

```bash
python train_and_visualize.py --analyze --model_path path/to/model.zip
```

## Military Combat Configuration

The military combat configuration (`src/configs/military_combat.yaml`) defines parameters for training drone swarms in military scenarios:

- **Swarm Size**: Configure the number of friendly and enemy drones
- **Weapon Parameters**: Set weapon range, damage, and hit probability
- **Sensor Systems**: Configure detection range, field of view, and sensor noise
- **Reward Structure**: Define rewards for tactical objectives, enemy elimination, and mission success
- **Network Architecture**: Deep neural network architecture specialized for tactical decision-making

## Visualization Controls

During visualization:

- **Arrow Keys**: Rotate camera
- **+/- Keys**: Zoom in/out
- **Mouse Drag**: Rotate camera
- **ESC**: Exit visualization

## Model Performance Metrics

The analysis component evaluates models on key military metrics:

- **Mission Success Rate**: Percentage of missions where all enemies were eliminated
- **Average Episode Reward**: Overall performance metric
- **Enemy Elimination Rate**: Percentage of enemy drones destroyed
- **Friendly Survival Rate**: Percentage of friendly drones that survived
- **Average Episode Length**: Mission duration in simulation steps

## Advanced Usage

### Custom Configurations

Create custom military scenarios by modifying the configuration:

```bash
python train_military_model.py --config src/configs/your_custom_config.yaml
```

### Recording Combat Footage

To record videos of combat simulations:

```bash
python visualize_combat.py --model path/to/model.zip --episodes 3 --record
```

### Benchmarking Models

Compare different tactical approaches:

```bash
python train_military_model.py --config src/configs/aggressive_tactics.yaml --analyze
python train_military_model.py --config src/configs/defensive_tactics.yaml --analyze
```

## Environment Setup

Remember to activate your virtual environment before running any commands:

```bash
# Navigate to project directory
cd drone_swarm_rl

# Activate virtual environment
source venv/bin/activate

# Now you can run any of the commands above
```

## Acknowledgments

This project builds upon research from:

- OpenAI's work on multi-agent reinforcement learning
- DARPA's OFFensive Swarm-Enabled Tactics (OFFSET) program
- Stable Baselines3 reinforcement learning library

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{drone_swarm_rl,
  author = {Your Name},
  title = {Military Drone Swarm Reinforcement Learning},
  year = {2024},
  url = {https://github.com/yourusername/drone_swarm_rl}
}
``` 
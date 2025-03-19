#!/bin/bash
# Setup and run script for Military Drone Swarm RL project
# This script helps new users get started quickly

set -e  # Exit on error

# Print section header
print_header() {
    echo "============================================="
    echo "$1"
    echo "============================================="
}

# Check if Python is installed
print_header "Checking Python installation"
if command -v python3 &> /dev/null; then
    python3 --version
else
    echo "Python 3 is not installed. Please install Python 3.8 or newer."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found."
    echo "Please run this script from the drone_swarm_rl directory."
    exit 1
fi

# Create virtual environment if it doesn't exist
print_header "Setting up virtual environment"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_header "Installing dependencies"
echo "Installing requirements..."
pip install -r requirements.txt

# Check for OpenGL
print_header "Checking system dependencies"
echo "Checking for OpenGL..."

if [ "$(uname)" == "Linux" ]; then
    # For Linux systems
    if ! dpkg -l | grep -q "python3-opengl"; then
        echo "OpenGL for Python not found. Install with:"
        echo "sudo apt-get install python3-opengl freeglut3-dev"
    else
        echo "OpenGL for Python is installed."
    fi
elif [ "$(uname)" == "Darwin" ]; then
    # For macOS
    echo "macOS detected. OpenGL should be available by default."
else
    # For Windows
    echo "Windows detected. Make sure you have up-to-date graphics drivers."
fi

# All set, run the program
print_header "Ready to run!"
echo "Your environment is set up. You can run the program with:"
echo "python train_and_visualize.py --full_pipeline --timesteps 100000"
echo ""
echo "Or run a quick demo with:"
echo "python train_and_visualize.py --train --timesteps 10000 --analyze"
echo ""

# Ask if user wants to run the demo
read -p "Would you like to run a quick demo now? (y/n): " run_demo
if [[ $run_demo == "y" || $run_demo == "Y" ]]; then
    python train_and_visualize.py --train --timesteps 10000 --analyze
fi

print_header "Setup complete"
echo "Whenever you want to run the program, activate the virtual environment first:"
echo "source venv/bin/activate"
echo ""
echo "Then run your command, for example:"
echo "python train_and_visualize.py --full_pipeline" 
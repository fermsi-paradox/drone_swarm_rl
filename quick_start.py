#!/usr/bin/env python3
"""
Quick Start for Military Drone Swarm RL

This script helps new users get started with the military drone swarm project
by setting up directories, checking dependencies, and running a simple demo.
"""

import os
import sys
import subprocess
import platform
import shutil
import time

def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: This project requires Python 3.8 or newer.")
        sys.exit(1)
    else:
        print("✓ Python version is compatible.")

def ensure_directories():
    """Ensure all required directories exist."""
    print_header("Setting Up Directory Structure")
    
    # Create required directories
    directories = [
        "src/environments",
        "src/training",
        "src/configs",
        "models",
        "logs",
        "output/visualization"
    ]
    
    for directory in directories:
        dir_path = os.path.join(os.getcwd(), directory)
        if not os.path.exists(dir_path):
            print(f"Creating directory: {directory}")
            os.makedirs(dir_path, exist_ok=True)
    
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/environments/__init__.py",
        "src/training/__init__.py"
    ]
    
    for init_file in init_files:
        file_path = os.path.join(os.getcwd(), init_file)
        if not os.path.exists(file_path):
            print(f"Creating file: {init_file}")
            with open(file_path, 'w') as f:
                f.write("# Auto-generated __init__.py file\n")
    
    print("✓ Directory structure is set up.")

def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("Checking Dependencies")
    
    # Try importing key packages
    dependencies = [
        "numpy", "scipy", "matplotlib", "gymnasium", "yaml", 
        "torch", "stable_baselines3", "pygame", "OpenGL"
    ]
    
    missing_deps = []
    
    for dep in dependencies:
        try:
            if dep == "yaml":
                __import__("pyyaml")
            elif dep == "OpenGL":
                __import__("OpenGL")
            else:
                __import__(dep)
            print(f"✓ {dep} is installed")
        except ImportError:
            print(f"✗ {dep} is not installed")
            missing_deps.append(dep)
    
    if missing_deps:
        print("\nSome dependencies are missing. Would you like to install them? (y/n)")
        choice = input().lower()
        
        if choice.startswith('y'):
            # Map package names to pip install names if they differ
            pip_names = {
                "yaml": "pyyaml",
                "OpenGL": "PyOpenGL"
            }
            
            for dep in missing_deps:
                pip_name = pip_names.get(dep, dep)
                print(f"Installing {pip_name}...")
                subprocess.run([sys.executable, "-m", "pip", "install", pip_name])
        else:
            print("Please install the missing dependencies before continuing.")
            print("You can run: pip install -r requirements.txt")
    else:
        print("✓ All dependencies are installed.")

def run_simple_demo():
    """Run a simple demonstration."""
    print_header("Running Simple Demo")
    
    print("This will run a small training job (10,000 timesteps) to verify everything works.")
    print("Would you like to continue? (y/n)")
    
    choice = input().lower()
    if not choice.startswith('y'):
        print("Demo skipped. You can run the demo later with:")
        print("python train_and_visualize.py --train --timesteps 10000")
        return
    
    print("Starting simple training job...")
    
    try:
        # Run a small training job
        cmd = [sys.executable, "train_and_visualize.py", "--train", "--timesteps", "10000"]
        subprocess.run(cmd, check=True)
        
        print("✓ Demo completed successfully.")
        print("You can now run the full pipeline with: python train_and_visualize.py --full_pipeline")
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
        print("Please check the log files for more information.")

def show_next_steps():
    """Show the next steps for the user."""
    print_header("Next Steps")
    
    print("Your military drone swarm reinforcement learning environment is set up.")
    print("Here are some things you can do next:")
    print()
    print("1. Train a model:")
    print("   python train_and_visualize.py --train --timesteps 500000")
    print()
    print("2. Analyze model performance:")
    print("   python train_and_visualize.py --analyze --model_path path/to/model.zip")
    print()
    print("3. Visualize model behavior:")
    print("   python train_and_visualize.py --visualize --model_path path/to/model.zip")
    print()
    print("4. Run the full pipeline (train, analyze, visualize):")
    print("   python train_and_visualize.py --full_pipeline")
    print()
    print("5. Modify the configuration file:")
    print("   src/configs/military_combat.yaml")

def main():
    """Main function to run the quick start script."""
    print_header("Military Drone Swarm RL - Quick Start")
    
    # Check if we're in the right directory
    if not os.path.exists("setup_directories.sh") and not os.path.exists("setup_directories.bat"):
        print("WARNING: You might not be in the drone_swarm_rl directory.")
        print("This script should be run from the root of the project.")
        print("Would you like to continue anyway? (y/n)")
        
        choice = input().lower()
        if not choice.startswith('y'):
            print("Exiting. Please run this script from the drone_swarm_rl directory.")
            sys.exit(1)
    
    # Run setup steps
    check_python_version()
    ensure_directories()
    check_dependencies()
    
    # Offer to run a demo
    run_simple_demo()
    
    # Show next steps
    show_next_steps()
    
    print("\nSetup complete! You're ready to start training military drone swarms.")

if __name__ == "__main__":
    main() 
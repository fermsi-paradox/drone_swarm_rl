@echo off
:: Directory setup script for drone_swarm_rl project (Windows)
:: This script creates the necessary directory structure and ensures files are in the right place

echo Setting up directory structure for drone_swarm_rl project...

:: Create main directory structure
mkdir src\environments src\training src\configs models logs output\visualization 2>nul

:: Create empty __init__.py files for Python packages
echo. > src\__init__.py
echo. > src\environments\__init__.py
echo. > src\training\__init__.py

echo Directory structure setup complete.
echo.
echo Your project structure should now look like this:
echo drone_swarm_rl\
echo ├── requirements.txt
echo ├── train_military_model.py
echo ├── visualize_combat.py
echo ├── train_and_visualize.py
echo ├── models\
echo ├── logs\
echo ├── output\
echo └── src\
echo     ├── __init__.py
echo     ├── environments\
echo     │   ├── __init__.py
echo     │   └── drone_env.py
echo     ├── training\
echo     │   ├── __init__.py
echo     │   └── train.py
echo     └── configs\
echo         └── military_combat.yaml
echo.
echo You can now run the following commands from the drone_swarm_rl directory:
echo python train_and_visualize.py --full_pipeline --timesteps 10000
echo or
echo python train_and_visualize.py --train --timesteps 10000

pause 
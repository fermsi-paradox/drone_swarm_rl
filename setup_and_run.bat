@echo off
:: Setup and run script for Military Drone Swarm RL project (Windows version)
:: This script helps new users get started quickly

echo =============================================
echo Checking Python installation
echo =============================================
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or newer.
    exit /b 1
)

:: Check if we're in the right directory
if not exist requirements.txt (
    echo Error: requirements.txt not found.
    echo Please run this script from the drone_swarm_rl directory.
    exit /b 1
)

echo =============================================
echo Setting up virtual environment
echo =============================================
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

echo =============================================
echo Installing dependencies
echo =============================================
echo Installing requirements...
pip install -r requirements.txt

echo =============================================
echo Checking system dependencies
echo =============================================
echo Windows detected. Make sure you have up-to-date graphics drivers.
echo If you encounter OpenGL errors, please update your graphics drivers.

:: All set, run the program
echo =============================================
echo Ready to run!
echo =============================================
echo Your environment is set up. You can run the program with:
echo python train_and_visualize.py --full_pipeline --timesteps 100000
echo.
echo Or run a quick demo with:
echo python train_and_visualize.py --train --timesteps 10000 --analyze
echo.

:: Ask if user wants to run the demo
set /p run_demo="Would you like to run a quick demo now? (y/n): "
if /i "%run_demo%"=="y" (
    python train_and_visualize.py --train --timesteps 10000 --analyze
)

echo =============================================
echo Setup complete
echo =============================================
echo Whenever you want to run the program, activate the virtual environment first:
echo call venv\Scripts\activate
echo.
echo Then run your command, for example:
echo python train_and_visualize.py --full_pipeline

:: Keep the window open
pause 
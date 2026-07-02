@echo off
REM _max_cyan_ — project_mxsa — install trainer environment on Windows

setlocal
cd /d "%~dp0\.."

echo ============================================
echo   Setting up Simba Trainer on Windows...
echo ============================================
echo.

REM Check if python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ and make sure "Add Python to PATH" is checked.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "trainer\venv" (
    echo Creating virtual environment in trainer\venv...
    python -m venv trainer\venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call trainer\venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip wheel setuptools

echo Installing dependencies from scripts\requirements_trainer.txt...
pip install -r scripts\requirements_trainer.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install some dependencies.
    pause
    exit /b 1
)

echo.
echo =========================================================
echo   Trainer setup complete!
echo   You can start it using scripts\start_trainer.bat
echo =========================================================
pause

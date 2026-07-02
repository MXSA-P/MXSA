@echo off
REM _max_cyan_ — project_mxsa — start trainer interface on Windows

setlocal
cd /d "%~dp0\.."

echo ============================================
echo   simba — start trainer interface
echo   _max_cyan_ — project_mxsa
echo ============================================
echo.

REM Check if virtual environment exists
if not exist "trainer\venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment trainer\venv not found!
    echo Please run scripts\install_trainer.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call trainer\venv\Scripts\activate.bat

echo starting simba trainer...
echo open http://localhost:5000 in your browser
echo.

REM Launch browser after 2 seconds asynchronously
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:5000"

python -m trainer.app %*
if %errorlevel% neq 0 (
    if %errorlevel% neq -1073741510 (
        echo.
        echo [ERROR] Trainer server exited with code %errorlevel%.
        pause
    )
)

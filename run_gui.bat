@echo off
REM LLMOCR Desktop Application Launcher for Windows

echo ============================================
echo   LLMOCR Desktop Application
echo   Korean OCR Analysis Platform
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    echo Please install Python 3.8 or higher from https://www.python.org
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if PyQt5 is installed
echo Checking dependencies...
python -c "import PyQt5" >nul 2>&1
if errorlevel 1 (
    echo.
    echo PyQt5 is not installed!
    echo.
    set /p INSTALL="Do you want to install PyQt5 now? (y/n): "
    if /i "%INSTALL%"=="y" (
        echo Installing PyQt5...
        pip install PyQt5
    ) else (
        echo Cannot run without PyQt5. Exiting...
        pause
        exit /b 1
    )
)

REM Check other dependencies
python -c "import PIL" >nul 2>&1
if errorlevel 1 (
    echo Installing additional dependencies...
    pip install pillow numpy python-Levenshtein
)

echo.
echo Starting LLMOCR Desktop Application...
echo.

REM Run the application
python gui\desktop_app.py

pause

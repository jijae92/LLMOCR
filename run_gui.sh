#!/bin/bash
# LLMOCR Desktop Application Launcher

echo "============================================"
echo "  LLMOCR Desktop Application"
echo "  Korean OCR Analysis Platform"
echo "============================================"
echo ""

# Detect virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
    PYTHON_CMD="python"
    PIP_CMD="pip"
elif [ -d ".venv312" ]; then
    echo "✓ Found .venv312, activating..."
    source .venv312/bin/activate
    PYTHON_CMD="python"
    PIP_CMD="pip"
elif [ -d ".venv" ]; then
    echo "✓ Found .venv, activating..."
    source .venv/bin/activate
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo "Using system Python"
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

# Check if Python is installed
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if PyQt5 is installed
echo "Checking dependencies..."
if ! $PYTHON_CMD -c "import PyQt5" 2>/dev/null; then
    echo ""
    echo "PyQt5 is not installed!"
    echo ""
    read -p "Do you want to install PyQt5 now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing PyQt5..."
        $PIP_CMD install PyQt5
    else
        echo "Cannot run without PyQt5. Exiting..."
        exit 1
    fi
fi

# Check other dependencies
if ! $PYTHON_CMD -c "import PIL" 2>/dev/null; then
    echo "Installing additional dependencies..."
    $PIP_CMD install pillow numpy python-Levenshtein
fi

echo ""
echo "Starting LLMOCR Desktop Application..."
echo ""

# Run the application
$PYTHON_CMD gui/desktop_app.py

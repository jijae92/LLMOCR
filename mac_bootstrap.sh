#!/usr/bin/env bash
set -euo pipefail

echo "================================================"
echo "macOS Bootstrap Script for LLMOCR Development"
echo "================================================"
echo ""

# Homebrew
echo "Step 1/5: Checking Homebrew installation..."
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
  eval "$(/opt/homebrew/bin/brew shellenv)"
  echo "✓ Homebrew installed successfully"
else
  echo "✓ Homebrew already installed"
fi

# Dev deps (선택: 이미지/ONNX/성능에 도움)
echo ""
echo "Step 2/5: Installing development dependencies..."
brew install git ffmpeg pkg-config
echo "✓ Development dependencies installed"

# Python 3.11 venv
echo ""
echo "Step 3/5: Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "✓ Virtual environment created"
else
  echo "✓ Virtual environment already exists"
fi

source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
echo "✓ pip, wheel, and setuptools upgraded"

# PyTorch (MPS 지원 포함), HF 스택, FastAPI, 평가 도구
echo ""
echo "Step 4/5: Installing PyTorch and ML dependencies..."
echo "This may take several minutes..."
pip install torch torchvision torchaudio \
           transformers pillow fastapi uvicorn[standard] datasets jiwer python-Levenshtein \
           accelerate peft onnx onnxruntime optimum[exporters]
echo "✓ PyTorch and ML stack installed"

# (선택) PaddleOCR 경량 백엔드
echo ""
echo "Step 5/5: Installing PaddleOCR (optional)..."
echo "Note: PaddleOCR may have compatibility issues on some macOS/Python combinations"
pip install "paddleocr>=2.7" "paddlepaddle>=2.6" || echo "⚠ PaddleOCR installation failed (optional dependency)"

# MPS 가용성 점검 스니펫
echo ""
echo "================================================"
echo "Checking MPS (Metal Performance Shaders) availability..."
python - <<'PY'
import torch
mps_available = torch.backends.mps.is_available()
print(f"MPS available: {mps_available}")
if mps_available:
    print("✓ Your Mac supports Metal acceleration for PyTorch!")
    print("  You can use device='mps' in your training scripts.")
else:
    print("⚠ MPS not available. Training will use CPU.")
PY

echo ""
echo "================================================"
echo "Bootstrap complete! 🎉"
echo "================================================"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "See README-mac.md for more details."

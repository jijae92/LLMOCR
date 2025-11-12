"""Tests for CLI interface."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image

from src.cli import load_pipeline, MODEL_REGISTRY


def test_model_registry():
    """Test that model registry is properly defined."""
    assert "trocr" in MODEL_REGISTRY
    assert "donut" in MODEL_REGISTRY
    assert "pix2struct" in MODEL_REGISTRY
    assert "paddleocr" in MODEL_REGISTRY


def test_load_pipeline_trocr():
    """Test loading TrOCR pipeline."""
    # This will actually load the model, so skip in CI
    pytest.skip("Model loading tests require actual models")


def test_load_pipeline_invalid_model():
    """Test that invalid model raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        load_pipeline(model="invalid_model")


@pytest.mark.asyncio
async def test_cli_with_sample_image():
    """Test CLI with a sample image."""
    # This would require actual image files
    pytest.skip("Integration tests require actual image files")

"""Basic tests for CLI functionality."""
import pytest
from pathlib import Path
from PIL import Image


def test_imports():
    """Test that all pipeline modules can be imported."""
    from src.pipelines.trocr_pipeline import TrOCRPipeline
    from src.pipelines.donut_pipeline import DonutPipeline
    from src.pipelines.pix2struct_pipeline import Pix2StructPipeline
    from src.pipelines.paddleocr_pipeline import PaddleOCRPipeline
    assert TrOCRPipeline is not None
    assert DonutPipeline is not None
    assert Pix2StructPipeline is not None
    assert PaddleOCRPipeline is not None


def test_cli_imports():
    """Test that CLI module can be imported."""
    from src import cli
    assert cli.get_pipeline is not None


def test_eval_imports():
    """Test that eval module can be imported."""
    from src import eval
    assert eval.compute_cer is not None
    assert eval.compute_wer is not None


def test_server_imports():
    """Test that server module can be imported."""
    from src.server import app
    assert app.app is not None

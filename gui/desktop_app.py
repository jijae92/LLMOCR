#!/usr/bin/env python3
"""
LLMOCR Desktop Application - Standalone Native GUI

A complete desktop application for Korean OCR analysis and operations.
Built with PyQt5 for native performance and user experience.

Features:
- Dataset Management
- Data Processing
- Benchmark Execution
- Continuous Learning
- Single Image OCR
- Error Analysis
- Audit Logs
- Batch Processing
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QTextEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QProgressBar, QFileDialog,
    QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout, QSplitter,
    QMessageBox, QScrollArea, QSlider, QFrame, QGridLayout, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QTextCursor

import json
import time
import subprocess
import tempfile
from datetime import datetime
from PIL import Image
import numpy as np

# Import local modules
try:
    from tools.audit_logging.audit_logger import (
        AuditLogger, ModelInfo, PreprocessingParams, EngineType
    )
    from tools.error_analysis.error_analyzer import ErrorAnalyzer
    from utils.visualization.bbox_visualizer import BBoxVisualizer
except ImportError as e:
    print(f"Warning: Could not import local modules: {e}")


class WorkerThread(QThread):
    """Worker thread for long-running operations"""
    finished = pyqtSignal(bool, str, str)
    progress = pyqtSignal(int, str)

    def __init__(self, command, description):
        super().__init__()
        self.command = command
        self.description = description

    def run(self):
        try:
            result = subprocess.run(
                self.command,
                capture_output=True,
                text=True,
                timeout=3600
            )
            success = result.returncode == 0
            self.finished.emit(success, result.stdout, result.stderr)
        except Exception as e:
            self.finished.emit(False, "", str(e))


class DatasetManagementTab(QWidget):
    """Dataset Management Tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        # Container widget
        container = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Dataset Management")
        title.setFont(QFont("Arial", 24, QFont.Bold))  # Increased from 16
        layout.addWidget(title)

        # SynthDoG-ko Section
        synthdog_group = QGroupBox("Download SynthDoG-ko")
        synthdog_layout = QFormLayout()

        self.synthdog_output = QLineEdit("datasets/raw/synthdog_ko")
        self.synthdog_limit = QSpinBox()
        self.synthdog_limit.setRange(10, 100000)
        self.synthdog_limit.setValue(1000)
        self.synthdog_limit.setSingleStep(100)
        self.synthdog_start = QSpinBox()
        self.synthdog_start.setRange(0, 1000000)
        self.synthdog_start.setValue(0)

        synthdog_layout.addRow("Output Directory:", self.synthdog_output)
        synthdog_layout.addRow("Sample Limit:", self.synthdog_limit)
        synthdog_layout.addRow("Start Index:", self.synthdog_start)

        self.download_btn = QPushButton("📥 Download SynthDoG-ko")
        self.download_btn.clicked.connect(self.download_synthdog)
        synthdog_layout.addRow(self.download_btn)

        synthdog_group.setLayout(synthdog_layout)
        layout.addWidget(synthdog_group)

        # AI-Hub Section
        aihub_group = QGroupBox("Download AI-Hub Dataset")
        aihub_layout = QFormLayout()

        info_label = QLabel(
            "AI-Hub datasets require manual download.\n"
            "1. Visit https://aihub.or.kr\n"
            "2. Login and request dataset access\n"
            "3. Download and extract to datasets/raw/"
        )
        info_label.setWordWrap(True)
        aihub_layout.addRow(info_label)

        self.aihub_dataset = QComboBox()
        self.aihub_dataset.addItems(["admin_docs", "korean_fonts"])
        aihub_layout.addRow("Dataset:", self.aihub_dataset)

        self.aihub_instructions_btn = QPushButton("📋 Show Instructions")
        self.aihub_instructions_btn.clicked.connect(self.show_aihub_instructions)
        aihub_layout.addRow(self.aihub_instructions_btn)

        aihub_group.setLayout(aihub_layout)
        layout.addWidget(aihub_group)

        # Log Output
        log_group = QGroupBox("Output Log")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        
        # Set container layout and add to scroll
        container.setLayout(layout)
        scroll.setWidget(container)
        
        # Main layout with scroll
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def download_synthdog(self):
        output_dir = self.synthdog_output.text()
        limit = self.synthdog_limit.value()
        start_idx = self.synthdog_start.value()

        cmd = [
            "python3", "datasets/scripts/download_synthdog_ko.py",
            "--output_dir", output_dir,
            "--limit", str(limit),
            "--start_idx", str(start_idx),
        ]

        self.log_output.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting download...")
        self.log_output.append(f"Command: {' '.join(cmd)}\n")

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.download_btn.setEnabled(False)

        self.worker = WorkerThread(cmd, "Download SynthDoG-ko")
        self.worker.finished.connect(self.on_download_finished)
        self.worker.start()

    def on_download_finished(self, success, stdout, stderr):
        self.progress_bar.setVisible(False)
        self.download_btn.setEnabled(True)

        if success:
            self.log_output.append("✓ Download completed successfully!\n")
            self.log_output.append(stdout)
            QMessageBox.information(self, "Success", "Dataset downloaded successfully!")
        else:
            self.log_output.append("✗ Download failed!\n")
            self.log_output.append(stderr)
            QMessageBox.critical(self, "Error", f"Download failed:\n{stderr}")

    def show_aihub_instructions(self):
        dataset = self.aihub_dataset.currentText()
        cmd = [
            "python3", "datasets/scripts/download_aihub.py",
            "--dataset", dataset,
            "--output_dir", f"datasets/raw/aihub_{dataset}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        self.log_output.append(f"\n{result.stdout}\n")


class DataProcessingTab(QWidget):
    """Data Processing Tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        # Container widget
        container = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Data Processing")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        layout.addWidget(title)

        # Clean Dataset Section
        clean_group = QGroupBox("1. Clean Dataset")
        clean_layout = QFormLayout()

        self.clean_source = QLineEdit("datasets/raw/synthdog_ko")
        self.clean_output = QLineEdit("datasets/processed/synthdog_ko_clean")
        self.clean_min_length = QSpinBox()
        self.clean_min_length.setRange(1, 100)
        self.clean_min_length.setValue(5)
        self.clean_max_length = QSpinBox()
        self.clean_max_length.setRange(100, 10000)
        self.clean_max_length.setValue(1000)
        self.clean_min_dim = QSpinBox()
        self.clean_min_dim.setRange(16, 512)
        self.clean_min_dim.setValue(32)
        self.clean_blur_threshold = QDoubleSpinBox()
        self.clean_blur_threshold.setRange(10.0, 500.0)
        self.clean_blur_threshold.setValue(100.0)
        self.clean_copy_images = QCheckBox("Copy/Link Images")
        self.clean_copy_images.setChecked(True)

        clean_layout.addRow("Source Directory:", self.clean_source)
        clean_layout.addRow("Output Directory:", self.clean_output)
        clean_layout.addRow("Min Text Length:", self.clean_min_length)
        clean_layout.addRow("Max Text Length:", self.clean_max_length)
        clean_layout.addRow("Min Image Dimension:", self.clean_min_dim)
        clean_layout.addRow("Blur Threshold:", self.clean_blur_threshold)
        clean_layout.addRow(self.clean_copy_images)

        self.clean_btn = QPushButton("🧹 Clean Dataset")
        self.clean_btn.clicked.connect(self.clean_dataset)
        clean_layout.addRow(self.clean_btn)

        clean_group.setLayout(clean_layout)
        layout.addWidget(clean_group)

        # Create Splits Section
        split_group = QGroupBox("2. Create Train/Val/Test Splits")
        split_layout = QFormLayout()

        self.split_input = QLineEdit("datasets/processed/synthdog_ko_clean")
        self.train_ratio = QDoubleSpinBox()
        self.train_ratio.setRange(0.0, 1.0)
        self.train_ratio.setValue(0.8)
        self.train_ratio.setSingleStep(0.05)
        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setRange(0.0, 1.0)
        self.val_ratio.setValue(0.1)
        self.val_ratio.setSingleStep(0.05)
        self.split_seed = QSpinBox()
        self.split_seed.setRange(0, 10000)
        self.split_seed.setValue(42)

        split_layout.addRow("Input Directory:", self.split_input)
        split_layout.addRow("Train Ratio:", self.train_ratio)
        split_layout.addRow("Validation Ratio:", self.val_ratio)
        split_layout.addRow("Random Seed:", self.split_seed)

        self.split_btn = QPushButton("✂️ Create Splits")
        self.split_btn.clicked.connect(self.create_splits)
        split_layout.addRow(self.split_btn)

        split_group.setLayout(split_layout)
        layout.addWidget(split_group)

        # Log Output
        log_group = QGroupBox("Output Log")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        
        # Set container layout and add to scroll
        container.setLayout(layout)
        scroll.setWidget(container)
        
        # Main layout with scroll
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def clean_dataset(self):
        cmd = [
            "python3", "datasets/scripts/clean_data.py",
            "--source", self.clean_source.text(),
            "--output", self.clean_output.text(),
            "--min_length", str(self.clean_min_length.value()),
            "--max_length", str(self.clean_max_length.value()),
            "--min_dimension", str(self.clean_min_dim.value()),
            "--blur_threshold", str(self.clean_blur_threshold.value()),
        ]

        if self.clean_copy_images.isChecked():
            cmd.append("--copy_images")

        self.log_output.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cleaning dataset...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.clean_btn.setEnabled(False)

        self.worker = WorkerThread(cmd, "Clean Dataset")
        self.worker.finished.connect(self.on_clean_finished)
        self.worker.start()

    def on_clean_finished(self, success, stdout, stderr):
        self.progress_bar.setVisible(False)
        self.clean_btn.setEnabled(True)

        if success:
            self.log_output.append("✓ Dataset cleaned successfully!\n")
            self.log_output.append(stdout)
            QMessageBox.information(self, "Success", "Dataset cleaned successfully!")
        else:
            self.log_output.append("✗ Cleaning failed!\n")
            self.log_output.append(stderr)
            QMessageBox.critical(self, "Error", f"Cleaning failed:\n{stderr}")

    def create_splits(self):
        if self.train_ratio.value() + self.val_ratio.value() >= 1.0:
            QMessageBox.warning(self, "Invalid Ratios", "Train + Val ratio must be < 1.0")
            return

        cmd = [
            "python3", "datasets/scripts/create_splits.py",
            "--input", self.split_input.text(),
            "--train_ratio", str(self.train_ratio.value()),
            "--val_ratio", str(self.val_ratio.value()),
            "--seed", str(self.split_seed.value()),
        ]

        self.log_output.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] Creating splits...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.split_btn.setEnabled(False)

        self.worker = WorkerThread(cmd, "Create Splits")
        self.worker.finished.connect(self.on_split_finished)
        self.worker.start()

    def on_split_finished(self, success, stdout, stderr):
        self.progress_bar.setVisible(False)
        self.split_btn.setEnabled(True)

        if success:
            self.log_output.append("✓ Splits created successfully!\n")
            self.log_output.append(stdout)
            QMessageBox.information(self, "Success", "Splits created successfully!")
        else:
            self.log_output.append("✗ Split creation failed!\n")
            self.log_output.append(stderr)
            QMessageBox.critical(self, "Error", f"Split creation failed:\n{stderr}")


class SingleImageOCRTab(QWidget):
    """Single Image OCR Tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.init_ui()

    def init_ui(self):
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        # Container widget
        container = QWidget()
        layout = QHBoxLayout()

        # Left Panel - Input
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        title = QLabel("Single Image OCR")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        left_layout.addWidget(title)

        # Image Selection
        img_group = QGroupBox("Image")
        img_layout = QVBoxLayout()

        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setFrameStyle(QFrame.Box)
        self.image_label.setScaledContents(True)
        img_layout.addWidget(self.image_label)

        self.select_image_btn = QPushButton("📁 Select Image")
        self.select_image_btn.clicked.connect(self.select_image)
        img_layout.addWidget(self.select_image_btn)

        img_group.setLayout(img_layout)
        left_layout.addWidget(img_group)

        # Ground Truth
        gt_group = QGroupBox("Ground Truth (Optional)")
        gt_layout = QVBoxLayout()
        self.ground_truth = QTextEdit()
        self.ground_truth.setMaximumHeight(80)
        gt_layout.addWidget(self.ground_truth)
        gt_group.setLayout(gt_layout)
        left_layout.addWidget(gt_group)

        # Process Button
        self.process_btn = QPushButton("🚀 Process Image")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        left_layout.addWidget(self.process_btn)

        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel)

        # Right Panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        results_title = QLabel("Results")
        results_title.setFont(QFont("Arial", 20, QFont.Bold))
        right_layout.addWidget(results_title)

        # Prediction
        pred_group = QGroupBox("Prediction")
        pred_layout = QVBoxLayout()
        self.prediction_text = QTextEdit()
        self.prediction_text.setReadOnly(True)
        self.prediction_text.setMaximumHeight(100)
        pred_layout.addWidget(self.prediction_text)
        pred_group.setLayout(pred_layout)
        right_layout.addWidget(pred_group)

        # Metrics
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QGridLayout()

        self.confidence_label = QLabel("Confidence: -")
        self.inference_time_label = QLabel("Inference Time: -")
        self.cer_label = QLabel("CER: -")

        metrics_layout.addWidget(self.confidence_label, 0, 0)
        metrics_layout.addWidget(self.inference_time_label, 0, 1)
        metrics_layout.addWidget(self.cer_label, 1, 0)

        metrics_group.setLayout(metrics_layout)
        right_layout.addWidget(metrics_group)

        # Word Details
        words_group = QGroupBox("Word-Level Details")
        words_layout = QVBoxLayout()
        self.words_text = QTextEdit()
        self.words_text.setReadOnly(True)
        words_layout.addWidget(self.words_text)
        words_group.setLayout(words_layout)
        right_layout.addWidget(words_group)

        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel)

        # Set container layout and add to scroll
        container.setLayout(layout)
        scroll.setWidget(container)

        # Main layout with scroll
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if file_path:
            self.image = Image.open(file_path).convert('RGB')

            # Display image
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

            self.process_btn.setEnabled(True)

    def process_image(self):
        if self.image is None:
            return

        # Mock OCR processing
        time.sleep(0.5)  # Simulate processing

        # Mock results
        prediction = "한국어 OCR 테스트 문서입니다"
        confidence = 0.87
        inference_time = 123.45

        self.prediction_text.setText(prediction)
        self.confidence_label.setText(f"Confidence: {confidence:.2%}")
        self.inference_time_label.setText(f"Inference Time: {inference_time:.1f} ms")

        # Calculate CER if ground truth provided
        gt = self.ground_truth.toPlainText().strip()
        if gt:
            import Levenshtein
            cer = Levenshtein.distance(prediction, gt) / len(gt)
            self.cer_label.setText(f"CER: {cer:.2%}")
        else:
            self.cer_label.setText("CER: N/A")

        # Mock word-level details
        words = [
            {'text': '한국어', 'confidence': 0.95},
            {'text': 'OCR', 'confidence': 0.92},
            {'text': '테스트', 'confidence': 0.65},
            {'text': '문서입니다', 'confidence': 0.88},
        ]

        words_html = ""
        for word in words:
            color = "green" if word['confidence'] >= 0.7 else "red"
            words_html += f"<span style='color: {color};'>● <b>{word['text']}</b>: {word['confidence']:.2%}</span><br>"

        self.words_text.setHtml(words_html)

        QMessageBox.information(self, "Success", "Image processed successfully!")


class MainWindow(QMainWindow):
    """Main Application Window"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("LLMOCR - Korean OCR Platform")
        self.setGeometry(100, 100, 1200, 800)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        layout = QVBoxLayout()

        # Header
        header = QLabel("📝 LLMOCR - Complete Korean OCR Platform")
        header.setFont(QFont("Arial", 24, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("padding: 20px; background-color: #1e1e1e; color: #61dafb;")
        layout.addWidget(header)

        # Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        # Add Tabs
        self.tabs.addTab(DatasetManagementTab(), "🗂️ Dataset Management")
        self.tabs.addTab(DataProcessingTab(), "🔄 Data Processing")
        self.tabs.addTab(self.create_placeholder_tab("Benchmark Execution"), "🚀 Benchmarks")
        self.tabs.addTab(self.create_placeholder_tab("Continuous Learning"), "🔁 Learning")
        self.tabs.addTab(SingleImageOCRTab(), "🖼️ Single Image")
        self.tabs.addTab(self.create_placeholder_tab("Error Analysis"), "📊 Analysis")
        self.tabs.addTab(self.create_placeholder_tab("Audit Logs"), "📋 Logs")
        self.tabs.addTab(self.create_placeholder_tab("Batch Processing"), "⚡ Batch")

        layout.addWidget(self.tabs)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        central_widget.setLayout(layout)

        # Apply Dark Mode stylesheet
        self.setStyleSheet("""
            /* Main Window - Dark Background */
            QMainWindow {
                background-color: #1e1e1e;
            }

            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }

            /* Group Boxes - Dark Theme */
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 15px;
                background-color: #252525;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px 0 5px;
                color: #61dafb;
            }

            /* Buttons - Modern Blue */
            QPushButton {
                background-color: #0d7377;
                color: #ffffff;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #14b1b8;
            }
            QPushButton:pressed {
                background-color: #0a5c5f;
            }
            QPushButton:disabled {
                background-color: #3d3d3d;
                color: #666666;
            }

            /* Input Fields - Dark */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 8px;
                selection-background-color: #0d7377;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #61dafb;
            }

            /* ComboBox Drop-down */
            QComboBox::drop-down {
                border: none;
                background-color: #3d3d3d;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #e0e0e0;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: #e0e0e0;
                selection-background-color: #0d7377;
                border: 1px solid #3d3d3d;
            }

            /* Text Edit - Dark */
            QTextEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 8px;
                selection-background-color: #0d7377;
            }
            QTextEdit:focus {
                border: 2px solid #61dafb;
            }

            /* Progress Bar - Cyan */
            QProgressBar {
                background-color: #2d2d2d;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                text-align: center;
                color: #e0e0e0;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #0d7377, stop:1 #14b1b8);
                border-radius: 3px;
            }

            /* Tabs - Dark Modern */
            QTabWidget::pane {
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                background-color: #252525;
                top: -2px;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #a0a0a0;
                padding: 12px 24px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border: 2px solid #3d3d3d;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #0d7377;
                color: #ffffff;
                border: 2px solid #0d7377;
                border-bottom: none;
            }
            QTabBar::tab:hover {
                background-color: #3d3d3d;
                color: #e0e0e0;
            }

            /* CheckBox - Dark */
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #3d3d3d;
                border-radius: 4px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #0d7377;
                border: 2px solid #0d7377;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNNyAxMEw5IDEyTDE0IDciIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIvPjwvc3ZnPg==);
            }
            QCheckBox::indicator:hover {
                border: 2px solid #61dafb;
            }

            /* Labels */
            QLabel {
                color: #e0e0e0;
                background-color: transparent;
            }

            /* Status Bar - Dark */
            QStatusBar {
                background-color: #1e1e1e;
                color: #61dafb;
                border-top: 1px solid #3d3d3d;
            }

            /* Scroll Bars - Dark */
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 14px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background-color: #4d4d4d;
                border-radius: 7px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #0d7377;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #2d2d2d;
                height: 14px;
                border-radius: 7px;
            }
            QScrollBar::handle:horizontal {
                background-color: #4d4d4d;
                border-radius: 7px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #0d7377;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }

            /* Frame - Image Container */
            QFrame {
                background-color: #2d2d2d;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
            }
        """)

    def create_placeholder_tab(self, name):
        """Create a placeholder tab for features to be implemented"""
        widget = QWidget()
        layout = QVBoxLayout()

        label = QLabel(f"{name}\n\nThis feature is available in the Streamlit version.\nDesktop implementation coming soon!")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 16))
        label.setWordWrap(True)

        layout.addWidget(label)
        widget.setLayout(layout)
        return widget


def main():
    app = QApplication(sys.argv)

    # Set application-wide font
    app.setFont(QFont("Arial", 13))

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Complete LLMOCR GUI - Korean OCR Analysis & Operations Platform

Features:
- Dataset Management: Download, clean, and process datasets
- Benchmark Execution: Run and compare OCR benchmarks
- Continuous Learning: Automated training and regression testing
- Single Image Processing: Real-time OCR with visualization
- Error Analysis: Identify and analyze common errors
- Audit Logging: Track all operations and performance
- Batch Processing: Process multiple images efficiently
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import time
import json
import subprocess
import tempfile
import shutil
from datetime import datetime
from PIL import Image
import numpy as np
import pandas as pd

# Import local modules
try:
    from tools.audit_logging.audit_logger import (
        AuditLogger, ModelInfo, PreprocessingParams, EngineType
    )
    from tools.error_analysis.error_analyzer import ErrorAnalyzer
    from utils.visualization.bbox_visualizer import BBoxVisualizer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


# Page config
st.set_page_config(
    page_title="LLMOCR - Complete Korean OCR Platform",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize session state
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

if 'audit_logger' not in st.session_state:
    st.session_state.audit_logger = AuditLogger(log_dir="logs/audit")

if 'visualizer' not in st.session_state:
    st.session_state.visualizer = BBoxVisualizer()

if 'datasets_downloaded' not in st.session_state:
    st.session_state.datasets_downloaded = []


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preprocess_image(
    image: Image.Image,
    dpi_scale: float = 1.0,
    denoise: bool = False,
    sharpen: bool = False,
) -> Image.Image:
    """Preprocess image with optional enhancements."""
    import cv2

    # Convert to numpy array
    img_array = np.array(image)

    # Scale for DPI
    if dpi_scale != 1.0:
        new_size = (
            int(img_array.shape[1] * dpi_scale),
            int(img_array.shape[0] * dpi_scale)
        )
        img_array = cv2.resize(img_array, new_size, interpolation=cv2.INTER_CUBIC)

    # Denoise
    if denoise:
        img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

    # Sharpen
    if sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_array = cv2.filter2D(img_array, -1, kernel)

    return Image.fromarray(img_array)


def mock_ocr_inference(
    image: Image.Image,
    model_name: str = "mock_model",
    engine: EngineType = EngineType.PYTORCH,
) -> dict:
    """
    Mock OCR inference (replace with real model).

    Returns prediction with bounding boxes and confidence scores.
    """
    # Simulate processing time
    time.sleep(0.5)

    # Mock results
    results = {
        'text': '한국어 OCR 테스트 문서입니다',
        'confidence': 0.87,
        'words': [
            {
                'text': '한국어',
                'box': [50, 50, 150, 80],
                'confidence': 0.95,
            },
            {
                'text': 'OCR',
                'box': [160, 50, 220, 80],
                'confidence': 0.92,
            },
            {
                'text': '테스트',
                'box': [230, 50, 320, 80],
                'confidence': 0.65,
            },
            {
                'text': '문서입니다',
                'box': [50, 90, 180, 120],
                'confidence': 0.88,
            },
        ],
        'inference_time_ms': 123.45,
    }

    return results


def process_single_image(
    image: Image.Image,
    preprocessing_params: PreprocessingParams,
    model_info: ModelInfo,
    ground_truth: str = None,
) -> dict:
    """Process a single image through OCR pipeline."""

    # Preprocess
    processed_image = preprocess_image(
        image,
        dpi_scale=preprocessing_params.dpi_scale,
        denoise=preprocessing_params.denoise,
        sharpen=preprocessing_params.sharpen,
    )

    # Run OCR (mock)
    start_time = time.perf_counter()
    ocr_results = mock_ocr_inference(processed_image, model_info.model_name, model_info.engine)
    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000

    # Calculate CER/WER if ground truth provided
    cer = None
    wer = None
    if ground_truth:
        import Levenshtein
        cer = Levenshtein.distance(ocr_results['text'], ground_truth) / len(ground_truth)

        pred_words = ocr_results['text'].split()
        gt_words = ground_truth.split()
        wer = Levenshtein.distance(' '.join(pred_words), ' '.join(gt_words)) / len(gt_words)

    return {
        'original_image': image,
        'processed_image': processed_image,
        'prediction': ocr_results['text'],
        'confidence': ocr_results['confidence'],
        'words': ocr_results['words'],
        'inference_time_ms': inference_time_ms,
        'cer': cer,
        'wer': wer,
        'ground_truth': ground_truth,
    }


def run_command(cmd: list, description: str) -> tuple:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings")

    # Model selection
    st.subheader("Model Configuration")
    model_name = st.selectbox(
        "Model",
        ["TrOCR-Korean", "EasyOCR", "PaddleOCR", "Custom"],
        help="Select OCR model to use"
    )

    adapter_name = st.text_input(
        "Adapter (optional)",
        "",
        help="LoRA adapter name if using fine-tuned model"
    )

    engine = st.selectbox(
        "Engine",
        ["PyTorch", "ONNX", "OpenVINO", "TensorRT"],
        help="Inference engine"
    )

    # Preprocessing options
    st.subheader("Preprocessing")

    dpi_scale = st.slider(
        "DPI Scale",
        min_value=1.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Scale factor for image resolution"
    )

    denoise = st.checkbox("Denoise", value=False)
    sharpen = st.checkbox("Sharpen", value=False)

    # Visualization options
    st.subheader("Visualization")

    show_bboxes = st.checkbox("Show Bounding Boxes", value=True)
    highlight_low_conf = st.checkbox("Highlight Low Confidence", value=True)
    low_conf_threshold = st.slider(
        "Low Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )

    # Audit logging
    st.subheader("Audit Logging")
    enable_audit_log = st.checkbox("Enable Audit Logging", value=True)


# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

st.title("📝 LLMOCR - Complete Korean OCR Platform")
st.markdown("**Comprehensive data management, benchmarking, and OCR operations**")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🗂️ Dataset Management",
    "🔄 Data Processing",
    "🚀 Benchmark Execution",
    "🔁 Continuous Learning",
    "🖼️ Single Image OCR",
    "📊 Error Analysis",
    "📋 Audit Logs",
    "⚡ Batch Processing"
])


# ============================================================================
# TAB 1: DATASET MANAGEMENT
# ============================================================================

with tab1:
    st.header("🗂️ Dataset Management")
    st.markdown("Download and manage Korean OCR datasets")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Download SynthDoG-ko")

        synthdog_output = st.text_input(
            "Output Directory",
            "datasets/raw/synthdog_ko",
            key="synthdog_output"
        )

        synthdog_limit = st.number_input(
            "Sample Limit",
            min_value=10,
            max_value=100000,
            value=1000,
            step=100,
            help="Number of samples to download",
            key="synthdog_limit"
        )

        synthdog_start = st.number_input(
            "Start Index",
            min_value=0,
            value=0,
            help="Start downloading from this index",
            key="synthdog_start"
        )

        if st.button("📥 Download SynthDoG-ko", type="primary"):
            with st.spinner("Downloading SynthDoG-ko dataset..."):
                cmd = [
                    "python", "datasets/scripts/download_synthdog_ko.py",
                    "--output_dir", synthdog_output,
                    "--limit", str(synthdog_limit),
                    "--start_idx", str(synthdog_start),
                ]

                success, stdout, stderr = run_command(cmd, "Download SynthDoG-ko")

                if success:
                    st.success(f"✓ Downloaded {synthdog_limit} samples to {synthdog_output}")
                    st.session_state.datasets_downloaded.append({
                        'name': 'synthdog_ko',
                        'path': synthdog_output,
                        'samples': synthdog_limit,
                        'timestamp': datetime.now().isoformat()
                    })
                    with st.expander("📄 Download Log"):
                        st.text(stdout)
                else:
                    st.error("❌ Download failed")
                    st.error(stderr)

    with col2:
        st.subheader("Download AI-Hub Dataset")

        st.info("""
        **AI-Hub datasets require manual download:**
        1. Visit [AI-Hub](https://aihub.or.kr)
        2. Login and request dataset access
        3. Download the dataset
        4. Extract to `datasets/raw/aihub_<dataset_name>/`
        """)

        aihub_dataset = st.selectbox(
            "Dataset",
            ["admin_docs", "korean_fonts"],
            help="Select AI-Hub dataset",
            key="aihub_dataset"
        )

        aihub_output = st.text_input(
            "Output Directory",
            f"datasets/raw/aihub_{aihub_dataset}",
            key="aihub_output"
        )

        if st.button("📋 Show Download Instructions"):
            cmd = [
                "python", "datasets/scripts/download_aihub.py",
                "--dataset", aihub_dataset,
                "--output_dir", aihub_output,
            ]

            success, stdout, stderr = run_command(cmd, "AI-Hub Instructions")
            st.code(stdout)

    # Show downloaded datasets
    st.subheader("Downloaded Datasets")

    if st.session_state.datasets_downloaded:
        df = pd.DataFrame(st.session_state.datasets_downloaded)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No datasets downloaded yet")


# ============================================================================
# TAB 2: DATA PROCESSING
# ============================================================================

with tab2:
    st.header("🔄 Data Processing")
    st.markdown("Clean, process, and split datasets for training")

    st.subheader("1. Clean Dataset")

    col1, col2 = st.columns([1, 1])

    with col1:
        clean_source = st.text_input(
            "Source Directory",
            "datasets/raw/synthdog_ko",
            key="clean_source"
        )

        clean_output = st.text_input(
            "Output Directory",
            "datasets/processed/synthdog_ko_clean",
            key="clean_output"
        )

        clean_min_length = st.number_input("Min Text Length", 1, 100, 5)
        clean_max_length = st.number_input("Max Text Length", 100, 10000, 1000)
        clean_copy_images = st.checkbox("Copy/Link Images", value=True)

    with col2:
        clean_min_dim = st.number_input("Min Image Dimension", 16, 512, 32)
        clean_blur_threshold = st.number_input("Blur Threshold", 10.0, 500.0, 100.0)

        st.markdown("**Cleaning will filter out:**")
        st.markdown("- Corrupted images")
        st.markdown("- Images too small or blurry")
        st.markdown("- Text too short or too long")
        st.markdown("- Invalid or empty text")

    if st.button("🧹 Clean Dataset", type="primary"):
        with st.spinner("Cleaning dataset..."):
            cmd = [
                "python", "datasets/scripts/clean_data.py",
                "--source", clean_source,
                "--output", clean_output,
                "--min_length", str(clean_min_length),
                "--max_length", str(clean_max_length),
                "--min_dimension", str(clean_min_dim),
                "--blur_threshold", str(clean_blur_threshold),
            ]

            if clean_copy_images:
                cmd.append("--copy_images")

            success, stdout, stderr = run_command(cmd, "Clean Dataset")

            if success:
                st.success(f"✓ Dataset cleaned successfully")
                with st.expander("📄 Cleaning Report"):
                    st.text(stdout)

                # Try to load stats
                stats_file = Path(clean_output) / "cleaning_stats.json"
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Samples", stats.get('total', 0))
                    col2.metric("Valid Samples", stats.get('valid', 0))
                    col3.metric("Filtered Out", stats.get('total', 0) - stats.get('valid', 0))
                    col4.metric("Success Rate", f"{stats.get('valid', 0) / stats.get('total', 1) * 100:.1f}%")
            else:
                st.error("❌ Cleaning failed")
                st.error(stderr)

    st.divider()

    st.subheader("2. Create Train/Val/Test Splits")

    col1, col2 = st.columns([1, 1])

    with col1:
        split_input = st.text_input(
            "Input Directory (Cleaned Dataset)",
            "datasets/processed/synthdog_ko_clean",
            key="split_input"
        )

        train_ratio = st.slider("Train Ratio", 0.0, 1.0, 0.8, 0.05)
        val_ratio = st.slider("Validation Ratio", 0.0, 1.0, 0.1, 0.05)
        test_ratio = 1.0 - train_ratio - val_ratio

        st.info(f"Test Ratio: {test_ratio:.2f}")

        if train_ratio + val_ratio >= 1.0:
            st.error("Train + Val ratio must be < 1.0")

    with col2:
        split_seed = st.number_input("Random Seed", 0, 10000, 42, help="For reproducibility")

        st.markdown("**Split Configuration:**")
        st.markdown(f"- Train: {train_ratio*100:.0f}%")
        st.markdown(f"- Validation: {val_ratio*100:.0f}%")
        st.markdown(f"- Test: {test_ratio*100:.0f}%")

    if st.button("✂️ Create Splits", type="primary", disabled=(train_ratio + val_ratio >= 1.0)):
        with st.spinner("Creating splits..."):
            cmd = [
                "python", "datasets/scripts/create_splits.py",
                "--input", split_input,
                "--train_ratio", str(train_ratio),
                "--val_ratio", str(val_ratio),
                "--seed", str(split_seed),
            ]

            success, stdout, stderr = run_command(cmd, "Create Splits")

            if success:
                st.success("✓ Splits created successfully")
                with st.expander("📄 Split Report"):
                    st.text(stdout)

                # Try to load split stats
                stats_file = Path(split_input) / "split_stats.json"
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)

                    st.markdown("**Split Statistics:**")
                    for split_name, split_stats in stats.items():
                        with st.expander(f"📊 {split_name.upper()} Split"):
                            st.json(split_stats)
            else:
                st.error("❌ Split creation failed")
                st.error(stderr)


# ============================================================================
# TAB 3: BENCHMARK EXECUTION
# ============================================================================

with tab3:
    st.header("🚀 Benchmark Execution")
    st.markdown("Run comprehensive OCR benchmarks on multiple datasets")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Configuration")

        bench_models = st.text_input(
            "Model Paths (comma-separated)",
            "models/baseline",
            help="Paths to models to benchmark",
            key="bench_models"
        )

        bench_datasets = st.text_input(
            "Datasets (comma-separated)",
            "synthdog_ko_clean",
            help="Dataset names in datasets/processed/",
            key="bench_datasets"
        )

        bench_limit = st.number_input(
            "Sample Limit (optional)",
            min_value=0,
            value=0,
            help="0 = use all samples",
            key="bench_limit"
        )

        bench_device = st.selectbox(
            "Device",
            ["cuda", "cpu"],
            key="bench_device"
        )

    with col2:
        st.subheader("Metrics")
        st.markdown("""
        The benchmark will measure:
        - **CER** (Character Error Rate)
        - **WER** (Word Error Rate)
        - **Throughput** (images/sec)
        - **Latency** (p50, p95, p99)

        Results will be saved in multiple formats:
        - JSON (machine-readable)
        - CSV (spreadsheet)
        - Markdown (report)
        """)

    if st.button("▶️ Run Benchmark", type="primary"):
        with st.spinner("Running benchmark... This may take a while"):
            cmd = [
                "python", "benchmarks/run_bench.py",
                "--models", bench_models,
                "--datasets", bench_datasets,
                "--output_dir", "reports",
                "--device", bench_device,
            ]

            if bench_limit > 0:
                cmd.extend(["--limit", str(bench_limit)])

            # Show progress
            progress_placeholder = st.empty()
            output_placeholder = st.empty()

            success, stdout, stderr = run_command(cmd, "Run Benchmark")

            if success:
                st.success("✓ Benchmark completed successfully")

                with st.expander("📄 Benchmark Output"):
                    st.text(stdout)

                # Try to find and display latest results
                reports_dir = Path("reports")
                result_files = sorted(reports_dir.glob("benchmark_results_*.json"))

                if result_files:
                    latest_results = result_files[-1]
                    with open(latest_results, 'r') as f:
                        results = json.load(f)

                    st.subheader("📊 Results")

                    # Convert to DataFrame
                    df = pd.DataFrame(results)

                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)

                    if len(df) > 0:
                        col1.metric("Avg CER", f"{df['cer'].mean():.4f}")
                        col2.metric("Avg WER", f"{df['wer'].mean():.4f}")
                        col3.metric("Avg Throughput", f"{df['throughput'].mean():.2f} img/s")
                        col4.metric("Avg p95 Latency", f"{df['p95_latency'].mean():.1f} ms")

                    # Display detailed table
                    st.dataframe(df, use_container_width=True)

                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download JSON",
                            latest_results.read_text(),
                            file_name=latest_results.name,
                            mime="application/json"
                        )
                    with col2:
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv_data,
                            file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            else:
                st.error("❌ Benchmark failed")
                st.error(stderr)


# ============================================================================
# TAB 4: CONTINUOUS LEARNING
# ============================================================================

with tab4:
    st.header("🔁 Continuous Learning Pipeline")
    st.markdown("Automated training, evaluation, and regression testing")

    st.info("""
    **Continuous Learning Workflow:**
    1. New data is processed and cleaned
    2. Model is fine-tuned with LoRA
    3. Benchmarks are run on test datasets
    4. Results are compared to baseline
    5. Model is promoted if improvements are found
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        cl_base_model = st.text_input(
            "Base Model Path",
            "models/baseline",
            key="cl_base_model"
        )

        cl_new_data = st.text_input(
            "New Data Path",
            "datasets/raw/new_data",
            key="cl_new_data"
        )

        cl_dataset_name = st.text_input(
            "Dataset Name",
            "new_dataset",
            help="Name for the processed dataset",
            key="cl_dataset_name"
        )

        cl_benchmark_datasets = st.text_input(
            "Benchmark Datasets (comma-separated)",
            "synthdog_ko_clean",
            key="cl_benchmark_datasets"
        )

    with col2:
        cl_epochs = st.number_input("Training Epochs", 1, 20, 1, key="cl_epochs")

        cl_regression_threshold = st.number_input(
            "Regression Threshold",
            0.0, 0.1, 0.02, 0.01,
            help="CER delta threshold for detecting regressions",
            key="cl_regression_threshold"
        )

        cl_limit = st.number_input(
            "Sample Limit (optional)",
            0, 10000, 0,
            help="0 = use all samples",
            key="cl_limit"
        )

        cl_auto_promote = st.checkbox(
            "Auto-promote on success",
            value=False,
            help="Automatically promote model to production if improvements found",
            key="cl_auto_promote"
        )

    if st.button("🚀 Run Continuous Learning Pipeline", type="primary"):
        with st.spinner("Running continuous learning pipeline... This will take a while"):
            cmd = [
                "python", "benchmarks/continuous_learning.py",
                "--base_model", cl_base_model,
                "--new_data", cl_new_data,
                "--dataset_name", cl_dataset_name,
                "--benchmark_datasets", cl_benchmark_datasets,
                "--epochs", str(cl_epochs),
                "--regression_threshold", str(cl_regression_threshold),
            ]

            if cl_limit > 0:
                cmd.extend(["--limit", str(cl_limit)])

            if cl_auto_promote:
                cmd.append("--auto_promote")

            success, stdout, stderr = run_command(cmd, "Continuous Learning")

            if success:
                st.success("✓ Continuous learning pipeline completed")

                with st.expander("📄 Pipeline Output"):
                    st.text(stdout)

                # Try to find latest pipeline report
                experiments_dir = Path("models/experiments")
                report_files = sorted(experiments_dir.glob("pipeline_report_*.json"))

                if report_files:
                    latest_report = report_files[-1]
                    with open(latest_report, 'r') as f:
                        report = json.load(f)

                    st.subheader("📊 Pipeline Report")

                    regression_report = report.get('regression_report', {})

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Improvements", len(regression_report.get('improvements', [])))
                    col2.metric("Regressions", len(regression_report.get('regressions', [])))
                    col3.metric(
                        "Status",
                        "✓ PASS" if regression_report.get('overall_improvement') else "✗ FAIL"
                    )

                    # Show improvements
                    if regression_report.get('improvements'):
                        st.success("**Improvements Found:**")
                        for imp in regression_report['improvements']:
                            st.markdown(f"- **{imp['dataset']}**: {imp['baseline_cer']:.4f} → {imp['new_cer']:.4f} ({imp['delta']:+.4f})")

                    # Show regressions
                    if regression_report.get('regressions'):
                        st.error("**Regressions Detected:**")
                        for reg in regression_report['regressions']:
                            st.markdown(f"- **{reg['dataset']}**: {reg['baseline_cer']:.4f} → {reg['new_cer']:.4f} ({reg['delta']:+.4f})")

                    st.download_button(
                        "📥 Download Full Report",
                        json.dumps(report, indent=2),
                        file_name=latest_report.name,
                        mime="application/json"
                    )
            else:
                st.error("❌ Pipeline failed")
                st.error(stderr)


# ============================================================================
# TAB 5: SINGLE IMAGE OCR (from original GUI)
# ============================================================================

with tab5:
    st.header("🖼️ Single Image OCR")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image file for OCR"
        )

        # Ground truth (optional)
        ground_truth = st.text_area(
            "Ground Truth (optional)",
            "",
            help="Provide ground truth text to calculate CER/WER"
        )

        # Process button
        if st.button("🚀 Process Image", type="primary"):
            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file).convert('RGB')

                # Create preprocessing params
                preprocess_params = PreprocessingParams(
                    dpi_scale=dpi_scale,
                    denoise=denoise,
                    sharpen=sharpen,
                )

                # Create model info
                engine_map = {
                    "PyTorch": EngineType.PYTORCH,
                    "ONNX": EngineType.ONNX,
                    "OpenVINO": EngineType.OPENVINO,
                    "TensorRT": EngineType.TENSORRT,
                }

                model_info = ModelInfo(
                    model_name=model_name,
                    model_version="v1.0.0",
                    model_path=f"models/{model_name}",
                    adapter_name=adapter_name if adapter_name else None,
                    engine=engine_map[engine],
                )

                # Process image
                with st.spinner("Processing..."):
                    results = process_single_image(
                        image,
                        preprocess_params,
                        model_info,
                        ground_truth if ground_truth else None,
                    )

                # Store in session state
                st.session_state.latest_results = results

                # Log to audit
                if enable_audit_log:
                    # Save temp file for logging
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        image.save(tmp.name)
                        tmp_path = Path(tmp.name)

                    st.session_state.audit_logger.log_inference(
                        image_path=tmp_path,
                        model_info=model_info,
                        preprocessing_params=preprocess_params,
                        prediction=results['prediction'],
                        confidence=results['confidence'],
                        inference_time_ms=results['inference_time_ms'],
                        ground_truth=results.get('ground_truth'),
                        cer=results.get('cer'),
                        wer=results.get('wer'),
                    )

                st.success("✓ Processing complete!")
            else:
                st.error("Please upload an image first")

    with col2:
        st.subheader("Results")

        if 'latest_results' in st.session_state:
            results = st.session_state.latest_results

            # Show visualization
            if show_bboxes and results.get('words'):
                annotated = st.session_state.visualizer.draw_word_boxes(
                    results['processed_image'],
                    results['words'],
                    show_confidence=True,
                )
                st.image(annotated, caption="Annotated Image", use_container_width=True)
            else:
                st.image(results['processed_image'], caption="Processed Image", use_container_width=True)

            # Show prediction
            st.text_area("Prediction", results['prediction'], height=100)

            # Show metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

            with metrics_col1:
                st.metric("Confidence", f"{results['confidence']:.2%}")

            with metrics_col2:
                st.metric("Inference Time", f"{results['inference_time_ms']:.1f} ms")

            with metrics_col3:
                if results.get('cer') is not None:
                    st.metric("CER", f"{results['cer']:.2%}")

            # Word-level details
            if results.get('words'):
                with st.expander("📝 Word-Level Details"):
                    for idx, word in enumerate(results['words']):
                        conf_color = "🟢" if word['confidence'] >= low_conf_threshold else "🔴"
                        st.write(f"{conf_color} **{word['text']}**: {word['confidence']:.2%}")

            # High DPI Retry
            if results['confidence'] < 0.8:
                st.warning("⚠️ Low confidence detected. Try high DPI retry?")

                if st.button("🔄 Retry with High DPI (2x)"):
                    # Reprocess with higher DPI
                    preprocess_params.dpi_scale = 2.0
                    with st.spinner("Reprocessing with 2x DPI..."):
                        retry_results = process_single_image(
                            results['original_image'],
                            preprocess_params,
                            model_info,
                            ground_truth if ground_truth else None,
                        )
                        st.session_state.latest_results = retry_results
                    st.rerun()


# ============================================================================
# TAB 6: ERROR ANALYSIS (from original GUI)
# ============================================================================

with tab6:
    st.header("📊 Error Analysis Dashboard")

    st.markdown("""
    Analyze common OCR errors to identify patterns and improvement opportunities.
    """)

    # Upload benchmark results
    results_file = st.file_uploader(
        "Upload Benchmark Results (JSON)",
        type=['json'],
        key='error_analysis_upload',
        help="Upload benchmark results from run_bench.py"
    )

    if results_file:
        results_data = json.load(results_file)

        # Create analyzer
        analyzer = ErrorAnalyzer(output_dir="reports/error_analysis_streamlit")

        # Find top errors
        n_samples = st.slider("Number of Top Errors", 5, 50, 20)

        with st.spinner("Analyzing errors..."):
            error_samples = analyzer.find_top_errors(results_data, n=n_samples)

        st.subheader(f"Top {len(error_samples)} Errors")

        # Display error samples
        for idx, sample in enumerate(error_samples[:10]):
            with st.expander(f"#{idx+1}: CER={sample.cer:.3f}, WER={sample.wer:.3f}"):
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Try to load thumbnail
                    try:
                        if sample.image_path.exists():
                            st.image(str(sample.image_path), use_container_width=True)
                    except:
                        st.warning("Image not found")

                with col2:
                    st.markdown("**Ground Truth:**")
                    st.code(sample.ground_truth)

                    st.markdown("**Prediction:**")
                    st.code(sample.prediction)

                    st.markdown("**Error Breakdown:**")
                    st.write(f"- Substitutions: {sample.error_types['substitution']}")
                    st.write(f"- Insertions: {sample.error_types['insertion']}")
                    st.write(f"- Deletions: {sample.error_types['deletion']}")

        # Error patterns
        st.subheader("Error Patterns")

        patterns = analyzer.analyze_error_patterns(error_samples)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Errors", patterns['total_errors'])

        with col2:
            st.metric("Avg CER", f"{patterns['avg_cer']:.2%}")

        with col3:
            st.metric("Avg WER", f"{patterns['avg_wer']:.2%}")

        # Show top substitution patterns
        if patterns['substitution_patterns']:
            st.markdown("**Top Substitution Patterns:**")
            for pattern, count in list(patterns['substitution_patterns'].items())[:10]:
                st.write(f"- `{pattern}`: {count} occurrences")


# ============================================================================
# TAB 7: AUDIT LOGS (from original GUI)
# ============================================================================

with tab7:
    st.header("📋 Audit Logs")

    st.markdown("""
    View and analyze audit logs of all OCR operations.
    """)

    # Query options
    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input("Start Date", value=None)

    with col2:
        end_date = st.date_input("End Date", value=None)

    with col3:
        filter_model = st.text_input("Model Name Filter", "")

    if st.button("🔍 Query Logs"):
        with st.spinner("Loading audit logs..."):
            entries = st.session_state.audit_logger.query_logs(
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
                model_name=filter_model if filter_model else None,
            )

        st.success(f"Found {len(entries)} log entries")

        if entries:
            # Show statistics
            stats = st.session_state.audit_logger.get_statistics(entries)

            st.subheader("Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Inferences", stats['total_inferences'])

            with col2:
                st.metric("Mean Inference Time", f"{stats['inference_time']['mean']:.1f} ms")

            with col3:
                st.metric("Mean Confidence", f"{stats['confidence']['mean']:.2%}")

            with col4:
                if 'cer' in stats:
                    st.metric("Mean CER", f"{stats['cer']['mean']:.2%}")

            # Show recent entries
            st.subheader("Recent Entries")

            # Convert to dataframe for display
            df_data = []
            for entry in entries[-20:]:  # Last 20 entries
                df_data.append({
                    'Timestamp': entry['timestamp'],
                    'Model': entry['model_info']['model_name'],
                    'Engine': entry['model_info']['engine'],
                    'Confidence': f"{entry['confidence']:.2%}",
                    'Inference Time (ms)': f"{entry['inference_time_ms']:.1f}",
                    'CER': f"{entry['cer']:.2%}" if entry.get('cer') is not None else 'N/A',
                })

            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)

            # Export report
            if st.button("📥 Export Report"):
                report_path = Path("reports/audit_report.md")
                st.session_state.audit_logger.export_report(
                    output_path=report_path,
                    entries=entries,
                    format="markdown"
                )
                st.success(f"✓ Report exported to {report_path}")


# ============================================================================
# TAB 8: BATCH PROCESSING (from original GUI)
# ============================================================================

with tab8:
    st.header("⚡ Batch Processing")

    st.markdown("""
    Process multiple images in batch mode.
    """)

    # Upload multiple files
    uploaded_files = st.file_uploader(
        "Upload Images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key='batch_upload'
    )

    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} images")

        if st.button("🚀 Process Batch", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results_list = []

            engine_map = {
                "PyTorch": EngineType.PYTORCH,
                "ONNX": EngineType.ONNX,
                "OpenVINO": EngineType.OPENVINO,
                "TensorRT": EngineType.TENSORRT,
            }

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")

                # Load and process
                image = Image.open(uploaded_file).convert('RGB')

                preprocess_params = PreprocessingParams(dpi_scale=dpi_scale)
                model_info = ModelInfo(
                    model_name=model_name,
                    model_version="v1.0.0",
                    model_path=f"models/{model_name}",
                    engine=engine_map[engine],
                )

                results = process_single_image(image, preprocess_params, model_info)

                results_list.append({
                    'filename': uploaded_file.name,
                    'prediction': results['prediction'],
                    'confidence': results['confidence'],
                    'inference_time_ms': results['inference_time_ms'],
                })

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("✓ Batch processing complete!")

            # Show results
            st.subheader("Batch Results")

            df = pd.DataFrame(results_list)
            st.dataframe(df, use_container_width=True)

            # Download as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                csv,
                f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                key='download-csv'
            )


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>LLMOCR v2.0.0 | Complete Korean OCR Analysis & Operations Platform</p>
        <p>Dataset Management • Benchmarking • Continuous Learning • OCR Operations</p>
    </div>
    """,
    unsafe_allow_html=True
)

#!/usr/bin/env python3
"""
Streamlit GUI for LLMOCR - Korean OCR Analysis & Operations

Features:
- Upload and process images
- Visualize bounding boxes and confidence
- High DPI retry functionality
- Error analysis dashboard
- Audit log viewer
- Batch processing
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import time
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
    st.error(f"Import error: {e}")
    st.stop()


# Page config
st.set_page_config(
    page_title="LLMOCR - Korean OCR Interface",
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
                'confidence': 0.65,  # Low confidence
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


# Sidebar
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


# Main content area
st.title("📝 LLMOCR - Korean OCR Interface")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Single Image",
    "📊 Error Analysis",
    "📋 Audit Logs",
    "⚡ Batch Processing"
])

# Tab 1: Single Image Processing
with tab1:
    st.header("Single Image OCR")

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

# Tab 2: Error Analysis
with tab2:
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
        import json

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

# Tab 3: Audit Logs
with tab3:
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
            import pandas as pd

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

# Tab 4: Batch Processing
with tab4:
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

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")

                # Load and process
                image = Image.open(uploaded_file).convert('RGB')

                preprocess_params = PreprocessingParams(dpi_scale=dpi_scale)
                model_info = ModelInfo(
                    model_name=model_name,
                    model_version="v1.0.0",
                    model_path=f"models/{model_name}",
                    engine=EngineType.PYTORCH,
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

            import pandas as pd
            df = pd.DataFrame(results_list)
            st.dataframe(df, use_container_width=True)

            # Download as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                csv,
                "batch_results.csv",
                "text/csv",
                key='download-csv'
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>LLMOCR v1.0.0 | Korean OCR Analysis & Operations Interface</p>
    </div>
    """,
    unsafe_allow_html=True
)

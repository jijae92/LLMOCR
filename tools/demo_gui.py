#!/usr/bin/env python3
"""
Demo script to showcase GUI & operational features.

This script demonstrates:
- Audit logging
- Error analysis
- Visualization
- High DPI retry

Usage:
    python tools/demo_gui.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tools.audit_logging import AuditLogger, ModelInfo, PreprocessingParams, EngineType
from tools.error_analysis import ErrorAnalyzer, ErrorSample
from utils.visualization import BBoxVisualizer


def create_demo_image(text: str, size=(800, 200)) -> Image.Image:
    """Create a demo image with text."""
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
    except:
        font = ImageFont.load_default()

    # Draw text
    draw.text((50, 80), text, fill='black', font=font)

    return img


def demo_audit_logging():
    """Demonstrate audit logging functionality."""
    print("\n" + "="*70)
    print("DEMO 1: Audit Logging")
    print("="*70)

    # Initialize logger
    logger = AuditLogger(log_dir="logs/demo_audit")

    # Create demo image
    demo_img = create_demo_image("한국어 OCR 테스트")
    demo_path = Path("logs/demo_image.jpg")
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_img.save(demo_path)

    # Model info
    model_info = ModelInfo(
        model_name="trocr-korean-demo",
        model_version="v1.0.0",
        model_path="models/trocr-korean",
        adapter_name="receipts_lora",
        adapter_version="v1.2.0",
        engine=EngineType.PYTORCH,
    )

    # Preprocessing params
    preprocess_params = PreprocessingParams(
        dpi_scale=1.5,
        denoise=True,
        sharpen=False,
    )

    # Log sample inference
    print("\n1. Logging OCR inference...")
    entry = logger.log_inference(
        image_path=demo_path,
        model_info=model_info,
        preprocessing_params=preprocess_params,
        prediction="한국어 OCR 테스트",
        confidence=0.95,
        inference_time_ms=123.45,
        ground_truth="한국어 OCR 테스트",
        cer=0.0,
        wer=0.0,
        metadata={'note': 'Perfect prediction'},
    )

    print(f"   ✓ Logged entry with hash: {entry.input_hash[:16]}...")

    # Log another with errors
    print("\n2. Logging inference with errors...")
    entry2 = logger.log_inference(
        image_path=demo_path,
        model_info=model_info,
        preprocessing_params=preprocess_params,
        prediction="한국어 OGR 테스듀",  # Intentional errors
        confidence=0.72,
        inference_time_ms=145.67,
        ground_truth="한국어 OCR 테스트",
        cer=0.15,
        wer=0.25,
        metadata={'note': 'Contains errors'},
    )

    print(f"   ✓ Logged entry with CER={entry2.cer:.2f}")

    # Query logs
    print("\n3. Querying audit logs...")
    entries = logger.query_logs()
    print(f"   ✓ Found {len(entries)} total entries")

    # Get statistics
    print("\n4. Generating statistics...")
    stats = logger.get_statistics(entries)
    print(f"   • Total inferences: {stats['total_inferences']}")
    print(f"   • Mean inference time: {stats['inference_time']['mean']:.2f} ms")
    print(f"   • Mean confidence: {stats['confidence']['mean']:.2%}")
    if 'cer' in stats:
        print(f"   • Mean CER: {stats['cer']['mean']:.2%}")

    # Export report
    print("\n5. Exporting report...")
    report_path = Path("reports/demo_audit_report.md")
    logger.export_report(report_path, entries, format="markdown")
    print(f"   ✓ Report saved to {report_path}")


def demo_error_analysis():
    """Demonstrate error analysis functionality."""
    print("\n" + "="*70)
    print("DEMO 2: Error Analysis")
    print("="*70)

    # Create analyzer
    analyzer = ErrorAnalyzer(output_dir="reports/demo_error_analysis")

    # Sample results with errors
    sample_results = [
        {
            'image_path': 'sample1.jpg',
            'prediction': '한국어 OCR 테스트',
            'ground_truth': '한국어 OCR 테스트',
        },
        {
            'image_path': 'sample2.jpg',
            'prediction': '문서 이미지 인식',
            'ground_truth': '문서 이미지 인식',
        },
        {
            'image_path': 'sample3.jpg',
            'prediction': '데이타 처리 시스템',  # Error: 데이타 -> 데이터
            'ground_truth': '데이터 처리 시스템',
        },
        {
            'image_path': 'sample4.jpg',
            'prediction': '자동화 파이플라인',  # Error: 파이플라인 -> 파이프라인
            'ground_truth': '자동화 파이프라인',
        },
        {
            'image_path': 'sample5.jpg',
            'prediction': '벤치마그 자동화',  # Error: 벤치마그 -> 벤치마크
            'ground_truth': '벤치마크 자동화',
        },
    ]

    # Find top errors
    print("\n1. Finding top errors...")
    error_samples = analyzer.find_top_errors(sample_results, n=5, metric='cer')

    for idx, sample in enumerate(error_samples):
        print(f"\n   Error #{idx+1}:")
        print(f"   • CER: {sample.cer:.3f}")
        print(f"   • Ground Truth: {sample.ground_truth}")
        print(f"   • Prediction: {sample.prediction}")
        print(f"   • Errors: S={sample.error_types['substitution']} "
              f"I={sample.error_types['insertion']} "
              f"D={sample.error_types['deletion']}")

    # Analyze patterns
    print("\n2. Analyzing error patterns...")
    patterns = analyzer.analyze_error_patterns(error_samples)
    print(f"   • Total errors: {patterns['total_errors']}")
    print(f"   • Avg CER: {patterns['avg_cer']:.3f}")

    if patterns['substitution_patterns']:
        print("\n   Top substitution patterns:")
        for pattern, count in list(patterns['substitution_patterns'].items())[:5]:
            print(f"   • {pattern}: {count} times")

    print("\n3. Error analysis complete!")


def demo_visualization():
    """Demonstrate visualization functionality."""
    print("\n" + "="*70)
    print("DEMO 3: Bounding Box Visualization")
    print("="*70)

    # Create demo image
    demo_img = create_demo_image("한국어 OCR 테스트 문서", size=(800, 300))

    # Example bounding boxes
    bboxes = [
        {
            'box': [40, 70, 180, 110],
            'text': '한국어',
            'confidence': 0.95,
        },
        {
            'box': [190, 70, 270, 110],
            'text': 'OCR',
            'confidence': 0.92,
        },
        {
            'box': [280, 70, 380, 110],
            'text': '테스트',
            'confidence': 0.65,  # Low confidence
        },
        {
            'box': [390, 70, 490, 110],
            'text': '문서',
            'confidence': 0.88,
        },
    ]

    # Create visualizer
    print("\n1. Creating visualizer...")
    visualizer = BBoxVisualizer(low_confidence_threshold=0.7)

    # Draw bounding boxes
    print("2. Drawing bounding boxes...")
    annotated = visualizer.draw_bboxes(
        demo_img,
        bboxes,
        highlight_low_confidence=True
    )

    # Save result
    output_dir = Path("reports/demo_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "bbox_demo.png"
    annotated.save(output_path)
    print(f"   ✓ Saved to {output_path}")

    # Create comparison view
    print("3. Creating comparison view...")
    comparison = visualizer.create_comparison_view(
        original=demo_img,
        processed=annotated,
        prediction="한국어 OCR 테스트 문서",
        ground_truth="한국어 OCR 테스트 문서",
        confidence=0.85,
    )

    comparison_path = output_dir / "comparison_demo.png"
    comparison.save(comparison_path)
    print(f"   ✓ Saved to {comparison_path}")

    print("\n4. Highlighting low confidence regions...")
    print(f"   • Low confidence threshold: 0.70")
    print(f"   • '테스트' marked as low confidence (0.65)")


def demo_high_dpi_retry():
    """Demonstrate high DPI retry concept."""
    print("\n" + "="*70)
    print("DEMO 4: High DPI Retry")
    print("="*70)

    print("\n1. Initial inference at 1x DPI:")
    print("   • Prediction: '한국어 OGR 테스트'")  # Error: OCR -> OGR
    print("   • Confidence: 0.72")
    print("   • CER: 0.15")
    print("   ⚠️  Low confidence detected!")

    print("\n2. Retrying with 2x DPI:")
    print("   • Scaling image to 2x resolution")
    print("   • Reprocessing...")
    print("   • Prediction: '한국어 OCR 테스트'")  # Correct!
    print("   • Confidence: 0.94")
    print("   • CER: 0.00")
    print("   ✓ Improved prediction!")

    print("\n3. When to use High DPI Retry:")
    print("   • Confidence < 80%")
    print("   • Small text")
    print("   • Blurry images")
    print("   • Low-quality scans")


def main():
    """Run all demos."""
    print("="*70)
    print("LLMOCR GUI & OPERATIONAL FEATURES DEMO")
    print("="*70)
    print("\nThis demo showcases the GUI and operational features:")
    print("  1. Audit Logging")
    print("  2. Error Analysis")
    print("  3. Bounding Box Visualization")
    print("  4. High DPI Retry")

    try:
        demo_audit_logging()
        demo_error_analysis()
        demo_visualization()
        demo_high_dpi_retry()

        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print("\nGenerated files:")
        print("  • logs/demo_audit/ - Audit log files")
        print("  • reports/demo_audit_report.md - Audit report")
        print("  • reports/demo_error_analysis/ - Error analysis")
        print("  • reports/demo_visualization/ - Visualization examples")
        print("\nTo launch the full GUI:")
        print("  streamlit run gui/streamlit_app.py")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

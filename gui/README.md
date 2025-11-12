# LLMOCR GUI - Streamlit Interface

Interactive web interface for Korean OCR analysis and operations.

## Features

### 📝 Single Image Processing
- Upload and process individual images
- Real-time OCR prediction
- Bounding box visualization
- Confidence score highlighting
- **High DPI Retry**: Automatically retry with higher resolution when confidence is low
- Adjustable preprocessing parameters (DPI scale, denoise, sharpen)

### 📊 Error Analysis Dashboard
- Upload benchmark results
- View top N errors with highest CER/WER
- Character-level diff visualization
- Error pattern analysis (substitutions, insertions, deletions)
- Identify common mistakes for model improvement

### 📋 Audit Log Viewer
- Query historical OCR operations
- Filter by date range, model, engine
- Performance statistics and trends
- Export reports in Markdown/CSV format
- Track model versions and preprocessing params

### ⚡ Batch Processing
- Process multiple images at once
- Progress tracking
- Export results as CSV
- Parallel processing support

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install GUI-specific requirements
pip install streamlit pandas plotly
```

## Usage

### Start the GUI

```bash
# From project root
streamlit run gui/streamlit_app.py

# Or from gui directory
cd gui
streamlit run streamlit_app.py
```

The interface will open in your browser at `http://localhost:8501`

### Configuration

Configure settings in the sidebar:

**Model Configuration:**
- Select OCR model (TrOCR, EasyOCR, PaddleOCR, Custom)
- Optional LoRA adapter
- Inference engine (PyTorch, ONNX, OpenVINO, TensorRT)

**Preprocessing:**
- DPI Scale: 1.0-3.0x (higher for better quality)
- Denoise: Remove noise from image
- Sharpen: Enhance text edges

**Visualization:**
- Toggle bounding boxes
- Highlight low confidence regions
- Adjust confidence threshold

**Audit Logging:**
- Enable/disable automatic logging
- All operations tracked with timestamps

## Features in Detail

### 1. High DPI Retry

When OCR confidence is below threshold (default 80%):
- ⚠️ Warning displayed
- Click "🔄 Retry with High DPI (2x)" button
- Image automatically reprocessed at 2x resolution
- New results compared with original

This is especially useful for:
- Small text
- Low-quality scans
- Blurry images
- Documents with fine details

### 2. Bounding Box Visualization

Color-coded confidence levels:
- 🟢 **Green**: High confidence (≥90%)
- 🟡 **Yellow**: Medium confidence (70-90%)
- 🔴 **Red**: Low confidence (<70%)

Low confidence regions are highlighted with semi-transparent red overlay.

### 3. Error Analysis

Upload benchmark results JSON to analyze:
- Top N samples with highest errors
- Character-level diff visualization
  - Green: Correct characters
  - Red: Substitutions
  - Blue: Insertions
  - Yellow: Deletions
- Common error patterns
- Substitution frequency analysis

### 4. Audit Logging

Every OCR operation logs:
- **Input**: SHA256 hash, image dimensions
- **Model**: Name, version, adapter, engine
- **Preprocessing**: All parameters used
- **Output**: Prediction, confidence
- **Performance**: Inference time
- **Accuracy**: CER/WER (if ground truth provided)

Query logs by:
- Date range
- Model name
- CER/WER thresholds
- Engine type

Export reports for:
- Performance monitoring
- Model comparison
- Debugging
- Compliance/audit requirements

## Customization

### Integrate Your Model

Edit `gui/streamlit_app.py` and replace `mock_ocr_inference()`:

```python
def mock_ocr_inference(image, model_name, engine):
    # Replace with your actual model
    from your_model import OCRModel

    model = OCRModel.load(model_name, engine)
    results = model.predict(image)

    return {
        'text': results.text,
        'confidence': results.confidence,
        'words': results.word_level_results,  # List of word bboxes
        'inference_time_ms': results.time_ms,
    }
```

### Add Custom Preprocessing

Add your preprocessing steps in `preprocess_image()`:

```python
def preprocess_image(image, **params):
    # Your custom preprocessing
    if params.get('custom_filter'):
        image = apply_custom_filter(image)

    return image
```

### Custom Visualizations

Use the `BBoxVisualizer` class from `utils/visualization/`:

```python
from utils.visualization import BBoxVisualizer

visualizer = BBoxVisualizer(low_confidence_threshold=0.7)
annotated_img = visualizer.draw_bboxes(image, bboxes)
```

## Tips for Best Results

1. **Image Quality**
   - Use high-resolution scans (300+ DPI)
   - Ensure good lighting and contrast
   - Minimize blur and noise

2. **DPI Scaling**
   - Start with 1.0x for normal documents
   - Use 1.5-2.0x for small text
   - Use 2.0-3.0x for very fine print

3. **Preprocessing**
   - Enable denoise for scanned documents
   - Enable sharpen for blurry images
   - Experiment to find best settings

4. **Batch Processing**
   - Process similar documents together
   - Use consistent preprocessing settings
   - Monitor memory usage with large batches

## Troubleshooting

### GUI Won't Start
```bash
# Check Streamlit installation
pip install --upgrade streamlit

# Check port availability
streamlit run gui/streamlit_app.py --server.port 8502
```

### Images Not Loading
- Check file format (PNG, JPEG, BMP, TIFF supported)
- Verify file size (Streamlit has 200MB upload limit)
- Try converting to PNG/JPEG first

### Low Performance
- Reduce DPI scale
- Disable denoise/sharpen
- Use ONNX/OpenVINO engine for faster inference
- Process smaller batches

### Memory Issues
- Process images one at a time
- Reduce batch size
- Clear browser cache
- Restart Streamlit

## Keyboard Shortcuts

In Streamlit:
- `R`: Rerun the app
- `C`: Clear cache
- `M`: Toggle sidebar

## Screenshots

### Single Image Processing
![Single Image](../docs/screenshots/gui_single_image.png)

### Error Analysis
![Error Analysis](../docs/screenshots/gui_error_analysis.png)

### Audit Logs
![Audit Logs](../docs/screenshots/gui_audit_logs.png)

## Advanced Usage

### Custom Themes

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Deploy to Production

#### Using Docker

```bash
# Build image
docker build -t llmocr-gui -f gui/Dockerfile .

# Run container
docker run -p 8501:8501 llmocr-gui
```

#### Using Streamlit Cloud

1. Push to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repository
4. Deploy!

## API Access

For programmatic access, use the underlying tools directly:

```python
from tools.audit_logging import AuditLogger
from tools.error_analysis import ErrorAnalyzer
from utils.visualization import BBoxVisualizer

# Audit logging
logger = AuditLogger()
logger.log_inference(...)

# Error analysis
analyzer = ErrorAnalyzer()
errors = analyzer.find_top_errors(results)

# Visualization
visualizer = BBoxVisualizer()
annotated = visualizer.draw_bboxes(image, bboxes)
```

## Contributing

To add new features:
1. Add functionality to appropriate module (`tools/`, `utils/`)
2. Create UI component in `streamlit_app.py`
3. Update this documentation
4. Test thoroughly

## License

See main project LICENSE file.

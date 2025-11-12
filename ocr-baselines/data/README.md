# Training Data

This directory contains sample training and validation data for fine-tuning OCR models.

## Data Formats

The training scripts support three data formats:

### 1. JSONL Format (Recommended)

Each line is a JSON object with `image` and `text` fields:

```jsonl
{"image": "path/to/image1.jpg", "text": "Ground truth text 1"}
{"image": "path/to/image2.jpg", "text": "Ground truth text 2"}
```

**Example:** `train.jsonl`, `val.jsonl`

### 2. CSV Format

CSV file with `image` and `text` columns:

```csv
image,text
path/to/image1.jpg,"Ground truth text 1"
path/to/image2.jpg,"Ground truth text 2"
```

### 3. Directory Format

Directory containing image files and corresponding text files:

```
data/train/
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
...
```

## Preparing Your Own Data

### Quick Start

1. **Collect images**: Gather your domain-specific images (receipts, invoices, contracts, etc.)

2. **Create ground truth**: Manually transcribe the text from each image

3. **Choose a format**: Use JSONL for flexibility, directory for simplicity

4. **Split data**: Create training and validation sets (typically 80/20 or 90/10)

### Example: Creating JSONL Data

```python
import json

# Your data
samples = [
    {"image": "receipts/r001.jpg", "text": "Receipt text here..."},
    {"image": "receipts/r002.jpg", "text": "Another receipt..."},
]

# Write to JSONL
with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
```

### Example: Converting Directory to JSONL

```python
import json
from pathlib import Path

data_dir = Path("data/my_images")
output_file = "data/train.jsonl"

with open(output_file, "w", encoding="utf-8") as out_f:
    for img_file in data_dir.glob("*.jpg"):
        txt_file = img_file.with_suffix(".txt")
        if txt_file.exists():
            with open(txt_file, "r", encoding="utf-8") as txt_f:
                text = txt_f.read().strip()

            sample = {"image": str(img_file), "text": text}
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
```

## Data Quality Tips

### Image Quality
- **Resolution**: 300 DPI or higher recommended
- **Format**: JPEG or PNG
- **Size**: Images will be automatically resized to max 1280px
- **Clarity**: Ensure text is readable

### Ground Truth Quality
- **Accuracy**: Transcribe exactly as shown in the image
- **Formatting**: Preserve line breaks, spacing, and structure
- **Encoding**: Use UTF-8 for Korean and other non-ASCII characters

### Dataset Size
- **Minimum**: 100 samples for initial experiments
- **Good**: 1,000+ samples for decent performance
- **Excellent**: 10,000+ samples for production quality

### Train/Val Split
- **Typical**: 80% training, 20% validation
- **Small datasets**: 90% training, 10% validation
- **Large datasets**: 95% training, 5% validation

## Example Workflow

1. **Collect 500 receipts** (your domain data)

2. **Manually transcribe** the text from each receipt

3. **Create directory structure:**
   ```
   data/receipts/
   ├── r001.jpg + r001.txt
   ├── r002.jpg + r002.txt
   ...
   ├── r500.jpg + r500.txt
   ```

4. **Convert to JSONL** (optional but recommended):
   ```bash
   python scripts/convert_to_jsonl.py data/receipts data/train.jsonl
   ```

5. **Create validation set** (split last 100 samples):
   ```bash
   head -400 data/train.jsonl > data/train_only.jsonl
   tail -100 data/train.jsonl > data/val.jsonl
   mv data/train_only.jsonl data/train.jsonl
   ```

6. **Start training:**
   ```bash
   python -m src.train.train_trocr_lora \
     --train_json data/train.jsonl \
     --val_json data/val.jsonl \
     --output_dir runs/receipts-trocr-lora
   ```

## Sample Data

The included `train.jsonl` and `val.jsonl` files contain sample data for demonstration purposes. Replace these with your own domain-specific data for actual fine-tuning.

## Data Augmentation

For document OCR, excessive augmentation is **not recommended**. Keep it minimal:

✅ **Recommended:**
- Slight rotation (±2°)
- Minor brightness/contrast adjustments

❌ **Not Recommended:**
- Heavy rotation
- Aggressive cropping
- Color jittering
- Extreme distortions

Document structure and layout are important for OCR, so preserve them during training.

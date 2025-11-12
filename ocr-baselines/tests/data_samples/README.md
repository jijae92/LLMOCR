# Test Data Samples

This directory contains test samples for OCR evaluation.

## Required Files

For each test sample, you need two files:
- `{name}.jpg` (or `.png`, `.jpeg`, `.bmp`) - The image file
- `{name}.txt` - Ground truth text (UTF-8 encoded)

## Current Samples

### invoice_ko
- **Description**: Korean tax invoice
- **Files needed**: `invoice_ko.jpg` + `invoice_ko.txt` (provided)

### receipt_ko
- **Description**: Korean cafe receipt
- **Files needed**: `receipt_ko.jpg` + `receipt_ko.txt` (provided)

## Adding Your Own Samples

1. Take a photo or scan of a Korean document
2. Save as `{name}.jpg` in this directory
3. Create `{name}.txt` with the ground truth text
4. Run evaluation:
   ```bash
   python -m src.eval --data-dir tests/data_samples --models trocr,donut
   ```

## Creating Test Images

Since actual image files cannot be committed to this repository initially, you can:

1. **Use your own document images**: Take photos of receipts, invoices, forms, etc.
2. **Generate synthetic images**: Use libraries like PIL to create text images:

```python
from PIL import Image, ImageDraw, ImageFont

# Create a simple test image
img = Image.new('RGB', (800, 600), color='white')
draw = ImageDraw.Draw(img)

# Add text (you may need to specify a Korean font path)
text = "테스트 텍스트\nTest Text"
draw.text((50, 50), text, fill='black')

img.save('tests/data_samples/test_image.jpg')
```

3. **Download public datasets**: Consider using:
   - [CORD Dataset](https://github.com/clovaai/cord) - Korean receipt OCR
   - [SROIE Dataset](https://rrc.cvc.uab.es/?ch=13) - Receipt OCR
   - Create custom samples from public documents

## Ground Truth Format

The `.txt` files should contain the exact text you want the OCR model to recognize, including:
- Line breaks (use `\n`)
- Punctuation
- Numbers and special characters
- Korean and English mixed text

Example:
```
영수증
Total: 10,000원
Thank you!
```

## Evaluation Metrics

The evaluation script computes:
- **CER** (Character Error Rate): Character-level accuracy
- **WER** (Word Error Rate): Word-level accuracy
- **Edit Distance**: Levenshtein distance between prediction and ground truth
- **Latency**: Inference time in milliseconds
- **Throughput**: Images processed per second

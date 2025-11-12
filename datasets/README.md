# Korean OCR Datasets

This directory contains scripts and documentation for Korean OCR datasets used in the LLMOCR project.

## Data Sources

### 1. AI-Hub (공공 행정문서 OCR)

AI-Hub provides various Korean public administrative document OCR datasets.

**Key Datasets:**
- **공공행정문서 OCR**: Public administrative documents with text annotations
- **한국어 글자체 이미지**: Korean character/font images
- **문서 이미지 인식**: Document image recognition

**License & Access:**
- License: AI-Hub 이용약관 (https://aihub.or.kr/intrcn/guid/usagepolicy.do)
- Access: Requires free account registration at https://aihub.or.kr
- Usage: Research and non-commercial purposes permitted with attribution

**Download Instructions:**
1. Register at https://aihub.or.kr
2. Navigate to dataset page
3. Apply for dataset access (immediate approval for most datasets)
4. Download via web interface or use provided API

**Relevant Datasets:**
- [공공행정문서 OCR](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=88)
- [한국어 글자체 이미지](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=89)

### 2. SynthDoG-ko (Synthetic Document Generator - Korean)

Synthetic Korean OCR dataset generated using document templates and Korean text.

**Source:**
- Repository: https://github.com/clovaai/donut
- Dataset: https://huggingface.co/datasets/naver-clova-ix/synthdog-ko

**License:**
- MIT License (permissive commercial use)
- No registration required

**Characteristics:**
- Synthetic document images with Korean text
- Various fonts, layouts, and backgrounds
- Perfect ground truth annotations
- Scalable generation

**Download:**
```bash
# Using Hugging Face datasets
python scripts/download_synthdog_ko.py --output_dir ./raw/synthdog_ko --split train --limit 10000
```

## Standard Data Format

All datasets are converted to a standard JSONL format for consistent processing:

```json
{"image_path": "relative/path/to/image.jpg", "text": "인식할 텍스트", "split": "train", "source": "aihub_admin", "metadata": {"difficulty": "medium", "ko_ratio": 0.95}}
```

**Fields:**
- `image_path`: Relative path from dataset root
- `text`: Ground truth text (normalized)
- `split`: train/val/test
- `source`: Dataset identifier (aihub_admin, synthdog_ko, etc.)
- `metadata`: Additional information
  - `difficulty`: easy/medium/hard (based on text complexity)
  - `ko_ratio`: Ratio of Korean characters (0.0-1.0)
  - `en_ratio`: Ratio of English characters (0.0-1.0)
  - `symbol_ratio`: Ratio of symbols/numbers (0.0-1.0)
  - `length`: Character count

## Dataset Splits

### Training Sets
- `ko_receipts`: Korean receipt images (~1,000 samples)
- `ko_contracts`: Korean contract documents (~800 samples)
- `ko_admin_docs`: Korean administrative documents (~1,500 samples)
- `synthdog_ko_small`: Synthetic Korean documents (10,000 samples)

### Validation Sets
Each training set has a corresponding validation split (10% of data)

### Test Sets
- `ko_receipts_test`: 200 samples
- `ko_contracts_test`: 200 samples
- `ko_admin_docs_test`: 200 samples
- `synthdog_ko_test`: 1,000 samples

## Directory Structure

```
datasets/
├── README.md (this file)
├── raw/                          # Raw downloaded data
│   ├── aihub_admin/
│   ├── synthdog_ko/
│   └── ...
├── processed/                    # Cleaned and processed data
│   ├── ko_receipts/
│   │   ├── images/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   ├── ko_contracts/
│   ├── ko_admin_docs/
│   └── synthdog_ko_small/
├── scripts/
│   ├── download_aihub.py         # Download AI-Hub datasets
│   ├── download_synthdog_ko.py   # Download SynthDoG-ko
│   ├── clean_data.py             # Data cleaning and filtering
│   ├── create_splits.py          # Create train/val/test splits
│   └── analyze_difficulty.py     # Tag difficulty levels
└── configs/
    ├── aihub_config.yaml
    └── synthdog_config.yaml
```

## Quick Start

### 1. Download Data

```bash
# Download SynthDoG-ko (no registration needed)
python scripts/download_synthdog_ko.py --limit 10000

# Download AI-Hub datasets (requires credentials)
# First, set up credentials in configs/aihub_config.yaml
python scripts/download_aihub.py --dataset admin_docs --limit 2000
```

### 2. Clean and Process

```bash
# Clean data: remove corrupted images, filter lengths, tag difficulty
python scripts/clean_data.py --source raw/synthdog_ko --output processed/synthdog_ko_small

python scripts/clean_data.py --source raw/aihub_admin --output processed/ko_admin_docs
```

### 3. Create Splits

```bash
# Create train/val/test splits (80/10/10)
python scripts/create_splits.py --input processed/synthdog_ko_small --train_ratio 0.8 --val_ratio 0.1
```

### 4. Analyze Dataset

```bash
# Get statistics and difficulty distribution
python scripts/analyze_difficulty.py --input processed/synthdog_ko_small/train.jsonl
```

## Data Cleaning Criteria

The cleaning pipeline applies the following filters:

1. **Image Validation**
   - Remove corrupted/unreadable images
   - Check minimum dimensions (32x32 pixels)
   - Verify image format (JPEG, PNG)

2. **Text Length Filtering**
   - Minimum: 5 characters
   - Maximum: 1000 characters
   - Remove empty annotations

3. **Quality Checks**
   - Remove images with excessive blur (Laplacian variance < threshold)
   - Filter out completely black/white images
   - Validate UTF-8 encoding

4. **Difficulty Tagging**
   - **Easy**: >90% Korean chars, simple layout
   - **Medium**: 70-90% Korean, mixed layout
   - **Hard**: <70% Korean, complex layout, low quality

## Data Statistics

After processing, each dataset includes a `stats.json` file with:
- Total samples per split
- Average text length
- Character distribution (Korean/English/symbols)
- Difficulty distribution
- Image dimension statistics

## Citation

If you use these datasets, please cite:

```bibtex
# AI-Hub
@misc{aihub2024ocr,
  title={AI-Hub Korean OCR Datasets},
  author={NIA},
  year={2024},
  url={https://aihub.or.kr}
}

# SynthDoG-ko
@article{kim2021donut,
  title={OCR-free Document Understanding Transformer},
  author={Kim, Geewook and Hong, Teakgyu and Yim, Moonbin and Nam, JeongYeon and Park, Jinyoung and Yim, Jinyeong and Hwang, Wonseok and Yun, Sangdoo and Han, Dongyoon and Park, Seunghyun},
  journal={arXiv preprint arXiv:2111.15664},
  year={2021}
}
```

## Troubleshooting

### AI-Hub Download Issues
- Ensure valid credentials in config file
- Check network connectivity
- Verify dataset access approval on AI-Hub website

### Memory Issues
- Process data in batches using `--batch_size` parameter
- Use `--limit` to download subset first

### Encoding Issues
- All scripts assume UTF-8 encoding
- Use `--encoding` flag if needed

## Contributing

To add a new dataset:
1. Create download script in `scripts/download_<dataset>.py`
2. Follow standard JSONL format
3. Add documentation to this README
4. Update benchmark configurations

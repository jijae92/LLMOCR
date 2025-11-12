"""Data loader for OCR fine-tuning."""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset


class OCRDataset(Dataset):
    """Dataset for OCR fine-tuning with (image, text) pairs."""

    def __init__(
        self,
        data_path: str,
        processor: Any,
        max_target_length: int = 512,
        image_size: Tuple[int, int] = (1280, 1280),
    ):
        """
        Initialize OCR dataset.

        Args:
            data_path: Path to JSONL, CSV, or directory with images/texts
            processor: HuggingFace processor (TrOCRProcessor, DonutProcessor, etc.)
            max_target_length: Maximum length for target sequences
            image_size: Maximum image dimensions (width, height)
        """
        self.processor = processor
        self.max_target_length = max_target_length
        self.image_size = image_size

        self.samples = self._load_data(data_path)

        if not self.samples:
            raise ValueError(f"No samples loaded from {data_path}")

    def _load_data(self, data_path: str) -> List[Dict[str, str]]:
        """
        Load data from various formats.

        Supported formats:
        - JSONL: {"image": "path/to/img.jpg", "text": "ground truth"}
        - CSV: image,text columns
        - Directory: image files with corresponding .txt files

        Args:
            data_path: Path to data

        Returns:
            List of {"image": path, "text": text} dicts
        """
        path = Path(data_path)
        samples = []

        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        # JSONL format
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    samples.append({
                        "image": data["image"],
                        "text": data["text"],
                    })

        # CSV format
        elif path.suffix == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append({
                        "image": row["image"],
                        "text": row["text"],
                    })

        # Directory format
        elif path.is_dir():
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            for image_file in path.iterdir():
                if image_file.suffix.lower() not in image_extensions:
                    continue

                # Look for corresponding text file
                text_file = image_file.with_suffix(".txt")
                if not text_file.exists():
                    print(f"Warning: No text file for {image_file.name}, skipping")
                    continue

                with open(text_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                samples.append({
                    "image": str(image_file),
                    "text": text,
                })

        else:
            raise ValueError(
                f"Unsupported data format: {path}. "
                "Use .jsonl, .csv, or directory with image+txt pairs."
            )

        return samples

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for training.

        Args:
            image: Input PIL Image

        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if too large (maintain aspect ratio)
        width, height = image.size
        max_width, max_height = self.image_size

        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dict with processed pixel_values and labels
        """
        sample = self.samples[idx]

        # Load and preprocess image
        image = Image.open(sample["image"])
        image = self.preprocess_image(image)

        # Process image
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Tokenize text
        labels = self.processor.tokenizer(
            sample["text"],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 (ignore in loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def create_data_loaders(
    train_path: str,
    val_path: Optional[str],
    processor: Any,
    batch_size: int = 2,
    num_workers: int = 0,
    max_target_length: int = 512,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Create training and validation data loaders.

    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        processor: HuggingFace processor
        batch_size: Batch size
        num_workers: Number of data loader workers
        max_target_length: Maximum target sequence length

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create training dataset
    train_dataset = OCRDataset(
        data_path=train_path,
        processor=processor,
        max_target_length=max_target_length,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disable for MPS compatibility
    )

    # Create validation dataset if path provided
    val_loader = None
    if val_path:
        val_dataset = OCRDataset(
            data_path=val_path,
            processor=processor,
            max_target_length=max_target_length,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    return train_loader, val_loader

"""
LoRA fine-tuning script for TrOCR (macOS/MPS compatible).

This script uses PEFT LoRA for memory-efficient fine-tuning.
Designed to work on macOS with MPS acceleration.

For QLoRA (4-bit quantization):
- NOT recommended on macOS due to bitsandbytes Metal limitations
- Use cloud GPU (A10/A100/H100) with CUDA
- Download adapter checkpoint for local inference

System Requirements:
- macOS: MPS-capable device, PyTorch 2.0+
- Linux: CUDA GPU or CPU
- Unified Memory/VRAM: Recommended 16GB+ for batch_size=2
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.train.data_loader import create_data_loaders


def get_device(prefer_mps: bool = True) -> str:
    """
    Get the best available device.

    Args:
        prefer_mps: Whether to prefer MPS over CPU on macOS

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif prefer_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def setup_lora_model(
    base_model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> tuple:
    """
    Setup TrOCR model with LoRA adapters.

    Args:
        base_model_name: Base model name from HuggingFace
        lora_r: LoRA rank (lower = fewer parameters)
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout for LoRA layers
        target_modules: Which modules to apply LoRA to

    Returns:
        Tuple of (processor, model)
    """
    print(f"Loading base model: {base_model_name}")
    processor = TrOCRProcessor.from_pretrained(base_model_name)
    model = VisionEncoderDecoderModel.from_pretrained(base_model_name)

    # Default target modules for TrOCR (attention + feed-forward)
    if target_modules is None:
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc1",
            "fc2",
        ]

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    print(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Target modules: {target_modules}")

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return processor, model


def compute_metrics(eval_pred, processor):
    """
    Compute CER metric during evaluation.

    Args:
        eval_pred: Predictions from model
        processor: TrOCR processor

    Returns:
        Dict with CER metric
    """
    cer_metric = evaluate.load("cer")

    predictions, labels = eval_pred

    # Decode predictions
    pred_ids = predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    # Replace -100 with pad_token_id for decoding
    labels = torch.where(
        torch.tensor(labels) != -100,
        torch.tensor(labels),
        processor.tokenizer.pad_token_id
    )

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for TrOCR (macOS/MPS compatible)"
    )

    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/trocr-base-printed",
        help="Base TrOCR model from HuggingFace",
    )

    # Data arguments
    parser.add_argument(
        "--train_json",
        type=str,
        required=True,
        help="Path to training data (JSONL, CSV, or directory)",
    )
    parser.add_argument(
        "--val_json",
        type=str,
        help="Path to validation data (optional)",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=512,
        help="Maximum target sequence length",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (4-64, lower = fewer params)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/trocr-lora",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--no_mps",
        action="store_true",
        help="Disable MPS even if available (use CPU instead)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision (not recommended for MPS)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = get_device(prefer_mps=not args.no_mps)
    else:
        device = args.device

    print(f"\n{'='*80}")
    print(f"TrOCR LoRA Fine-tuning")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Base Model: {args.base_model}")
    print(f"Training Data: {args.train_json}")
    print(f"Validation Data: {args.val_json or 'None'}")
    print(f"Output Directory: {args.output_dir}")
    print(f"LoRA Config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Batch Size: {args.per_device_train_batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*80}\n")

    # macOS MPS warnings
    if device == "mps":
        print("⚠️  MPS (Apple Silicon) Detected:")
        print("  - Using stable FP32 training (FP16 may cause issues on some PyTorch versions)")
        print("  - If you encounter crashes, try --no_mps to use CPU")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - MPS available: {torch.backends.mps.is_available()}")
        print()

        # Set float32 matmul precision for stability
        torch.set_float32_matmul_precision("high")

    # Setup model with LoRA
    processor, model = setup_lora_model(
        base_model_name=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Create data loaders
    print("\nLoading training data...")
    train_loader, val_loader = create_data_loaders(
        train_path=args.train_json,
        val_path=args.val_json,
        processor=processor,
        batch_size=args.per_device_train_batch_size,
        max_target_length=args.max_target_length,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps if val_loader else None,
        save_steps=args.save_steps,
        evaluation_strategy="steps" if val_loader else "no",
        save_strategy="steps",
        load_best_model_at_end=True if val_loader else False,
        metric_for_best_model="cer" if val_loader else None,
        greater_is_better=False,  # Lower CER is better
        fp16=args.fp16 and device == "cuda",  # Only enable FP16 on CUDA
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=3,  # Keep only 3 best checkpoints
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
        # MPS-specific settings
        use_cpu=(device == "cpu"),
    )

    # Custom compute metrics function
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, processor)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset if val_loader else None,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_fn if val_loader else None,
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save final model (LoRA adapters only)
    print("\nSaving LoRA adapters...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Save training metrics
    metrics = train_result.metrics
    metrics_file = Path(args.output_dir) / "train_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"LoRA adapters saved to: {args.output_dir}")
    print(f"Training metrics saved to: {metrics_file}")

    if val_loader:
        print(f"\nFinal CER: {metrics.get('eval_cer', 'N/A'):.4f}")

    print(f"\nTo use the fine-tuned model:")
    print(f"  python -m src.cli --model trocr --adapter-path {args.output_dir} --image path/to/image.jpg")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Continuous learning loop for Korean OCR models.

Workflow:
1. New data arrives → Clean and process
2. Fine-tune model with LoRA
3. Run evaluation benchmarks
4. Check for regressions
5. Optionally promote model if improvements found

Usage:
    python continuous_learning.py --base_model models/baseline --new_data datasets/raw/new_data
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import shutil


class ContinuousLearningPipeline:
    """Automated continuous learning pipeline."""

    def __init__(
        self,
        base_model_path: str,
        output_dir: str = "models/experiments",
        regression_threshold: float = 0.02,
    ):
        self.base_model_path = Path(base_model_path)
        self.output_dir = Path(output_dir)
        self.regression_threshold = regression_threshold

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_new_data(
        self,
        raw_data_path: Path,
        dataset_name: str,
    ) -> Path:
        """Process new raw data through cleaning pipeline."""
        print("\n" + "="*70)
        print("Step 1: Processing New Data")
        print("="*70)

        processed_path = Path(f"datasets/processed/{dataset_name}")

        # Run cleaning script
        cmd = [
            "python", "datasets/scripts/clean_data.py",
            "--source", str(raw_data_path),
            "--output", str(processed_path),
            "--copy_images",
        ]

        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error cleaning data: {result.stderr}")
            raise RuntimeError("Data cleaning failed")

        print(result.stdout)

        # Create splits
        cmd = [
            "python", "datasets/scripts/create_splits.py",
            "--input", str(processed_path),
        ]

        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error creating splits: {result.stderr}")
            raise RuntimeError("Split creation failed")

        print(result.stdout)

        return processed_path

    def train_lora_model(
        self,
        dataset_path: Path,
        experiment_name: str,
        epochs: int = 1,
    ) -> Path:
        """Train LoRA model on new data."""
        print("\n" + "="*70)
        print("Step 2: Training LoRA Model")
        print("="*70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_path = self.output_dir / f"{experiment_name}_{timestamp}"
        model_output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nBase model: {self.base_model_path}")
        print(f"Output path: {model_output_path}")
        print(f"Dataset: {dataset_path}")
        print(f"Epochs: {epochs}")

        # TODO: Implement actual LoRA training
        # Example for TrOCR with PEFT:
        """
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
        from peft import get_peft_model, LoraConfig, TaskType

        # Load base model
        model = VisionEncoderDecoderModel.from_pretrained(self.base_model_path)
        processor = TrOCRProcessor.from_pretrained(self.base_model_path)

        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        # Get PEFT model
        model = get_peft_model(model, lora_config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(model_output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            save_steps=500,
            logging_steps=100,
        )

        # Create dataset
        train_dataset = create_dataset(dataset_path / "train.jsonl", processor)

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()
        trainer.save_model(str(model_output_path))
        """

        # Placeholder
        print("\n⚠️  Using placeholder training. Implement train_lora_model() with your model.")
        print("   See comments in the code for example implementation.")

        # Create placeholder model directory
        (model_output_path / "config.json").write_text(
            json.dumps({"base_model": str(self.base_model_path), "epochs": epochs}, indent=2)
        )

        print(f"\n✓ Model saved to {model_output_path}")

        return model_output_path

    def evaluate_model(
        self,
        model_path: Path,
        benchmark_datasets: List[str],
        limit: Optional[int] = None,
    ) -> Dict:
        """Run evaluation benchmarks on model."""
        print("\n" + "="*70)
        print("Step 3: Evaluating Model")
        print("="*70)

        # Run benchmarks
        cmd = [
            "python", "benchmarks/run_bench.py",
            "--models", str(model_path),
            "--datasets", ",".join(benchmark_datasets),
            "--output_dir", "reports",
        ]

        if limit:
            cmd.extend(["--limit", str(limit)])

        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error running benchmarks: {result.stderr}")
            raise RuntimeError("Benchmark evaluation failed")

        print(result.stdout)

        # Load latest results
        reports_dir = Path("reports")
        result_files = sorted(reports_dir.glob("benchmark_results_*.json"))

        if not result_files:
            raise RuntimeError("No benchmark results found")

        latest_results = result_files[-1]
        with open(latest_results, 'r') as f:
            results = json.load(f)

        return results

    def check_regression(
        self,
        baseline_results: Dict,
        new_results: Dict,
    ) -> Dict:
        """Check for regressions compared to baseline."""
        print("\n" + "="*70)
        print("Step 4: Regression Check")
        print("="*70)

        # Group results by dataset
        baseline_by_dataset = {r["dataset_name"]: r for r in baseline_results}
        new_by_dataset = {r["dataset_name"]: r for r in new_results}

        regression_report = {
            "regressions": [],
            "improvements": [],
            "overall_improvement": False,
        }

        print("\nComparing results:")
        print("-" * 70)

        for dataset_name in baseline_by_dataset:
            if dataset_name not in new_by_dataset:
                continue

            baseline = baseline_by_dataset[dataset_name]
            new = new_by_dataset[dataset_name]

            baseline_cer = baseline["cer"]
            new_cer = new["cer"]

            cer_delta = new_cer - baseline_cer
            cer_delta_pct = (cer_delta / baseline_cer * 100) if baseline_cer > 0 else 0

            print(f"\n{dataset_name}:")
            print(f"  Baseline CER: {baseline_cer:.4f}")
            print(f"  New CER: {new_cer:.4f}")
            print(f"  Delta: {cer_delta:+.4f} ({cer_delta_pct:+.2f}%)")

            if cer_delta > self.regression_threshold:
                print(f"  ⚠️  REGRESSION DETECTED")
                regression_report["regressions"].append({
                    "dataset": dataset_name,
                    "baseline_cer": baseline_cer,
                    "new_cer": new_cer,
                    "delta": cer_delta,
                })
            elif cer_delta < -self.regression_threshold:
                print(f"  ✓ IMPROVEMENT")
                regression_report["improvements"].append({
                    "dataset": dataset_name,
                    "baseline_cer": baseline_cer,
                    "new_cer": new_cer,
                    "delta": cer_delta,
                })
            else:
                print(f"  = No significant change")

        # Determine overall improvement
        if regression_report["improvements"] and not regression_report["regressions"]:
            regression_report["overall_improvement"] = True

        print("\n" + "-" * 70)
        print(f"\nSummary:")
        print(f"  Improvements: {len(regression_report['improvements'])}")
        print(f"  Regressions: {len(regression_report['regressions'])}")
        print(f"  Overall: {'✓ PASS' if regression_report['overall_improvement'] else '✗ FAIL'}")

        return regression_report

    def run_pipeline(
        self,
        new_data_path: str,
        dataset_name: str,
        benchmark_datasets: List[str],
        baseline_model: Optional[str] = None,
        epochs: int = 1,
        limit: Optional[int] = None,
        auto_promote: bool = False,
    ):
        """Run complete continuous learning pipeline."""
        print("="*70)
        print("CONTINUOUS LEARNING PIPELINE")
        print("="*70)
        print(f"\nNew data: {new_data_path}")
        print(f"Dataset name: {dataset_name}")
        print(f"Benchmark datasets: {benchmark_datasets}")
        print(f"Baseline model: {baseline_model or self.base_model_path}")

        # Step 1: Process new data
        processed_data = self.process_new_data(
            raw_data_path=Path(new_data_path),
            dataset_name=dataset_name,
        )

        # Step 2: Train LoRA model
        experiment_name = f"lora_{dataset_name}"
        new_model = self.train_lora_model(
            dataset_path=processed_data,
            experiment_name=experiment_name,
            epochs=epochs,
        )

        # Step 3: Evaluate new model
        new_results = self.evaluate_model(
            model_path=new_model,
            benchmark_datasets=benchmark_datasets,
            limit=limit,
        )

        # Step 4: Get baseline results or evaluate baseline
        if baseline_model:
            baseline_path = Path(baseline_model)
        else:
            baseline_path = self.base_model_path

        print(f"\nEvaluating baseline model: {baseline_path}")
        baseline_results = self.evaluate_model(
            model_path=baseline_path,
            benchmark_datasets=benchmark_datasets,
            limit=limit,
        )

        # Step 5: Check for regressions
        regression_report = self.check_regression(
            baseline_results=baseline_results,
            new_results=new_results,
        )

        # Save report
        report_path = self.output_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump({
                "dataset_name": dataset_name,
                "new_model": str(new_model),
                "baseline_model": str(baseline_path),
                "regression_report": regression_report,
                "new_results": new_results,
                "baseline_results": baseline_results,
            }, f, indent=2)

        print(f"\n✓ Pipeline report saved to {report_path}")

        # Step 6: Promote model if auto_promote enabled
        if auto_promote and regression_report["overall_improvement"]:
            print("\n" + "="*70)
            print("Auto-promoting model to production")
            print("="*70)

            production_path = Path("models/production")
            production_path.mkdir(parents=True, exist_ok=True)

            # Copy model
            new_production = production_path / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(new_model, new_production)

            print(f"\n✓ Model promoted to {new_production}")

        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Run continuous learning pipeline"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to base model"
    )
    parser.add_argument(
        "--new_data",
        type=str,
        required=True,
        help="Path to new raw data"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name for the new dataset"
    )
    parser.add_argument(
        "--benchmark_datasets",
        type=str,
        default="synthdog_ko_small",
        help="Comma-separated list of datasets to benchmark on"
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default=None,
        help="Optional baseline model for comparison (defaults to base_model)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples for quick testing"
    )
    parser.add_argument(
        "--regression_threshold",
        type=float,
        default=0.02,
        help="CER delta threshold for regression detection"
    )
    parser.add_argument(
        "--auto_promote",
        action="store_true",
        help="Automatically promote model if improvements found"
    )

    args = parser.parse_args()

    benchmark_datasets = [d.strip() for d in args.benchmark_datasets.split(',')]

    pipeline = ContinuousLearningPipeline(
        base_model_path=args.base_model,
        regression_threshold=args.regression_threshold,
    )

    pipeline.run_pipeline(
        new_data_path=args.new_data,
        dataset_name=args.dataset_name,
        benchmark_datasets=benchmark_datasets,
        baseline_model=args.baseline_model,
        epochs=args.epochs,
        limit=args.limit,
        auto_promote=args.auto_promote,
    )


if __name__ == "__main__":
    main()

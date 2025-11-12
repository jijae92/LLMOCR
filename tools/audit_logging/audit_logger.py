#!/usr/bin/env python3
"""
Audit logging system for OCR operations.

Tracks:
- Input hash (SHA256)
- Model and adapter versions
- Preprocessing parameters
- Engine type (PyTorch/ONNX/OpenVINO)
- Timestamps and performance metrics
"""

import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum


class EngineType(Enum):
    """Supported inference engines."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    OPENVINO = "openvino"
    TENSORRT = "tensorrt"


@dataclass
class PreprocessingParams:
    """Preprocessing parameters."""
    target_height: int = 384
    target_width: int = 384
    maintain_aspect_ratio: bool = True
    padding_color: tuple = (255, 255, 255)
    dpi_scale: float = 1.0
    normalize_mean: tuple = (0.5, 0.5, 0.5)
    normalize_std: tuple = (0.5, 0.5, 0.5)
    grayscale: bool = False
    denoise: bool = False
    sharpen: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelInfo:
    """Model and adapter version information."""
    model_name: str
    model_version: str
    model_path: str
    adapter_name: Optional[str] = None
    adapter_version: Optional[str] = None
    adapter_path: Optional[str] = None
    engine: EngineType = EngineType.PYTORCH

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['engine'] = self.engine.value
        return data


@dataclass
class AuditLogEntry:
    """Single audit log entry."""
    timestamp: str
    input_hash: str
    input_path: Optional[str]
    input_size: tuple  # (width, height)
    model_info: ModelInfo
    preprocessing_params: PreprocessingParams
    prediction: str
    confidence: float
    inference_time_ms: float
    ground_truth: Optional[str] = None
    cer: Optional[float] = None
    wer: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'input_hash': self.input_hash,
            'input_path': self.input_path,
            'input_size': {
                'width': self.input_size[0],
                'height': self.input_size[1]
            },
            'model_info': self.model_info.to_dict(),
            'preprocessing_params': self.preprocessing_params.to_dict(),
            'prediction': self.prediction,
            'confidence': self.confidence,
            'inference_time_ms': self.inference_time_ms,
            'ground_truth': self.ground_truth,
            'cer': self.cer,
            'wer': self.wer,
            'metadata': self.metadata,
        }


class AuditLogger:
    """Audit logging system for OCR operations."""

    def __init__(self, log_dir: str = "logs/audit", log_format: str = "jsonl"):
        """
        Initialize audit logger.

        Args:
            log_dir: Directory to store audit logs
            log_format: Log format (jsonl, json, csv)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_format = log_format

        # Initialize Python logger for errors
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create file handler
        log_file = self.log_dir / "audit_logger.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def compute_image_hash(self, image_path: Path) -> str:
        """Compute SHA256 hash of image file."""
        sha256_hash = hashlib.sha256()

        with open(image_path, "rb") as f:
            # Read in chunks for memory efficiency
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def log_inference(
        self,
        image_path: Path,
        model_info: ModelInfo,
        preprocessing_params: PreprocessingParams,
        prediction: str,
        confidence: float,
        inference_time_ms: float,
        ground_truth: Optional[str] = None,
        cer: Optional[float] = None,
        wer: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLogEntry:
        """
        Log an OCR inference operation.

        Args:
            image_path: Path to input image
            model_info: Model and adapter information
            preprocessing_params: Preprocessing parameters used
            prediction: OCR prediction text
            confidence: Prediction confidence score
            inference_time_ms: Inference time in milliseconds
            ground_truth: Optional ground truth text
            cer: Optional character error rate
            wer: Optional word error rate
            metadata: Optional additional metadata

        Returns:
            AuditLogEntry object
        """
        try:
            # Get image info
            from PIL import Image
            with Image.open(image_path) as img:
                input_size = img.size  # (width, height)

            # Compute hash
            input_hash = self.compute_image_hash(image_path)

            # Create log entry
            entry = AuditLogEntry(
                timestamp=datetime.now().isoformat(),
                input_hash=input_hash,
                input_path=str(image_path),
                input_size=input_size,
                model_info=model_info,
                preprocessing_params=preprocessing_params,
                prediction=prediction,
                confidence=confidence,
                inference_time_ms=inference_time_ms,
                ground_truth=ground_truth,
                cer=cer,
                wer=wer,
                metadata=metadata or {},
            )

            # Write to log file
            self._write_entry(entry)

            self.logger.info(f"Logged inference for {image_path.name} (hash: {input_hash[:16]}...)")

            return entry

        except Exception as e:
            self.logger.error(f"Error logging inference: {e}")
            raise

    def _write_entry(self, entry: AuditLogEntry):
        """Write log entry to file."""
        timestamp_date = entry.timestamp.split('T')[0]

        if self.log_format == "jsonl":
            log_file = self.log_dir / f"audit_{timestamp_date}.jsonl"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + '\n')

        elif self.log_format == "json":
            log_file = self.log_dir / f"audit_{timestamp_date}.json"

            # Read existing entries
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
            else:
                entries = []

            # Append new entry
            entries.append(entry.to_dict())

            # Write back
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)

    def query_logs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_name: Optional[str] = None,
        min_cer: Optional[float] = None,
        max_cer: Optional[float] = None,
        engine: Optional[EngineType] = None,
    ) -> list:
        """
        Query audit logs with filters.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            model_name: Filter by model name
            min_cer: Minimum CER threshold
            max_cer: Maximum CER threshold
            engine: Filter by engine type

        Returns:
            List of matching log entries
        """
        entries = []

        # Get all log files
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))

        for log_file in log_files:
            # Check date range
            file_date = log_file.stem.replace('audit_', '')

            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

            # Read entries
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    entry = json.loads(line)

                    # Apply filters
                    if model_name and entry['model_info']['model_name'] != model_name:
                        continue

                    if min_cer is not None and (entry.get('cer') is None or entry['cer'] < min_cer):
                        continue

                    if max_cer is not None and (entry.get('cer') is None or entry['cer'] > max_cer):
                        continue

                    if engine and entry['model_info']['engine'] != engine.value:
                        continue

                    entries.append(entry)

        return entries

    def get_statistics(self, entries: Optional[list] = None) -> Dict[str, Any]:
        """
        Get statistics from audit logs.

        Args:
            entries: Optional list of entries (if None, loads all)

        Returns:
            Dictionary with statistics
        """
        if entries is None:
            entries = self.query_logs()

        if not entries:
            return {}

        # Calculate statistics
        total_inferences = len(entries)

        inference_times = [e['inference_time_ms'] for e in entries]
        confidences = [e['confidence'] for e in entries]

        cers = [e['cer'] for e in entries if e.get('cer') is not None]
        wers = [e['wer'] for e in entries if e.get('wer') is not None]

        # Count by engine
        engines = {}
        for e in entries:
            engine = e['model_info']['engine']
            engines[engine] = engines.get(engine, 0) + 1

        # Count by model
        models = {}
        for e in entries:
            model = e['model_info']['model_name']
            models[model] = models.get(model, 0) + 1

        stats = {
            'total_inferences': total_inferences,
            'inference_time': {
                'mean': sum(inference_times) / len(inference_times),
                'min': min(inference_times),
                'max': max(inference_times),
                'p95': sorted(inference_times)[int(len(inference_times) * 0.95)],
            },
            'confidence': {
                'mean': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences),
            },
            'engines': engines,
            'models': models,
        }

        if cers:
            stats['cer'] = {
                'mean': sum(cers) / len(cers),
                'min': min(cers),
                'max': max(cers),
            }

        if wers:
            stats['wer'] = {
                'mean': sum(wers) / len(wers),
                'min': min(wers),
                'max': max(wers),
            }

        return stats

    def export_report(
        self,
        output_path: Path,
        entries: Optional[list] = None,
        format: str = "markdown"
    ):
        """
        Export audit report.

        Args:
            output_path: Path to output file
            entries: Optional list of entries (if None, loads all)
            format: Report format (markdown, html, csv)
        """
        if entries is None:
            entries = self.query_logs()

        stats = self.get_statistics(entries)

        if format == "markdown":
            self._export_markdown_report(output_path, entries, stats)
        elif format == "csv":
            self._export_csv_report(output_path, entries)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_markdown_report(self, output_path: Path, entries: list, stats: Dict):
        """Export markdown report."""
        lines = [
            "# OCR Audit Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Inferences:** {stats['total_inferences']:,}",
            "",
            "## Performance Statistics",
            "",
            "### Inference Time",
            "",
            f"- **Mean:** {stats['inference_time']['mean']:.2f} ms",
            f"- **Min:** {stats['inference_time']['min']:.2f} ms",
            f"- **Max:** {stats['inference_time']['max']:.2f} ms",
            f"- **P95:** {stats['inference_time']['p95']:.2f} ms",
            "",
            "### Confidence",
            "",
            f"- **Mean:** {stats['confidence']['mean']:.4f}",
            f"- **Min:** {stats['confidence']['min']:.4f}",
            f"- **Max:** {stats['confidence']['max']:.4f}",
            "",
        ]

        if 'cer' in stats:
            lines.extend([
                "### Character Error Rate (CER)",
                "",
                f"- **Mean:** {stats['cer']['mean']:.4f}",
                f"- **Min:** {stats['cer']['min']:.4f}",
                f"- **Max:** {stats['cer']['max']:.4f}",
                "",
            ])

        lines.extend([
            "## Usage by Engine",
            "",
            "| Engine | Count | Percentage |",
            "|--------|-------|------------|",
        ])

        for engine, count in stats['engines'].items():
            pct = count / stats['total_inferences'] * 100
            lines.append(f"| {engine} | {count:,} | {pct:.1f}% |")

        lines.extend([
            "",
            "## Usage by Model",
            "",
            "| Model | Count | Percentage |",
            "|-------|-------|------------|",
        ])

        for model, count in stats['models'].items():
            pct = count / stats['total_inferences'] * 100
            lines.append(f"| {model} | {count:,} | {pct:.1f}% |")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _export_csv_report(self, output_path: Path, entries: list):
        """Export CSV report."""
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'timestamp', 'input_hash', 'input_path',
                'model_name', 'model_version', 'adapter_name', 'engine',
                'prediction', 'confidence', 'inference_time_ms',
                'ground_truth', 'cer', 'wer'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in entries:
                row = {
                    'timestamp': entry['timestamp'],
                    'input_hash': entry['input_hash'],
                    'input_path': entry['input_path'],
                    'model_name': entry['model_info']['model_name'],
                    'model_version': entry['model_info']['model_version'],
                    'adapter_name': entry['model_info'].get('adapter_name', ''),
                    'engine': entry['model_info']['engine'],
                    'prediction': entry['prediction'],
                    'confidence': entry['confidence'],
                    'inference_time_ms': entry['inference_time_ms'],
                    'ground_truth': entry.get('ground_truth', ''),
                    'cer': entry.get('cer', ''),
                    'wer': entry.get('wer', ''),
                }
                writer.writerow(row)


def main():
    """Example usage."""
    # Initialize logger
    logger = AuditLogger()

    # Example model info
    model_info = ModelInfo(
        model_name="trocr-korean",
        model_version="v1.0.0",
        model_path="models/trocr-korean",
        adapter_name="receipts_lora",
        adapter_version="v1.2.0",
        adapter_path="models/adapters/receipts_lora",
        engine=EngineType.PYTORCH,
    )

    # Example preprocessing params
    preprocess_params = PreprocessingParams(
        target_height=384,
        target_width=384,
        dpi_scale=1.5,
    )

    print("Audit Logger Example")
    print("=" * 70)

    # Log example (would need real image)
    # entry = logger.log_inference(
    #     image_path=Path("test.jpg"),
    #     model_info=model_info,
    #     preprocessing_params=preprocess_params,
    #     prediction="예측 텍스트",
    #     confidence=0.95,
    #     inference_time_ms=123.45,
    #     ground_truth="정답 텍스트",
    #     cer=0.05,
    #     wer=0.10,
    # )

    # Query logs
    entries = logger.query_logs()
    print(f"Found {len(entries)} log entries")

    # Get statistics
    if entries:
        stats = logger.get_statistics(entries)
        print(f"\nStatistics:")
        print(json.dumps(stats, indent=2))

        # Export report
        logger.export_report(
            output_path=Path("logs/audit_report.md"),
            entries=entries,
            format="markdown"
        )
        print(f"\n✓ Report exported to logs/audit_report.md")


if __name__ == "__main__":
    main()

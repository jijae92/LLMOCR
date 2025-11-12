#!/usr/bin/env python3
"""
Download AI-Hub Korean OCR datasets.

Note: Requires AI-Hub account and dataset access approval.
Configure credentials in configs/aihub_config.yaml before running.

Usage:
    python download_aihub.py --dataset admin_docs --limit 2000
"""

import argparse
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Optional, Dict
import yaml

try:
    import requests
    from tqdm import tqdm
    from PIL import Image
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install requests pyyaml pillow tqdm")
    exit(1)


class AIHubDownloader:
    """Download datasets from AI-Hub with authentication."""

    def __init__(self, config_path: str = "configs/aihub_config.yaml"):
        self.config = self._load_config(config_path)
        self.session = requests.Session()
        self.authenticated = False

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            print("Creating template config file...")
            self._create_template_config(config_path)
            print(f"Please edit {config_path} with your credentials and re-run.")
            exit(1)

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_template_config(self, config_path: str):
        """Create template configuration file."""
        template = {
            "credentials": {
                "username": "your_aihub_username",
                "password": "your_aihub_password",
            },
            "datasets": {
                "admin_docs": {
                    "dataset_id": "88",
                    "name": "공공행정문서 OCR",
                    "url": "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=88",
                },
                "korean_fonts": {
                    "dataset_id": "89",
                    "name": "한국어 글자체 이미지",
                    "url": "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=89",
                },
            },
        }

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(template, f, allow_unicode=True, default_flow_style=False)

    def authenticate(self) -> bool:
        """Authenticate with AI-Hub."""
        username = self.config['credentials']['username']
        password = self.config['credentials']['password']

        if username == "your_aihub_username":
            print("Error: Please configure your AI-Hub credentials in configs/aihub_config.yaml")
            return False

        print("Authenticating with AI-Hub...")
        # Note: This is a placeholder implementation
        # Actual implementation requires AI-Hub API documentation
        # Users should manually download from AI-Hub website or use official API

        print("⚠️  Note: Automatic download requires AI-Hub API access.")
        print("    Please manually download the dataset from:")
        print(f"    https://aihub.or.kr")
        print()
        print("    After downloading:")
        print("    1. Extract the zip file")
        print("    2. Place images and annotations in raw/<dataset_name>/")
        print("    3. Run the cleaning script")
        return False

    def download_dataset(
        self,
        dataset_name: str,
        output_dir: str,
        limit: Optional[int] = None
    ):
        """Download dataset from AI-Hub."""
        if dataset_name not in self.config['datasets']:
            print(f"Unknown dataset: {dataset_name}")
            print(f"Available datasets: {list(self.config['datasets'].keys())}")
            return

        dataset_info = self.config['datasets'][dataset_name]
        print(f"Dataset: {dataset_info['name']}")
        print(f"URL: {dataset_info['url']}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # For now, provide manual download instructions
        self._manual_download_instructions(dataset_info, output_path)

    def _manual_download_instructions(self, dataset_info: Dict, output_path: Path):
        """Provide instructions for manual download."""
        print("\n" + "="*70)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("="*70)
        print(f"\n1. Visit: {dataset_info['url']}")
        print("\n2. Click 'Download' button (requires login)")
        print("\n3. Extract downloaded ZIP file")
        print("\n4. Organize files as follows:")
        print(f"\n   {output_path}/")
        print("   ├── images/           # All image files")
        print("   └── annotations/      # JSON/XML annotation files")
        print("\n5. Run the parsing script:")
        print(f"\n   python scripts/parse_aihub_format.py --input {output_path}")
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Download AI-Hub Korean OCR datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (admin_docs, korean_fonts, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ./raw/<dataset>)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/aihub_config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    output_dir = args.output_dir or f"./raw/aihub_{args.dataset}"

    downloader = AIHubDownloader(config_path=args.config)
    downloader.download_dataset(
        dataset_name=args.dataset,
        output_dir=output_dir,
        limit=args.limit
    )


if __name__ == "__main__":
    main()

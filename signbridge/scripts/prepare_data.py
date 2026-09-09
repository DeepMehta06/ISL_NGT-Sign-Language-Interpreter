"""Data preparation script — extracts raw datasets and runs pose extraction.

Usage:
    python scripts/prepare_data.py --dataset include
    python scripts/prepare_data.py --dataset islvt
    python scripts/prepare_data.py --dataset isl_dataset

Extracts each zip file, runs KeypointExtractor on every video frame-by-frame,
saves processed .npy arrays to data/processed/, then deletes the extracted
video folders to reclaim disk space. The original zip files are also deleted
after successful extraction and processing.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import settings
from backend.core.logger import get_logger
from backend.ml.data.loader import DatasetLoader

logger = get_logger(__name__)

SUPPORTED_DATASETS = ("include", "islvt", "isl_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and preprocess SignBridge datasets.")
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        required=True,
        help="Which dataset to prepare.",
    )
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep extracted video folders after processing (default: delete them).",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep original zip files after extraction (default: delete them).",
    )
    return parser.parse_args()


def prepare_include(keep_extracted: bool, keep_zips: bool) -> None:
    dataset_dir = settings.include_dataset_dir
    zip_files = sorted(dataset_dir.glob("*.zip"))

    if not zip_files:
        logger.error(f"No zip files found in {dataset_dir}. Download INCLUDE first.")
        sys.exit(1)

    logger.info(f"Found {len(zip_files)} INCLUDE zip files to process.")
    extract_root = settings.processed_data_dir / "_include_extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    for i, zip_path in enumerate(zip_files, 1):
        logger.info(f"[{i}/{len(zip_files)}] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)

        if not keep_zips:
            zip_path.unlink()
            logger.info(f"Deleted zip: {zip_path.name}")

    logger.info("All zips extracted. Running pose extraction ...")

    with DatasetLoader() as loader:
        loader.process_include()

    if not keep_extracted and extract_root.exists():
        shutil.rmtree(extract_root)
        logger.info(f"Deleted extracted folder: {extract_root}")

    logger.info("INCLUDE preparation complete.")


def prepare_islvt(keep_extracted: bool, keep_zips: bool) -> None:
    with DatasetLoader() as loader:
        loader.process_islvt()
    logger.info("ISLVT preparation complete.")


def prepare_isl_dataset(keep_extracted: bool, keep_zips: bool) -> None:
    zip_path = settings.isl_dataset_dir / "ISL.zip"
    if not zip_path.exists():
        logger.error(f"ISL.zip not found at {zip_path}.")
        sys.exit(1)

    with DatasetLoader() as loader:
        loader.process_isl_dataset()

    if not keep_zips:
        zip_path.unlink()
        logger.info("Deleted ISL.zip after processing.")

    logger.info("ISL_Dataset preparation complete.")


def main() -> None:
    args = parse_args()

    dispatch = {
        "include": prepare_include,
        "islvt": prepare_islvt,
        "isl_dataset": prepare_isl_dataset,
    }

    dispatch[args.dataset](
        keep_extracted=args.keep_extracted,
        keep_zips=args.keep_zips,
    )


if __name__ == "__main__":
    main()

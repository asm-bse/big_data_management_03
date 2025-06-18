#!/usr/bin/env python
"""
landing.py – Raw-data ingestion & preview for Lab 3 (Data Management Backbone)
=============================================================================
Reads three source datasets (Idealista JSON, Citizen Incidences CSV, Family Income CSV)
from a local folder or S3, and optionally copies them into the project’s Landing zone,
preserving their original formats. Supports preview of schema and counts.

Usage examples:
  # Preview local landing data
  uv run python -- landing.py \
      --source-base zones/landing \
      --mode preview

  # Copy raw files from raw source into landing
  uv run python -- landing.py \
      --source-base zones/raw \
      --landing-base zones/landing \
      --mode copy

Args:
  --source-base, -s    Base path for original raw files (local dir or s3 URI)
  --landing-base, -l   Target landing zone path (local dir or s3 URI)
  --mode, -m           Operation mode: 'preview' or 'copy' (default: preview)

Requirements:
  - PySpark for preview mode
  - shutil/os for copy mode
"""
import argparse
import logging
import shutil
import os
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")


def build_spark():
    spark = (
        SparkSession.builder
            .appName("BDM-Lab3-Landing")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def preview_datasets(spark, base):
    # Idealista JSON
    path_json = os.path.join(base, 'idealista', '*.json')
    df_json = spark.read.json(path_json)
    logging.info("=== Preview: idealista ===")
    df_json.printSchema()
    logging.info("Rows: %d", df_json.count())
    df_json.show(5, truncate=False)

    # Incidences CSV
    path_inc = os.path.join(base, 'incidences', '*.csv')
    df_inc = (
        spark.read
             .option('header', True)
             .csv(path_inc)
    )
    logging.info("=== Preview: incidences ===")
    df_inc.printSchema()
    logging.info("Rows: %d", df_inc.count())
    df_inc.show(5, truncate=False)

    # Income CSV
    path_incme = os.path.join(base, 'income', '*.csv')
    df_incme = (
        spark.read
             .option('header', True)
             .csv(path_incme)
    )
    logging.info("=== Preview: income ===")
    df_incme.printSchema()
    logging.info("Rows: %d", df_incme.count())
    df_incme.show(5, truncate=False)


def copy_raw_files(src_base, dst_base):
    for ds in ['idealista', 'incidences', 'income']:
        src_dir = os.path.join(src_base, ds)
        dst_dir = os.path.join(dst_base, ds, 'raw')
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            src_file = os.path.join(src_dir, fname)
            dst_file = os.path.join(dst_dir, fname)
            shutil.copy2(src_file, dst_file)
            logging.info("Copied %s → %s", src_file, dst_file)


def main():
    parser = argparse.ArgumentParser(description="Landing zone ingestion & preview")
    parser.add_argument('--source-base', '-s', required=True,
                        help='Base folder or URI for raw source files')
    parser.add_argument('--landing-base', '-l', default='./project_data/landing',
                        help='Target landing zone path')
    parser.add_argument('--mode', '-m', choices=['preview', 'copy'], default='preview',
                        help="'preview' to show schema/counts; 'copy' to copy raw files")
    args = parser.parse_args()

    if args.mode == 'preview':
        spark = build_spark()
        preview_datasets(spark, args.source_base)
        spark.stop()
    else:
        copy_raw_files(args.source_base, args.landing_base)


if __name__ == '__main__':
    main()

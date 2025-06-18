#!/usr/bin/env python
"""
formatted.py – Data formatting & transformation for Lab 3 (Data Management Backbone)
===============================================================================
Reads raw datasets from Landing zone (local directories), applies schema alignment,
partitioning, and writes unified Parquet tables into the project’s Formatted zone.
Also supports preview mode to display schemas and record samples.

Usage examples:
  # Preview formatted output without writing
  uv run python -- formatted.py \
      --landing-base zones/landing \
      --formatted-base zones/formatted_preview \
      --format parquet \
      --preview

  # Write formatted data and overwrite
  uv run python -- formatted.py \
      --landing-base zones/landing \
      --formatted-base zones/formatted \
      --format parquet --overwrite
"""
import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, expr

# Configure root logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")


def build_spark() -> SparkSession:
    spark = (
        SparkSession.builder
            .appName("BDM-Lab3-Formatted")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_landing(spark: SparkSession, landing_base: str):
    paths = {
        "idealista": f"{landing_base}/idealista/*.json",
        "incidences": f"{landing_base}/incidences/*.csv",
        "income": f"{landing_base}/income/*.csv",
    }
    dfs = {}
    dfs["idealista"] = spark.read.json(paths["idealista"])
    csv_opts = {"header": "true", "multiLine": "true", "encoding": "UTF-8"}
    dfs["incidences"] = spark.read.options(**csv_opts).csv(paths["incidences"])
    dfs["income"]     = spark.read.options(**csv_opts).csv(paths["income"])
    return dfs


def preview_dfs(dfs: dict):
    for name, df in dfs.items():
        logging.info(f"=== Preview: {name.capitalize()} ===")
        df.printSchema()
        logging.info("Rows: %d", df.count())
        df.show(5, truncate=False)


def transform_and_write(dfs, formatted_base: str, fmt: str, overwrite: bool):
    mode = "overwrite" if overwrite else "errorifexists"

    # Idealista: cast numeric columns
    df_ideal = (
        dfs['idealista']
        .withColumn('price', col('price').cast('double'))
        .withColumn('size', col('size').cast('double'))
        .withColumn('priceByArea', col('priceByArea').cast('double'))
    )
    df_ideal.write.format(fmt).mode(mode).save(f"{formatted_base}/idealista")

    # Incidences: parse dates and types
    df_inc = (
        dfs['incidences']
        .withColumn('opened_date', to_date(col('ANY_DATA_ALTA'), 'yyyy'))
        .withColumn('closed_date', to_date(col('ANY_DATA_TANCAMENT'), 'yyyy'))
        .withColumn('year', year(col('opened_date')))
    )
    df_inc.write.format(fmt).mode(mode).partitionBy('year')\
        .save(f"{formatted_base}/incidences")

    # Income: cast year and index with try_cast to handle malformed values
    df_incme = (
        dfs['income']
        .withColumn('year', col('Any').cast('int'))
        .withColumn('avg_income', expr("try_cast(`Índex RFD Barcelona = 100` as double)"))
    )
    df_incme.write.format(fmt).mode(mode).partitionBy('year')\
        .save(f"{formatted_base}/income")

    logging.info("Formatted data written to %s (format=%s)", formatted_base, fmt)


def main():
    parser = argparse.ArgumentParser(
        description="Format raw datasets into Parquet tables."
    )
    parser.add_argument(
        "--landing-base", "-l", required=True,
        help="Base path for raw landing data (local or URI)"
    )
    parser.add_argument(
        "--formatted-base", "-f", required=True,
        help="Target path for formatted data"
    )
    parser.add_argument(
        "--format", choices=["parquet","delta"], default="parquet",
        help="Output table format (default: parquet)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing formatted data"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Preview schemas and sample rows without writing"
    )
    args = parser.parse_args()

    spark = build_spark()
    dfs = read_landing(spark, args.landing_base)
    if args.preview:
        preview_dfs(dfs)
    else:
        transform_and_write(dfs, args.formatted_base, args.format, args.overwrite)
    spark.stop()


if __name__ == "__main__":
    main()

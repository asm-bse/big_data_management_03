#!/usr/bin/env python
"""
formatted.py – Data formatting & transformation for Lab 3 (Data Management Backbone)
==================================================================================
This script implements the second stage of the Data Management Backbone.
Key steps:
1.  Reads raw datasets (Idealista, Incidences, Income) from the Landing zone.
2.  For Idealista data, it extracts the 'year' from the source filename, safely handling
    files that do not match the expected pattern.
3.  Reads lookup tables for data reconciliation.
4.  Cleans join keys and enriches data with a unified 'district_id'.
5.  Partitions all datasets by 'year' for efficient querying.
6.  Writes the final, cleaned, and enriched datasets into the Formatted zone.
"""
import argparse
import logging
import re
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, to_date, year, expr, concat, lit, lpad, trim, lower, regexp_replace, input_file_name, regexp_extract

# Configure root logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")


def build_spark() -> SparkSession:
    return SparkSession.builder.appName("BDM-Lab3-Formatted").getOrCreate()


def read_landing(spark: SparkSession, landing_base: str) -> dict[str, DataFrame]:
    logging.info("Reading raw datasets from Landing Zone: %s", landing_base)
    paths = {
        "incidences": f"{landing_base}/incidences/*.csv",
        "income": f"{landing_base}/income/*.csv",
    }
    dfs = {}
    csv_opts = {"header": "true", "multiLine": "true", "encoding": "UTF-8", "inferSchema": "true"}
    for name, path in paths.items():
        dfs[name] = spark.read.options(**csv_opts).csv(path)
    
    # Special handling for idealista to extract year from filename
    idealista_path = f"{landing_base}/idealista/*.json"
    dfs["idealista"] = spark.read.json(idealista_path).withColumn("filename", input_file_name())
    
    return dfs

def read_lookup_tables(spark: SparkSession, lookup_base: str) -> dict[str, DataFrame]:
    logging.info("Reading lookup tables from: %s", lookup_base)
    path = f"{lookup_base}/Income OpenDataBCN Extended.csv"
    lookup_df = spark.read.options(header="true", inferSchema="true").csv(path)
    return {"common_lookup": lookup_df}


def clean_district_name(district_col: col) -> col:
    return lower(trim(regexp_replace(district_col, "\\s*-\\s*", "-")))


def transform_and_write(dfs: dict, lookups: dict, formatted_base: str, fmt: str, overwrite: bool):
    mode = "overwrite" if overwrite else "errorifexists"
    logging.info("Starting transformation and writing process...")
    
    lookup_df = lookups['common_lookup'].select(
        col("district").alias("lookup_district_raw"),
        "district_id"
    ).distinct().withColumn("district_key", clean_district_name(col("lookup_district_raw")))

    # --- 1. Idealista ---
    df_ideal = dfs['idealista']
    # Extract year from filename using regex
    df_ideal_with_year_str = df_ideal.withColumn("year_str", regexp_extract(col("filename"), r"(\d{4})_\d{2}_\d{2}_idealista.json", 1))
    
    # Filter out rows where the year could not be extracted (empty string) BEFORE casting
    df_ideal_filtered = df_ideal_with_year_str.filter(col("year_str") != "")
    
    # Now, safely cast the extracted string to an integer
    df_ideal_with_year = df_ideal_filtered.withColumn("year", col("year_str").cast("int"))

    df_ideal_enriched = df_ideal_with_year.withColumn("district_key", clean_district_name(col("district"))) \
        .join(lookup_df, "district_key", "left") \
        .drop("district_key", "lookup_district_raw", "filename", "year_str")
    
    df_ideal_enriched.write.format(fmt).mode(mode).partitionBy("year").save(f"{formatted_base}/idealista")
    logging.info("Written 'idealista' to Formatted Zone, partitioned by year.")

    # --- 2. Incidences ---
    df_inc = dfs['incidences']
    df_inc_with_date = df_inc.withColumn("opened_date", to_date(concat(col("ANY_DATA_ALTA"), lit("-"), lpad(col("MES_DATA_ALTA"), 2, "0"), lit("-"), lpad(col("DIA_DATA_ALTA"), 2, "0"))))
    df_inc_with_year = df_inc_with_date.withColumn('year', year(col('opened_date')))
    
    df_inc_enriched = df_inc_with_year.withColumn("district_key", clean_district_name(col("DISTRICTE"))) \
        .join(lookup_df, "district_key", "left") \
        .drop("district_key", "lookup_district_raw")
        
    df_inc_enriched.write.format(fmt).mode(mode).partitionBy('year').save(f"{formatted_base}/incidences")
    logging.info("Written 'incidences' to Formatted Zone, partitioned by year.")

    # --- 3. Income ---
    df_incme = dfs['income']
    df_incme_enriched = df_incme.withColumn("district_key", clean_district_name(col("Nom_Districte"))) \
        .withColumnRenamed("Any", "year") \
        .join(lookup_df, "district_key", "left") \
        .drop("district_key", "lookup_district_raw")

    df_incme_enriched.write.format(fmt).mode(mode).partitionBy('year').save(f"{formatted_base}/income")
    logging.info("Written 'income' to Formatted Zone, partitioned by year.")

    logging.info("Formatted data written successfully.")

def main():
    parser = argparse.ArgumentParser(description="Format raw datasets.")
    parser.add_argument("--landing-base", "-l", required=True)
    parser.add_argument("--lookup-base", "-k", required=True)
    parser.add_argument("--formatted-base", "-f", required=True)
    parser.add_argument("--format", default="parquet")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")
    
    dfs = read_landing(spark, args.landing_base)
    lookups = read_lookup_tables(spark, args.lookup_base)
    transform_and_write(dfs, lookups, args.formatted_base, args.format, args.overwrite)
    
    spark.stop()

if __name__ == "__main__":
    main()

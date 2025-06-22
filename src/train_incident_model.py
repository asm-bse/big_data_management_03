#!/usr/bin/env python
"""
ml_pipeline.py - ML Model Training and Management for Lab 3 (Data Analysis Backbone)
=====================================================================================
This script implements the B.2 Predictive Analysis task. It reads the aggregated
master table from the Exploitation Zone and performs the following steps to predict
the number of incidents in a district.

1.  **Load Data**: Reads the 'district_kpis' table.
2.  **Data Preparation**:
    -   Filters data to keep only rows where both the target ('incidence_count') and a key
        feature ('avg_income_index') are present.
    -   Imputes any remaining missing values in other feature columns using a general Imputer.
3.  **Feature Engineering**: Assembles relevant features (income, property stats) into a vector.
4.  **Hyperparameter Tuning & Model Training**:
    -   Splits the data into training and validation sets.
    -   Defines a parameter grid for each regression model (Linear Regression, Random Forest).
    -   Uses Cross-Validation to find the best hyperparameters based on the RMSE metric.
5.  **MLflow Tracking**:
    -   Logs parameters, the best cross-validated metrics (RMSE), and the best trained model to MLflow.
6.  **Model Management**:
    -   Identifies the best overall model across all types based on the lowest RMSE.
    -   Registers the best model in the MLflow Model Registry and promotes it to 'Production'.

Usage (typically run via an orchestrator like Airflow):
  python ml_pipeline.py \
      --exploitation-path /path/to/your/exploitation_zone \
      --mlflow-uri http://mlflow:5000
"""
import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.spark

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main(exploitation_path, mlflow_uri):
    """Main function to run the ML pipeline."""

    # --- Setup MLflow ---
    logging.info(f"Setting MLflow tracking URI to: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "District_Incidence_Prediction"
    mlflow.set_experiment(experiment_name)
    logging.info(f"Using MLflow experiment: '{experiment_name}'")

    # --- Spark Session ---
    spark = SparkSession.builder.appName("DistrictIncidencePrediction").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- 1. Load Data ---
    data_path = f"{exploitation_path}/district_kpis"
    logging.info(f"Loading data from: {data_path}")
    df = spark.read.parquet(data_path)

    # --- 2. Data Preparation ---
    logging.info("Preparing data for model training...")
    # Filter for rows where both target and a key feature are present
    df = df.dropna(subset=["incidence_count", "avg_income_index"])
    rows_count = df.count()
    logging.info(f"Rows after filtering for valid target and features: {rows_count}")
    if rows_count < 50:
        logging.warning("Dataset size is very small. Model performance metrics may not be reliable.")

    # --- 3. Feature Engineering ---
    # Define features to predict incidence_count.
    feature_cols = ["avg_income_index", "avg_price", "avg_size", "avg_price_per_sqm", "property_count"]
    
    imputer = Imputer(inputCols=feature_cols, outputCols=[f"{c}_imputed" for c in feature_cols], strategy="mean")
    assembler = VectorAssembler(inputCols=[f"{c}_imputed" for c in feature_cols], outputCol="features")
    
    # --- 4. Train/Test Split ---
    train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)
    logging.info(f"Data split into training ({train_df.count()} rows) and validation ({val_df.count()} rows).")

    # --- 5. Model Training and Hyperparameter Tuning ---
    # Define Regression models
    lr = LinearRegression(featuresCol="features", labelCol="incidence_count")
    rf = RandomForestRegressor(featuresCol="features", labelCol="incidence_count", seed=42)
    
    # Define Parameter Grids for Cross-Validation
    lr_param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .addGrid(lr.elasticNetParam, [0.5, 0.8]) \
        .build()

    rf_param_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()
        
    models_and_grids = {
        "LinearRegression": (lr, lr_param_grid),
        "RandomForestRegressor": (rf, rf_param_grid)
    }
    
    # Use RegressionEvaluator for regression metrics like RMSE
    evaluator = RegressionEvaluator(labelCol="incidence_count", predictionCol="prediction", metricName="rmse")
    
    best_rmse = float('inf')
    best_model_run_id = None

    for name, (model, param_grid) in models_and_grids.items():
        logging.info(f"--- Tuning and Training model: {name} ---")
        with mlflow.start_run(run_name=f"Tune_and_Train_{name}") as run:
            
            pipeline = Pipeline(stages=[imputer, assembler, model])

            cv = CrossValidator(estimator=pipeline,
                                estimatorParamMaps=param_grid,
                                evaluator=evaluator,
                                numFolds=3)
            
            cv_model = cv.fit(train_df)
            predictions = cv_model.transform(val_df)
            
            rmse = evaluator.evaluate(predictions)
            logging.info(f"Best '{name}' model -> RMSE: {rmse:.4f}")
            
            mlflow.log_param("model_type", name)
            mlflow.log_metric("rmse", rmse)
            
            best_pipeline_model = cv_model.bestModel
            model_stage = best_pipeline_model.stages[-1]
            params_to_log = {f"best_{p.name}": v for p, v in model_stage.extractParamMap().items() if p.name in [gp.name for gp in model.params]}
            mlflow.log_params(params_to_log)

            input_example = train_df.limit(5).toPandas()
            signature = infer_signature(input_example[feature_cols], predictions.select("prediction").toPandas())

            mlflow.spark.log_model(spark_model=best_pipeline_model, artifact_path="model", signature=signature)
            
            run_id = run.info.run_id
            logging.info(f"Run {run_id} for model '{name}' finished.")
            
            # The best model has the lowest RMSE
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_run_id = run_id

    logging.info(f"Best overall model is from run ID '{best_model_run_id}' with RMSE: {best_rmse:.4f}")

    # --- 6. Model Registration and Promotion ---
    if best_model_run_id:
        model_name = "DistrictIncidencePredictor"
        model_uri = f"runs:/{best_model_run_id}/model"
        
        logging.info(f"Registering best model under name: '{model_name}'")
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        
        client = MlflowClient()
        latest_version = registered_model.version
        
        logging.info(f"Promoting model version {latest_version} to 'Production'")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        logging.info(f"Model '{model_name}' version {latest_version} is now in Production.")
    else:
        logging.warning("No model was trained successfully. Skipping registration.")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML model training pipeline.")
    parser.add_argument("--exploitation-path", default="/opt/airflow/zones/exploitation",
                        help="Base path for the Exploitation Zone data.")
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000",
                        help="MLflow tracking server URI.")
    args = parser.parse_args()
    main(args.exploitation_path, args.mlflow_uri)

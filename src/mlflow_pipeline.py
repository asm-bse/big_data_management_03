from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import when
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.spark
import pandas as pd

# Set tracking URI and experiment
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("housing_incidents")

# Spark session
spark = SparkSession.builder.appName("IncidentPrediction").getOrCreate()

print("📥 Loading data from /opt/airflow/zones/exploitation/idealista ...")
df = spark.read.parquet("/opt/airflow/zones/exploitation/idealista")

print("🔍 Schema before cleaning:")
df.printSchema()
print(f"🔢 Total rows before cleaning: {df.count()}")

# Drop rows with missing key features
df = df.dropna(subset=["price", "rooms", "size", "price_per_room"])
print(f"✅ Rows after dropping nulls: {df.count()}")

# Label: 1 if expensive per room, else 0
df = df.withColumn("label", when(df["price_per_room"] > 100000, 1).otherwise(0))

# Feature engineering
assembler = VectorAssembler(
    inputCols=["price", "rooms", "size", "price_per_room"],
    outputCol="features"
)
df = assembler.transform(df)

print("🧱 Features assembled. Schema now:")
df.printSchema()

# Split into train/test
train, val = df.randomSplit([0.8, 0.2], seed=42)
print(f"📊 Train size: {train.count()} rows")
print(f"📊 Validation size: {val.count()} rows")

# Evaluation metric
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Models to try
models = {
    "LogisticRegression": LogisticRegression(maxIter=10, featuresCol="features", labelCol="label"),
    "DecisionTree": DecisionTreeClassifier(maxDepth=5, featuresCol="features", labelCol="label")
}

best_score = 0.0
best_model = None
best_run_id = None

for name, model in models.items():
    print(f"\n🚀 Training model: {name}")
    with mlflow.start_run(run_name=name) as run:
        pipeline = Pipeline(stages=[model])
        fitted_model = pipeline.fit(train)

        predictions = fitted_model.transform(val)
        pred_count = predictions.count()
        print(f"📈 Predictions generated: {pred_count}")

        accuracy = evaluator.evaluate(predictions)
        print(f"🎯 {name} accuracy: {accuracy:.4f}")

        # Infer signature only if predictions are valid
        pred_pdf = predictions.select("prediction").toPandas()
        if pred_pdf["prediction"].notna().any():
            input_example = df.select("price", "rooms", "size", "price_per_room").limit(1).toPandas()
            signature = infer_signature(df.select("price", "rooms", "size", "price_per_room").toPandas(), pred_pdf)
        else:
            raise ValueError("⚠️ Predictions contain only None values. Check model/data input.")

        # Log to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.spark.log_model(
            spark_model=fitted_model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        print(f"🔗 View run {name} at: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

        if accuracy > best_score:
            best_score = accuracy
            best_model = fitted_model
            best_run_id = run.info.run_id

print(f"\n🏆 Best model is from run {best_run_id} with accuracy = {best_score:.4f}")

# Register and promote best model
model_uri = f"runs:/{best_run_id}/model"
model_details = mlflow.register_model(model_uri=model_uri, name="BestIncidentModel")

client = MlflowClient()
latest_version = client.get_latest_versions("BestIncidentModel", stages=["None"])[0].version
client.transition_model_version_stage(
    name="BestIncidentModel",
    version=latest_version,
    stage="Production",
    archive_existing_versions=True
)

print(f"🚀 BestIncidentModel version {latest_version} is now in Production.")

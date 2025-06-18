
# Housing Incidents Prediction Pipeline with Airflow, Spark, and MLflow

This project builds an end-to-end machine learning pipeline for predicting housing-related incidents based on property features. The pipeline is orchestrated using Apache Airflow, powered by PySpark for data processing and MLflow for model tracking and deployment.

## 🚀 Project Overview

- **Data Source**: Real estate and income datasets stored in local Parquet format.
- **ML Task**: Binary classification — whether a house is in a high-price-per-room zone.
- **Models Used**: Logistic Regression and Decision Tree (Spark MLlib).
- **Orchestration**: Apache Airflow DAG (`ml_pipeline_dag.py`).
- **Model Logging & Deployment**: MLflow Tracking Server with local backend store.

## 📁 Project Structure

```
.
├── airflow/                   # Airflow DAGs, logs, plugins
│   └── dags/ml_pipeline_dag.py
├── best_model/               # Exported model artifacts
├── docker-compose.yml        # Docker services (Airflow, MLflow)
├── Dockerfile.airflow        # Custom Dockerfile for Airflow service
├── mlruns/                   # MLflow local run history
├── src/
│   └── mlflow_pipeline.py    # Core training pipeline
├── zones/                    # Raw, formatted, exploitation data zones
├── requirements.txt          # Python dependencies
├── README.md                 # This file
```

## 🛠️ How to Run the Project

### 1. Build and Start Docker Containers

Run the following commands to set up the services:

```bash
docker compose down -v        # Clean previous volumes and containers
docker compose up --build     # Build and launch MLflow + Airflow
```

This will spin up:

- An **MLflow Tracking Server** at http://localhost:5050
- An **Airflow Web UI** at http://localhost:8080

### 2. Access Airflow

Visit http://localhost:8080 and log in with:

- **Username**: `admin`
- **Password**: `admin`

Trigger the DAG named `mlflow_pipeline_dag`.

### 3. What the DAG Does

- Loads dataset from the exploitation zone (`zones/exploitation/idealista`)
- Trains two models with Spark MLlib
- Evaluates accuracy
- Logs metrics and models to MLflow
- Promotes the best model to Production stage

## ✅ Features

- End-to-end ML lifecycle automation
- Local MLflow server with Spark model logging
- Clear model comparison based on accuracy
- Model promotion to production-ready stage
- Dockerized for easy reproducibility

## 📦 Dependencies

Dependencies are automatically installed inside the Docker container using the provided `Dockerfile.airflow` and `requirements.txt`.

## ✍️ Authors

- **Aleksandr Smolin**
- **Maria Simakova**

## 📄 License

MIT License

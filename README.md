
# Housing Incidents Prediction Pipeline with Airflow, Spark, and MLflow

This project builds an end-to-end machine learning pipeline for predicting housing-related incidents based on property features. The pipeline is orchestrated using Apache Airflow, powered by PySpark for data processing and MLflow for model tracking and deployment.

## Project Overview

- **Data Source**: The pipeline ingests and integrates three distinct datasets:

1. Idealista: Real estate listings (JSON format).
2. Incidences: Public incident reports from OpenData BCN (CSV format).
3. Income: Yearly income statistics per district from OpenData BCN (CSV format).

- **ML Tasks**: Two regression models are trained in parallel to solve different business problems:

1. Incident Prediction: Predicts the number of public incidents (incidence_count) in a district based on socio-economic and real estate data.
2. Income Prediction: Predicts the average income index (avg_income_index) of a district based on property market and safety metrics.

- **Models Used**: Logistic Regression and Decision Tree (Spark MLlib).
- **Orchestration**: Apache Airflow DAG (`full_data_science_pipeline_dag.py`).
- **Model Logging & Deployment**: MLflow Tracking Server with local backend storage.

## Project Structure

```
.
├── airflow/
│   └── dags/
│       └── full_data_science_pipeline_dag.py
├── docker-compose.yml
├── Dockerfile.airflow
├── mlruns/
├── src/
│   ├── formatted.py
│   ├── exploitation.py
│   ├── train_incident_model.py
│   └── train_income_model.py
├── zones/
│   ├── lookup/
│   ├── landing/
│   ├── formatted/
│   └── exploitation/
├── requirements.txt
└── README.md

```

## How to Run the Project

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

Trigger the DAG named `full_data_science_pipeline_dag`.

### 3. What the DAG Does

The full_data_science_pipeline_dag orchestrates the entire project workflow:
 - run_formatted_pipeline: Executes the formatted.py script. It reads raw data from the landing zone, cleans it, extracts the year from Idealista filenames, and enriches all datasets with a unified district_id. The output is saved to the formatted zone, partitioned by year.
 - run_exploitation_pipeline: Executes exploitation.py. It takes the formatted data, creates yearly aggregated KPIs for each district, and joins them into a single master table (district_kpis) in the exploitation zone.

Parallel Model Training: Once the data is prepared, two tasks run in parallel:

 - run_train_incident_model: Executes train_incident_model.py to predict incident counts.
 - run_train_income_model: Executes train_income_model.py to predict the income index.

Each ML script uses Cross-Validation to find the best hyperparameters, logs all results to MLflow, and promotes the best-performing model to the "Production" stage in the MLflow Model Registry.

## Key Pipeline Features

 - End-to-End Orchestration: A single Airflow DAG manages the entire data flow, from raw files to production-ready models, ensuring data dependencies and execution order are respected.
 - Data Reconciliation: Implements robust logic to handle inconsistencies in district names across different data sources, using cleaning functions and a common lookup table.
 - Temporal Data Alignment: Cleverly extracts year information from filenames (idealista) to enable powerful time-based analysis and aggregation.
 - Advanced Imputation: Fills missing data points not just with a global average, but with more accurate, context-aware values like the district-specific mean calculated using Spark's Window functions.
 - Robust Model Evaluation: Uses CrossValidator for hyperparameter tuning, providing a more reliable measure of model performance and preventing overfitting.
 - Dual ML Use-Cases: Demonstrates the flexibility of the data platform by training two different models for two different prediction tasks from the same master data table.
 - Complete MLOps Cycle: Integrates MLflow to track experiments, compare models, and manage the model lifecycle by automatically registering and deploying the best version.

## Dependencies

Dependencies are automatically installed inside the Docker container using the provided `Dockerfile.airflow` and `requirements.txt`.

## Authors

- **Aleksandr Smolin**
- **Maria Simakova**


from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 6, 22),
    'retries': 1,
}

with DAG(
    dag_id='full_data_science_pipeline_dag',
    default_args=default_args,
    description='A complete DAG to run the data formatting, exploitation, and ML model training pipelines.',
    schedule_interval=None,  # This DAG will only be triggered manually
    catchup=False,
    tags=['data_management', 'ml_pipeline'],
) as dag:

    # Task 1: Run the Formatted Zone script
    # This task cleans the raw data and enriches it with district IDs.
    run_formatted_pipeline = BashOperator(
        task_id='run_formatted_pipeline',
        bash_command=(
            'python /opt/airflow/src/formatted.py '
            '--landing-base /opt/airflow/zones/landing '
            '--lookup-base /opt/airflow/zones/lookup '
            '--formatted-base /opt/airflow/zones/formatted '
            '--format parquet --overwrite'
        )
    )

    # Task 2: Run the Exploitation Zone script
    # This task aggregates the formatted data into a master KPI table.
    run_exploitation_pipeline = BashOperator(
        task_id='run_exploitation_pipeline',
        bash_command=(
            'python /opt/airflow/src/exploitation.py '
            '--formatted-base /opt/airflow/zones/formatted '
            '--exploitation-base /opt/airflow/zones/exploitation '
            '--format parquet --overwrite'
        )
    )

    # Task 3: Train the Incident Prediction Model
    # This task runs the first ML pipeline to predict incident counts.
    train_incident_model_task = BashOperator(
        task_id='run_train_incident_model',
        bash_command='python /opt/airflow/src/train_incident_model.py'
    )

    # Task 4: Train the Income Prediction Model
    # This task runs the second ML pipeline to predict the income index.
    train_income_model_task = BashOperator(
        task_id='run_train_income_model',
        bash_command='python /opt/airflow/src/train_income_model.py'
    )

    # --- Define Task Dependencies ---
    # The formatted pipeline must run before the exploitation pipeline.
    # Both ML models can run in parallel, but only after the exploitation pipeline is complete.
    
    run_formatted_pipeline >> run_exploitation_pipeline >> [train_incident_model_task, train_income_model_task]


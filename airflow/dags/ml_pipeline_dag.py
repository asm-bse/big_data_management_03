from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'anus',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    'mlflow_pipeline_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    run_ml_pipeline = BashOperator(
        task_id='run_ml_pipeline',
        bash_command='python /opt/airflow/mlflow_pipeline.py'
    )

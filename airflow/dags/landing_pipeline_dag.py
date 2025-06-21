from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'a',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    'landing_pipeline_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    run_ml_pipeline = BashOperator(
        task_id='run_landing_pipeline',
        bash_command='python /opt/airflow/src/landing.py --source-base /opt/airflow/zones/raw --landing-base /opt/airflow/zones/landing --mode copy'

    )

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'a',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    'formatted_pipeline_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    run_ml_pipeline = BashOperator(
        task_id='run_formatted_pipeline',
        bash_command='python /opt/airflow/src/formatted.py --landing-base /opt/airflow/zones/landing --formatted-base /opt/airflow/zones/formatted --format parquet --overwrite'

    )

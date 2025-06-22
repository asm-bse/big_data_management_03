from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'a',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    'train_income_model_dag',
    default_args=default_args,
    schedule_interval=None,
    is_paused_upon_creation=True,
    catchup=False
) as dag:

    run_ml_pipeline = BashOperator(
        task_id='train_income_model',
        bash_command='python /opt/airflow/src/train_income_model.py'
    )

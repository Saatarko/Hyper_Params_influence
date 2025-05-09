from datetime import datetime
from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import sys
import os

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === DAG конфигурация ===
with DAG(
    dag_id="sintetic_generate",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["hyper"]
) as dag:

    sintetic_generate = BashOperator(
        task_id="sintetic_generate",
        bash_command="cd /home/saatarko/PycharmProjects/Hyper_Params_influence && dvc repro generate_sintetic_data_stage",
        doc_md = "**Генерация синтетичских данных для тестов**"
    )


    sintetic_generate
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
    dag_id="run_transfer_learning",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["image"]
) as dag:

    run_transfer_learning = BashOperator(
        task_id="run_transfer_learning",
        bash_command="cd /home/saatarko/PycharmProjects/Hyper_Params_influence && dvc repro run_transfer_learning_stage",
        doc_md = "**Проведение исследований при обучении**"
    )


    run_transfer_learning
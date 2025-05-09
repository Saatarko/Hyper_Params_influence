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
    dag_id="search_train_nn_class",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["hyper"]
) as dag:

    search_train_nn_class = BashOperator(
        task_id="search_train_nn_class",
        bash_command="cd /home/saatarko/PycharmProjects/Hyper_Params_influence && dvc repro optuna_search_synt_class_stage",
        doc_md = "**Проведение исследований при классификации**"
    )


    search_train_nn_class
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
    dag_id="train_cnn_dag",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["image"]
) as dag:

    train_cnn_dag = BashOperator(
        task_id="train_cnn_dag",
        bash_command="cd /home/saatarko/PycharmProjects/Hyper_Params_influence && dvc repro train_cnn_dag_stage",
        doc_md = "**Проведение обучение модели для классификации изображения котиков и собачек с аугментацией**"
    )


    train_cnn_dag
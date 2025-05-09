import os
from pathlib import Path

import mlflow
import numpy as np
import yaml
from matplotlib import pyplot as plt
mlflow.set_tracking_uri('http://localhost:5000')

def load_vectors_npz(path: Path, key: str = "vectors") -> np.ndarray:
    return np.load(path)[key]


def get_project_paths():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    return {
        "project_root": PROJECT_ROOT,
        "raw_dir": PROJECT_ROOT / paths["raw_data"],
        "processed_dir": PROJECT_ROOT / paths["processed_data"],
        "models_dir": PROJECT_ROOT / paths["models_dir"],
        "scripts_dir": PROJECT_ROOT / paths["scripts"],
        "image_dir": PROJECT_ROOT / paths["image_dir"],
        "vectors_dir": PROJECT_ROOT / paths["vectors_dir"],
        "logs_dir": PROJECT_ROOT / paths["logs_dir"]
    }

def get_project_root():
    """Возвращает абсолютный путь к корню проекта."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def plot_losses(train_losses, val_losses, model_name_tag)->str:
    """
    Функция готовит графики для сохранения/передачи в mlflow
    :param train_losses: Функция потерь при обучении
    :param val_losses: Функция потерь при валидации
    :param model_name_tag:  название текущего графика
    :return: Путь к изображению
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    plot_path =  paths["image_dir"] /f"training_loss_curve_{model_name_tag}.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path  # Возвращаем путь к сохранённому файлу
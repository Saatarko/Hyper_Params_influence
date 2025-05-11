import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import seaborn as sns
import mlflow
import numpy as np
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

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


# Глобальный список активаций для группировки


def plot_training_results(class_mode=False):
    activation_keywords = ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "swish", "mish"]
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()

    image_dir = paths["image_dir"]
    metrics_dir = paths["models_dir"]

    # 1. Собираем изображения
    all_images = list(image_dir.glob("*.png"))

    # 2. Фильтрация по наличию или отсутствию слова 'Class'
    if class_mode:
        images = [img for img in all_images if "Class" in img.name]
    else:
        images = [img for img in all_images if "Class" not in img.name]

    # 3. Группировка по активационной функции
    groups = defaultdict(list)
    for img in images:
        for act in activation_keywords:
            if act in img.name:
                groups[act].append(img)
                break
        else:
            groups["unknown"].append(img)

    # 4. Сортировка внутри каждой группы по длине имени
    for key in groups:
        groups[key].sort(key=lambda x: len(x.name))

    # 5. Отображение
    for act_func, imgs in groups.items():
        print(f"\n=== Activation: {act_func.upper()} ===\n")
        n = len(imgs)
        rows = (n + 2) // 3  # по 3 в строку
        fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
        axes = axes.flatten() if n > 1 else [axes]

        for ax in axes[len(imgs):]:
            ax.axis('off')

        for ax, img_path in zip(axes, imgs):
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(img_path.name, fontsize=10)

            # Пытаемся найти соответствующий JSON
            base_name = img_path.name.replace("training_loss_curve_", "").replace(".png", "")
            json_name = f"{base_name}_train_metrics.json"
            json_path = metrics_dir / json_name

            if json_path.exists():
                with open(json_path) as f:
                    metrics = json.load(f)
                # Формируем текст метрик
                info = f"Best val: {metrics.get('best_val_loss', 'NA'):.4f}\nTrain: {metrics.get('final_train_loss', 'NA'):.4f}\nVal: {metrics.get('final_val_loss', 'NA'):.4f}"
            else:
                info = "No metrics"

            ax.text(0.5, -0.15, info, ha="center", va="top", fontsize=9, transform=ax.transAxes)

        plt.tight_layout()
        plt.show()


def log_confusion_matrix(all_preds, all_labels, model_name_tag):

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()

    # Вычисление матрицы ошибок
    cm = confusion_matrix(all_labels, all_preds)

    # Визуализация матрицы ошибок
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plot_path = paths["image_dir"]/f"{model_name_tag}_confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)  # Логирование матрицы ошибок
    plt.close()


def plot_matrix():
    # Пути
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    paths = get_project_paths()
    image_dir = paths["image_dir"]

    # Загрузка таблицы метрик
    metrics_df = pd.read_csv(paths['raw_dir']/"runs_aug.csv")

    # Сбор изображений
    all_images = list(image_dir.glob("*.png"))
    images = [img for img in all_images if "optuna_confmat" in img.name]

    # Сортируем по номеру эксперимента (из имени)
    def extract_trial_number(path):
        # Например: "optuna_confmat_59_confusion_matrix.png" → 59
        try:
            return int(path.name.split("_")[2])
        except (IndexError, ValueError):
            return -1

    images = sorted(images, key=extract_trial_number)

    n = len(images)
    rows = (n + 2) // 3  # по 3 в строку
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax in axes[n:]:
        ax.axis('off')

    for ax, img_path in zip(axes, images):
        trial_num = extract_trial_number(img_path)

        # Извлекаем строку с метриками
        row = metrics_df[metrics_df["model_name_tag"] == f"optuna_aug_trial_{trial_num}"]
        if not row.empty:
            row_data = row.iloc[0]
            info = (
                f"Эксперимент {trial_num}\n"
                f"Accuracy: {row_data['accuracy']:.3f}\n"
                f"Train Loss: {row_data['train_loss']:.4f}\n"
                f"Val Loss: {row_data['val_loss']:.4f}"
            )
        else:
            info = f"Эксперимент {trial_num}\n(данные не найдены)"

        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(img_path.name, fontsize=10)
        ax.text(0.5, -0.2, info, ha="center", va="top", fontsize=9, transform=ax.transAxes)

    plt.tight_layout()
    plt.show()


def plot_grouped_by_param(param_name: str, bins: int = 3):
    # Пути
    paths = get_project_paths()
    image_dir = paths["image_dir"]
    models_dir = paths["models_dir"]
    metrics_df = pd.read_csv(paths['raw_dir'] / "runs_aug.csv")

    # Загружаем параметры всех экспериментов
    param_files = list(models_dir.glob("optuna_aug_trial_*_params.json"))
    experiment_data = []

    for param_file in param_files:
        try:
            trial_num = int(param_file.name.split("_")[3])
            with open(param_file) as f:
                aug_params = json.load(f)
            aug_params["trial_num"] = trial_num
            experiment_data.append(aug_params)
        except Exception:
            continue

    # Группировка по указанному параметру
    values = [exp[param_name] for exp in experiment_data]
    if isinstance(values[0], bool):
        groups = defaultdict(list)
        for exp in experiment_data:
            groups[str(exp[param_name])].append(exp)
    else:
        # binarизуем значения
        min_val, max_val = min(values), max(values)
        step = (max_val - min_val) / bins
        groups = defaultdict(list)
        for exp in experiment_data:
            val = exp[param_name]
            bin_index = int((val - min_val) / step)
            bin_index = min(bin_index, bins - 1)
            key = f"{min_val + bin_index * step:.2f}–{min_val + (bin_index + 1) * step:.2f}"
            groups[key].append(exp)

    # Рисуем
    for group_key, group_experiments in groups.items():
        n = len(group_experiments)
        rows = (n + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
        axes = axes.flatten() if n > 1 else [axes]

        for ax in axes[n:]:
            ax.axis("off")

        for ax, exp in zip(axes, group_experiments):
            trial_num = exp["trial_num"]
            img_path = image_dir / f"optuna_confmat_{trial_num}_confusion_matrix.png"
            if not img_path.exists():
                continue

            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis("off")

            # Получаем метрики
            row = metrics_df[metrics_df["model_name_tag"] == f"optuna_aug_trial_{trial_num}"]
            if not row.empty:
                row_data = row.iloc[0]
                info = (
                    f"Trial {trial_num}\n"
                    f"Accuracy: {row_data['accuracy']:.3f}\n"
                    f"Train Loss: {row_data['train_loss']:.4f}\n"
                    f"Val Loss: {row_data['val_loss']:.4f}\n"
                    f"{param_name}: {exp[param_name]}"
                )
            else:
                info = f"Trial {trial_num}\n(данные не найдены)"

            ax.set_title(f"Trial {trial_num}", fontsize=10)
            ax.text(0.5, -0.2, info, ha="center", va="top", fontsize=9, transform=ax.transAxes)

        fig.suptitle(f"Группировка по параметру '{param_name}' — значение {group_key}", fontsize=16)
        plt.tight_layout()
        plt.show()
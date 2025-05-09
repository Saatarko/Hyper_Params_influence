import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from optuna.samplers import GridSampler
import dill
import mlflow
import optuna
from functools import partial
import torch.nn.init as init
import numpy as np
import pandas as pd
import yaml
from torch import nn, optim

from auxiliary_functions import get_project_paths, plot_losses
from train_nn import SimpleNN, nn_train, nn_train_class
from task_registry import task, main
mlflow.set_tracking_uri('http://localhost:5000')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
with open(PROJECT_ROOT / "params.yaml") as f:
    paths = get_project_paths()

# Настройка отдельного логгера для неудачных трaйлов
log_path = paths["logs_dir"] / "optuna_failures.log"

error_logger = logging.getLogger("optuna_failures")
error_logger.setLevel(logging.WARNING)
handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
error_logger.addHandler(handler)



@task("data:generate_synthetic_data")
def generate_synthetic_data():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()
        experiment_synthetic = yaml.safe_load(f)["experiment_synthetic"]

    n_samples = experiment_synthetic['n_samples']
    seed = experiment_synthetic['seed']
    noise_std = experiment_synthetic['noise_std']


    np.random.seed(seed)
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.uniform(1, 2, n_samples)  # чтобы лог был корректный
    x4 = np.random.normal(0, 1, n_samples)
    x5 = np.random.normal(0, 1, n_samples)
    x6 = np.random.normal(0, 1, n_samples)

    x7 = x1 + x2
    x8 = np.log(x3)
    x9 = x7 * np.sin(x6)
    x10 = 20 * x1 + x2 / x3

    X = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], axis=1)

    # Целевая переменная: функция + шум
    y = np.sin(X[:, 0]) + np.sin(X[:, 1]) + np.random.normal(0, noise_std, n_samples)

    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(X.shape[1])])
    df["y"] = y

    df.to_csv(paths["raw_dir"]/'synthetic_regression.csv')

    return df



def init_weights(model, initializer):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if initializer == "xavier":
                init.xavier_uniform_(m.weight)
            elif initializer == "kaiming":
                init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif initializer == "normal":
                init.normal_(m.weight, mean=0.0, std=0.02)
            elif initializer == "zeros":
                init.constant_(m.weight, 0)
            elif initializer == "he":
                init.kaiming_normal_(m.weight, nonlinearity="relu")


def get_loss_function(name, l1=0.0, l2=0.0, model=None):
    if name == "mse":
        base_loss = nn.MSELoss()
    elif name == "BCE":
        base_loss = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss: {name}")

    def loss_fn(output, target):
        loss = base_loss(output, target)
        if model is not None:
            l1_loss = sum(p.abs().sum() for p in model.parameters())
            l2_loss = sum(p.pow(2).sum() for p in model.parameters())
            loss += l1 * l1_loss + l2 * l2_loss
        return loss

    return loss_fn


def objective(trial, X, y, input_dim):
    try:
        activation = trial.suggest_categorical("activation",
                                               ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "swish", "mish"])
        dropout = trial.suggest_categorical("dropout", [0.0, 0.2])
        l1 = trial.suggest_categorical("l1", [0.0, 0.01])
        l2 = trial.suggest_categorical("l2", [0.0, 0.01])
        initializer = trial.suggest_categorical("initializer", ["xavier", "kaiming", "normal", "zeros", "he"])
        loss_name = trial.suggest_categorical("loss_fn", ["mse"])
        hidden_size = trial.suggest_categorical("hidden_size", [32])
        lr = 0.01
        num_epochs = 200
        batch_size = trial.suggest_categorical("batch_size", [64])
        seed = 42

        model = SimpleNN(input_dim=input_dim, hidden_size=hidden_size, activation=activation, dropout=dropout)
        init_weights(model, initializer)

        criterion = get_loss_function(loss_name, l1, l2, model)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model_name_tag = f"{activation}"
        if l1 > 0: model_name_tag += "_l1"
        if l2 > 0: model_name_tag += "_l2"
        if dropout > 0: model_name_tag += "_drop"
        model_name_tag += f"_{initializer}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Выбираем устройство

        best_model_state, X_val_tensor, y_val_tensor = nn_train(
            X=X,
            y=y,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            model_name_tag=model_name_tag,
            num_epochs=num_epochs,
            batch_size=batch_size,
            seed=seed,
            device=device  # Передаем устройство
        )

        model.load_state_dict(best_model_state)
        model.eval()

        with torch.no_grad():
            # Прогнозы и вычисление потерь
            val_preds = model(X_val_tensor).squeeze()
            val_loss = nn.MSELoss()(val_preds, y_val_tensor.squeeze()).item()

            # Логирование модели в MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric(f"val_loss", val_loss)
                mlflow.pytorch.log_model(model, "model")  # Можно добавить input_example, если нужно

        return val_loss

    except Exception as e:
        # Логируем ошибку в отдельный лог-файл
        error_logger.warning(f"Trial failed with params: {trial.params} -> {e}")
        return float("inf")


@task("data:run_optuna_search_synt")
def run_optuna_search_synt():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()

    synthetic_regression = pd.read_csv(paths["raw_dir"] / "synthetic_regression.csv")
    y = synthetic_regression['y']
    X = synthetic_regression.drop(columns=['y','Unnamed: 0'])
    input_dim = X.shape[1]

    search_space = {
        "activation": ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "swish", "mish"],
        "initializer": ["xavier", "kaiming", "normal", "zeros", "he"],
        "dropout": [0.0, 0.2],
        "loss_fn": ["mse"],
        "l1": [0.0, 0.01],
        "l2": [0.0, 0.01],
        "hidden_size": [32]
    }

    sampler = GridSampler(search_space)
    study = optuna.create_study(sampler=sampler, direction="minimize")

    with mlflow.start_run(run_name=f"GridSearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("search_type", "GridSampler")

        study.optimize(lambda trial: objective(trial, X, y, input_dim), n_trials=len(sampler._all_grids))

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_loss", study.best_value)

        result_path = paths["models_dir"] / "optuna_best_result.json"
        with open(result_path, "w") as f:
            json.dump({
                "best_params": study.best_params,
                "best_val_loss": study.best_value
            }, f, indent=4)
        mlflow.log_artifact(result_path)

        optuna_path = paths["models_dir"] / "optuna_study.pkl"
        with open(optuna_path, "wb") as f:
            dill.dump(study, f)
        mlflow.log_artifact(optuna_path)

    return study



def objective_class(trial, X, y, input_dim):
    try:
        activation = trial.suggest_categorical("activation",
                                               ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "swish", "mish"])
        dropout = trial.suggest_categorical("dropout", [0.0, 0.2])
        l1 = trial.suggest_categorical("l1", [0.0, 0.01])
        l2 = trial.suggest_categorical("l2", [0.0, 0.01])
        initializer = trial.suggest_categorical("initializer", ["xavier", "kaiming", "normal", "zeros", "he"])
        loss_name = trial.suggest_categorical("loss_fn", ["BCE"])
        hidden_size = trial.suggest_categorical("hidden_size", [32])
        lr = 0.01
        num_epochs = 200
        batch_size = trial.suggest_categorical("batch_size", [64])
        seed = 42

        model = SimpleNN(input_dim=input_dim, hidden_size=hidden_size, activation=activation, dropout=dropout)
        init_weights(model, initializer)

        criterion = get_loss_function(loss_name, l1, l2, model)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model_name_tag = f"Class_{activation}"
        if l1 > 0: model_name_tag += "_l1"
        if l2 > 0: model_name_tag += "_l2"
        if dropout > 0: model_name_tag += "_drop"
        model_name_tag += f"_{initializer}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Выбираем устройство

        best_model_state, X_val_tensor, y_val_tensor = nn_train_class(
            X=X,
            y=y,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            model_name_tag=model_name_tag,
            num_epochs=num_epochs,
            batch_size=batch_size,
            seed=seed,
            device=device  # Передаем устройство
        )

        model.load_state_dict(best_model_state)
        model.eval()

        with torch.no_grad():
            # Прогнозы и вычисление потерь
            val_preds = model(X_val_tensor).squeeze()
            val_loss = torch.nn.BCEWithLogitsLoss()(val_preds, y_val_tensor.squeeze()).item()
            preds_binary = (torch.sigmoid(val_preds) > 0.5).int()
            y_true = y_val_tensor.int()
            acc = (preds_binary == y_true).float().mean().item()
            mlflow.log_metric(f"val_accuracy_{model_name_tag}", acc)

            # Логирование модели в MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric(f"val_bce_loss", val_loss)
                mlflow.log_metric(f"val_accuracy", acc)
                mlflow.pytorch.log_model(model, f"class_model")

        return val_loss

    except Exception as e:
        # Логируем ошибку в отдельный лог-файл
        error_logger.warning(f"Trial failed with params: {trial.params} -> {e}")
        return float("inf")



@task("data:run_optuna_search_synt_class")
def run_optuna_search_synt_class():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()

    synthetic_class = pd.read_csv(paths["raw_dir"] / "synthetic_dataset.csv")
    y = synthetic_class['target']
    X = synthetic_class.drop(columns=['target'])
    input_dim = X.shape[1]

    search_space = {
        "activation": ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "swish", "mish"],
        "initializer": ["xavier", "kaiming", "normal", "zeros", "he"],
        "dropout": [0.0, 0.2],
        "loss_fn": ["BCE"],
        "l1": [0.0, 0.01],
        "l2": [0.0, 0.01],
        "hidden_size": [32]
    }

    sampler = GridSampler(search_space)
    study = optuna.create_study(sampler=sampler, direction="minimize")

    with mlflow.start_run(run_name=f"Class_GridSearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("search_type", "GridSampler")

        study.optimize(lambda trial: objective_class(trial, X, y, input_dim), n_trials=len(sampler._all_grids))

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_loss", study.best_value)

        result_path = paths["models_dir"] / "class_optuna_best_result.json"
        with open(result_path, "w") as f:
            json.dump({
                "best_params": study.best_params,
                "best_val_loss": study.best_value
            }, f, indent=4)
        mlflow.log_artifact(result_path)

        optuna_path = paths["models_dir"] / "class_optuna_study.pkl"
        with open(optuna_path, "wb") as f:
            dill.dump(study, f)
        mlflow.log_artifact(optuna_path)

    return study





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()

    if args.tasks:
        main(args.tasks)  # Здесь передаем задачи, которые указаны в командной строке

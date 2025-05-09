import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim, nn
from torch.utils.data import DataLoader, random_split, Dataset
from auxiliary_functions import get_project_paths, plot_losses
from task_registry import task, main
import torch.nn.functional as F
mlflow.set_tracking_uri('http://localhost:5000')

class NNDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            # Возвращаем как тензоры
            return self.X[idx].float(), self.y[idx].float()
        return self.X[idx].float()


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))  # softplus(x) = ln(1 + e^x)


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_size, activation='relu', dropout=None):
        super(SimpleNN, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_size)

        # Активации
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "swish":
            self.activation = Swish()
        elif activation == "mish":
            self.activation = Mish()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.use_dropout = dropout is not None
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.output(x)
        return x


def nn_train(X, y, model, criterion, optimizer, model_name_tag, num_epochs=100, batch_size=32, seed=42, device=None):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()

    # Разделение на train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=seed)

    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, paths["processed_dir"]/'scaler.pkl')

    # Сохранение вектора
    np.save( paths["vectors_dir"]/'X_train_scaled.npy', X_train_scaled)
    np.save( paths["vectors_dir"]/'X_val_scaled.npy', X_val_scaled)
    np.save( paths["vectors_dir"]/'X_test_scaled.npy', X_test_scaled)

    # Тензоры для модели
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

    # Датасеты и загрузчики
    train_dataset = NNDataset(X_train_tensor, y_train_tensor)
    val_dataset = NNDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Обучение
    best_val_loss = float("inf")
    best_model_state = None

    epoch_train_losses = []
    epoch_val_losses = []

    model = model.to(device)  # Переносим модель на устройство

    model_path = paths["models_dir"] /f"{model_name_tag}_final_model.pt"
    metrics_path = paths["models_dir"] /f"{model_name_tag}_train_metrics.json"

    with mlflow.start_run(nested=True):
        mlflow.log_param("model_name_tag", model_name_tag)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("seed", seed)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Переносим данные на устройство
                output = model(X_batch)
                loss = criterion(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Валидация
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Переносим данные на устройство
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)

            epoch_train_losses.append(avg_train)
            epoch_val_losses.append(avg_val)

            mlflow.log_metric("{train_loss", avg_train, step=epoch)
            mlflow.log_metric("{val_loss", avg_val, step=epoch)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_state = model.state_dict()

            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)
        mlflow.pytorch.log_model(model, model_name_tag)

        plot_path = plot_losses(epoch_train_losses, epoch_val_losses, model_name_tag)
        mlflow.log_artifact(str(plot_path))

        train_metrics = {
            "best_val_loss": best_val_loss,
            "final_train_loss": epoch_train_losses[-1],
            "final_val_loss": epoch_val_losses[-1],
            "num_epochs": num_epochs
        }

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=4)

    return best_model_state, X_val_tensor, y_val_tensor



def nn_train_class(X, y, model, criterion, optimizer, model_name_tag, num_epochs=100, batch_size=32, seed=42, device=None):


    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()

    # Разделение на train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=seed)

    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, paths["processed_dir"] / 'scaler.pkl')

    # Сохранение вектора
    np.save(paths["vectors_dir"] / 'X_train_scaled.npy', X_train_scaled)
    np.save(paths["vectors_dir"] / 'X_val_scaled.npy', X_val_scaled)
    np.save(paths["vectors_dir"] / 'X_test_scaled.npy', X_test_scaled)

    # Тензоры для модели
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

    # Датасеты и загрузчики
    train_dataset = NNDataset(X_train_tensor, y_train_tensor)
    val_dataset = NNDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Обучение
    best_val_loss = float("inf")
    best_model_state = None

    epoch_train_losses = []
    epoch_val_losses = []

    model = model.to(device)

    model_path = paths["models_dir"] / f"{model_name_tag}_final_model.pt"
    metrics_path = paths["models_dir"] / f"{model_name_tag}_train_metrics.json"

    with mlflow.start_run(nested=True):
        mlflow.log_param("model_name_tag", model_name_tag)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("seed", seed)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Валидация
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)

            epoch_train_losses.append(avg_train)
            epoch_val_losses.append(avg_val)

            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss", avg_val, step=epoch)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_state = model.state_dict()

            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)
        mlflow.pytorch.log_model(model, model_name_tag)

        plot_path = plot_losses(epoch_train_losses, epoch_val_losses, model_name_tag)
        mlflow.log_artifact(str(plot_path))

        train_metrics = {
            "best_val_loss": best_val_loss,
            "final_train_loss": epoch_train_losses[-1],
            "final_val_loss": epoch_val_losses[-1],
            "num_epochs": num_epochs
        }

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=4)

    return best_model_state, X_val_tensor, y_val_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()

    if args.tasks:
        main(args.tasks)  # Здесь передаем задачи, которые указаны в командной строке

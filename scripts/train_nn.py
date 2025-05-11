import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim, nn
from torch.utils.data import DataLoader, random_split, Dataset
from auxiliary_functions import get_project_paths, plot_losses, log_confusion_matrix
from task_registry import task, main
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder



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



class SimpleMLP(nn.Module):
    def __init__(self, input_size=64*64*3, hidden_size=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            Mish(),
            nn.BatchNorm1d(hidden_size),  # Batch normalization
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # Без Sigmoid
        )

    def forward(self, x):
        return self.net(x)


def train_image_model(
    model,
    full_dataset,
    criterion,
    optimizer,
    device,
    num_epochs=200,
    batch_size=32,
    val_split=0.2,
    model_name_tag="image_model",
    early_stopping_patience=30,
):
    mlflow.log_param("model_name_tag", model_name_tag)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("val_split", val_split)
    mlflow.log_param("early_stopping_patience", early_stopping_patience)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()

    model_path = paths["models_dir"] / f"{model_name_tag}_final_model.pt"

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()


        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Save the best model
    if model_path and best_model_state:
        torch.save(best_model_state, model_path)
        mlflow.log_artifact(str(model_path))

    return train_losses, val_losses

def predict_image(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).int().cpu().numpy()  # Преобразование выхода модели в метки (0 или 1)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)

def get_data_loaders(data_dir, batch_size=32, augment=False):
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # стандартные значения для ImageNet
        std=[0.229, 0.224, 0.225]
    )

    if augment:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalization,
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            normalization
        ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        normalization
    ])

    train_ds = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_ds = ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_ds.classes




class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)




def ewc_loss(model, fisher, prev_params, lambda_ewc=1000):
    loss = 0.0
    for name, param in model.named_parameters():
        if name in fisher:
            loss += (fisher[name] * (param - prev_params[name]).pow(2)).sum()
    return loss * lambda_ewc


def train_image_model_ewc(
    model,
    full_dataset,
    criterion,
    optimizer,
    device,
    fisher=None,
    prev_params=None,
    lambda_ewc=1000,
    num_epochs=200,
    batch_size=32,
    val_split=0.2,
    model_name_tag="image_model",
    early_stopping_patience=30,
):
    mlflow.log_param("model_name_tag", model_name_tag)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("val_split", val_split)
    mlflow.log_param("early_stopping_patience", early_stopping_patience)

    paths = get_project_paths()
    model_path = paths["models_dir"] / f"{model_name_tag}_final_model.pt"

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if fisher and prev_params:
                loss += ewc_loss(model, fisher, prev_params, lambda_ewc)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if model_path and best_model_state:
        torch.save(best_model_state, model_path)
        mlflow.log_artifact(str(model_path))

    return train_losses, val_losses


def define_augmentations(trial):
    rotation = trial.suggest_int("rotation", 0, 10)
    translate = trial.suggest_float("translate", 0.0, 0.2)
    scale_min = trial.suggest_float("scale_min", 0.8, 1.0)
    scale_max = trial.suggest_float("scale_max", 1.0, 1.2)
    use_color_jitter = trial.suggest_categorical("color_jitter", [True, False])
    erasing_p = trial.suggest_float("random_erasing_p", 0.0, 0.4)

    aug_params = {
        "rotation": rotation,
        "translate": translate,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "color_jitter": use_color_jitter,
        "random_erasing_p": erasing_p
    }

    transform_list = [
        transforms.RandomResizedCrop(64, scale=(scale_min, scale_max)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation),
        transforms.RandomAffine(degrees=0, translate=(translate, translate), scale=(scale_min, scale_max)),
    ]

    if use_color_jitter:
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if erasing_p > 0.0:
        transform_list.append(transforms.RandomErasing(p=erasing_p, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0))

    return transforms.Compose(transform_list), aug_params


def objective_cnn(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = get_project_paths()
    data_dir = paths["raw_dir"] / "horsehuman"
    batch_size = 32
    num_epochs = 50

    with mlflow.start_run(nested=True):
        tag = f"optuna_aug_trial_{trial.number}"
        aug_transform, aug_params = define_augmentations(trial)

        # Логируем параметры в MLflow
        mlflow.log_params(aug_params)

        # Сохраняем параметры в JSON
        params_save_path = paths["models_dir"] / f"optuna_aug_trial_{trial.number}_params.json"
        with open(params_save_path, "w") as f:
            json.dump(aug_params, f, indent=4)

        train_loader, test_loader, _ = get_data_loaders(data_dir, batch_size=batch_size, augment=False)
        train_loader.dataset.transform = aug_transform  # вручную заменяем transform

        model = SimpleCNN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        train_losses, val_losses = train_image_model(
            model,
            train_loader.dataset,
            criterion,
            optimizer,
            device,
            num_epochs=num_epochs,
            model_name_tag=tag
        )

        all_preds, all_labels = predict_image(model, test_loader, device)
        acc = (all_preds.flatten() == all_labels).mean()
        mlflow.log_metric("accuracy", acc)

        log_confusion_matrix(all_preds, all_labels, f"optuna_confmat_{trial.number}")
        plot_losses(train_losses, val_losses, f"optuna_loss_{trial.number}")

        return acc  # целевая метрика


# Модель нейронной сети
class RegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size, activation_fn, dropout):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation_fn  # swish или другие активации

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def nn_train_best_gp(X, y, model_name_tag, num_epochs=100, batch_size=64, seed=42, device=None):
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

    np.save(paths["vectors_dir"]/'X_train_scaled.npy', X_train_scaled)
    np.save(paths["vectors_dir"]/'X_val_scaled.npy', X_val_scaled)
    np.save(paths["vectors_dir"]/'X_test_scaled.npy', X_test_scaled)

    # Тензоры
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)


    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    # Датасеты и загрузчики
    train_loader = DataLoader(NNDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NNDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    # Гиперпараметры и модель
    activation_fn = nn.SiLU()
    dropout = 0.0
    loss_fn = nn.MSELoss()
    hidden_size = 32

    input_size = X_train.shape[1]
    model = RegressionNN(input_size, hidden_size, activation_fn, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_model_state = None
    epoch_train_losses = []
    epoch_val_losses = []

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
                loss = loss_fn(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = model(X_batch)
                    loss = loss_fn(output, y_batch)
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

        # Сохранение модели и метрик
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)
        mlflow.pytorch.log_model(model, model_name_tag)

        train_metrics = {
            "best_val_loss": best_val_loss,
            "final_train_loss": epoch_train_losses[-1],
            "final_val_loss": epoch_val_losses[-1],
            "num_epochs": num_epochs
        }
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=4)

        # Оценка на тестовой выборке
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_tensor).squeeze()
            test_loss = nn.MSELoss()(test_preds, y_test_tensor.squeeze()).item()
            mlflow.log_metric("test_loss", test_loss)

            # График: True vs Predicted
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test_tensor.cpu(), test_preds.cpu(), alpha=0.6)
            plt.plot([y_test_tensor.min(), y_test_tensor.max()],
                     [y_test_tensor.min(), y_test_tensor.max()],
                     color='red', linestyle='--')
            plt.xlabel("True Values")
            plt.ylabel("Predicted Values")
            plt.title("True vs Predicted on Test Set")
            plot_path = paths["reports_dir"] / f"{model_name_tag}_test_predictions.png"
            os.makedirs(plot_path.parent, exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(str(plot_path))

    return best_model_state, best_val_loss, test_loss


class ClassificationNN(nn.Module):
    def __init__(self, input_size, hidden_size, activation_fn, dropout=0.0, init_type='xavier'):
        super(ClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.output_activation = nn.Sigmoid()  # Для бинарной классификации

        self._initialize_weights(init_type)

    def _initialize_weights(self, init_type):
        if init_type == 'xavier':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        elif init_type == 'he':
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
            nn.init.zeros_(self.fc1.bias)
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
            nn.init.zeros_(self.fc2.bias)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.output_activation(x)
        return x

def nn_train_classification_best_gp(X, y, model_name_tag, num_epochs=100, batch_size=64, seed=42, device=None):
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

    # Сохранение векторов
    np.save(paths["vectors_dir"] / 'X_train_scaled.npy', X_train_scaled)
    np.save(paths["vectors_dir"] / 'X_val_scaled.npy', X_val_scaled)
    np.save(paths["vectors_dir"] / 'X_test_scaled.npy', X_test_scaled)

    # Тензоры
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

    # Датасеты и загрузчики
    train_loader = DataLoader(NNDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NNDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    # Модель
    input_size = X_train.shape[1]
    model = ClassificationNN(
        input_size=input_size,
        hidden_size=32,
        activation_fn=nn.Tanh(),
        dropout=0.2,
        init_type="xavier"
    ).to(device)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epoch_train_losses = []
    epoch_val_losses = []
    best_val_loss = float("inf")
    best_model_state = None

    model_path = paths["models_dir"] / f"{model_name_tag}_final_model.pt"
    metrics_path = paths["models_dir"] / f"{model_name_tag}_train_metrics.json"

    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "model_name_tag": model_name_tag,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "seed": seed,
            "activation": "tanh",
            "dropout": 0.2,
            "initializer": "xavier",
            "loss_fn": "BCE",
            "hidden_size": 32
        })

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Валидация
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = model(X_batch)
                    loss = loss_fn(output, y_batch)
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

        # Сохраняем модель
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)
        mlflow.pytorch.log_model(model, model_name_tag)

        # Метрики
        train_metrics = {
            "best_val_loss": best_val_loss,
            "final_train_loss": epoch_train_losses[-1],
            "final_val_loss": epoch_val_losses[-1],
            "num_epochs": num_epochs
        }
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=4)

        # Графики
        loss_plot_path = plot_losses(epoch_train_losses, epoch_val_losses, f"{model_name_tag}_loss")
        mlflow.log_artifact(loss_plot_path)

        # Матрица ошибок
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor).squeeze().cpu().numpy()
            val_preds_class = (val_preds >= 0.5).astype(int)
            val_true = y_val_tensor.squeeze().cpu().numpy()
            confmat_path = log_confusion_matrix(val_preds_class, val_true, f"{model_name_tag}_confmat")
            mlflow.log_artifact(confmat_path)

    return best_model_state, best_val_loss, val_preds_class





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()

    if args.tasks:
        main(args.tasks)  # Здесь передаем задачи, которые указаны в командной строке

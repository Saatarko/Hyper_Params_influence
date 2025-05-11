# Часть 1. Исследования функций потерь, регуляризаторов, dropout, нормализации 

## 📊 Регрессия
### 🧪 Исходные данные

Регрессия (ручная генерация):

x1 = np.random.normal(0, 1, n_samples)
x2 = np.random.normal(0, 1, n_samples)
x3 = np.random.uniform(1, 2, n_samples)
x4 = np.random.normal(0, 1, n_samples)
x5 = np.random.normal(0, 1, n_samples)
x6 = np.random.normal(0, 1, n_samples)

x7 = x1 + x2
x8 = np.log(x3)
x9 = x7 * np.sin(x6)
x10 = 20 * x1 + x2 / x3

X = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], axis=1)
y = np.sin(X[:, 0]) + np.sin(X[:, 1]) + np.random.normal(0, noise_std, n_samples)

### 🔄 Выводы:


## Классификация (make_classification):

### 🧪 Исходные данные

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    random_state=42
)

### 🔄 Выводы:


# Часть 2. Применении аугментации и трансфер обученных моделей

🧠 1. Базовый классификатор (без аугментации)

- Загрузка и разбиение датасета (напр. Cats vs Dogs)
- Простейшая `NN` на PyTorch
- Обучение без аугментации
- Логирование метрик (accuracy, confusion matrix) в MLflow
- Сохранение модели через DVC

### 🔄 Выводы:

🔁 2. Обучение с аугментацией

- Применение `transforms` (RandomFlip, ColorJitter, RandomCrop)
- Повторное обучение той же архитектуры
- Сравнение метрик
- Визуализация `confusion_matrix` до/после
- Логирование в MLflow
- 
### 🔄 Выводы:

🔄 3. Transfer Learning

- Загрузка предобученной модели
- Замена классификатора 
- Fine-tune модель на Cats vs Dogs(small)
- Применение модели:
- Логирование метрик и выводов

### 🔄 Выводы:

# Часть 3. Исследование аугментации

- Загрузка предобученной модели
- Замена классификатора 
- Настройка optuna для исследования аугментации (на трансферной модели и сете лошади/люди)
- Применение модели:
- Логирование метрик и выводов
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from pathlib import Path
from src.data_pipeline import PRISMDataset
# Вам нужно создать или убедиться, что эти файлы существуют:
from src.models import STGCN_PRISM 
from src.privacy_module import PrivacyPreservingTrainer, create_privacy_config

# --- КОНФИГУРАЦИЯ ОБУЧЕНИЯ ---
DATA_ROOT = "data"
MODEL_OUTPUT_DIR = "models/stgcn"
RESULTS_OUTPUT_DIR = "results/benchmarks"

# Параметры UCF-101 и Модели
SPLIT_ID = 1          # Используем официальный Сплит 1
NUM_CLASSES = 101     # Количество классов в UCF-101
SEQUENCE_LENGTH = 30  # Длина последовательности кадров
USE_KINEMATICS = True # Использовать кинематические признаки
IN_CHANNELS = 100     # 100 фич: 50 статика + 50 скорость

# Параметры DP (Дифференциальная Приватность)
EPSILON = 1.0         # Бюджет приватности (чем меньше, тем выше приватность)
DELTA = 1e-5          # Вероятность отказа (должно быть меньше 1/N)
MAX_GRAD_NORM = 1.0   # Клиппирование градиентов (важно для DP)
# Параметры Обучения
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
# --------------------

def run_training_pipeline():
    # 1. Загрузка Данных
    print(f"--- 1. Loading UCF-101 Split {SPLIT_ID} Data ---")
    try:
        train_dataset = PRISMDataset(
            data_root=DATA_ROOT, split_id=SPLIT_ID, subset='train',
            sequence_length=SEQUENCE_LENGTH, use_kinematics=USE_KINEMATICS
        )
        test_dataset = PRISMDataset(
            data_root=DATA_ROOT, split_id=SPLIT_ID, subset='test',
            sequence_length=SEQUENCE_LENGTH, use_kinematics=USE_KINEMATICS
        )
    except Exception as e:
        print(f"FATAL: Error loading dataset. Ensure feature files exist and the data_pipeline is correct. Error: {e}")
        return

    # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
    # Мы отключаем 'num_workers', чтобы исправить ошибку 'resize storage'.
    # Это заставит DataLoader работать в 1 поток, но зато он не "упадет".
    
    # DataLoader требует, чтобы train_loader имел drop_last=True для DP-SGD.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True, 
        num_workers=0  # <-- ИСПРАВЛЕНО (было 4)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0  # <-- ИСПРАВЛЕНО (было 4)
    )
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---


    # 2. Инициализация Модели и Конфигурации Приватности
    print("\n--- 2. Initializing Model and Privacy ---")
    
    model = STGCN_PRISM(num_joints=1, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    
    # (Этот код мы исправили в прошлый раз, он правильный)
    privacy_config = create_privacy_config(
        epsilon=EPSILON, 
        delta=DELTA, 
        max_grad_norm=MAX_GRAD_NORM,
        learning_rate=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE
    )

    # 3. Запуск Приватного Обучения
    print(f"\n--- 3. Training with DP (ε={EPSILON}) ---")
    
    trainer = PrivacyPreservingTrainer(
        model, 
        privacy_config
    )
    
    # Обучение
    training_results = trainer.train(train_loader, test_loader, num_epochs=NUM_EPOCHS)
    
    # 4. Сохранение и Оценка
    print("\n--- 4. Saving Results ---")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    model_path = Path(MODEL_OUTPUT_DIR) / f"prism_ucf101_split{SPLIT_ID}_e{str(EPSILON).replace('.', 'p')}.pth"
    torch.save(model.state_dict(), str(model_path))
    print(f"Model saved to: {model_path}")
    
    # (Добавляем проверку, что 'test_accuracies' не пустой)
    final_accuracy = training_results['val_accuracies'][-1] if training_results.get('val_accuracies') else 0.0
    print(f"\nTraining Finished. Final Test Accuracy (Split {SPLIT_ID}): {final_accuracy:.2f}%")
    
if __name__ == "__main__":
    run_training_pipeline()


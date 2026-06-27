# ============================================================
# app.py — Веб-интерфейс для распознавания птиц
# ============================================================

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- 1. Настройки устройства ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Используется устройство: {device}")

# --- 2. Путь к модели (локально в Space) ---
MODEL_PATH = "bird_model_russia.pth"
NUM_CLASSES = 20

# --- 3. Загрузка модели ---
print("🖥️ Загрузка модели...")
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Проверяем, есть ли файл модели локально
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ Модель загружена из локального файла")
else:
    print("❌ Файл модели не найден! Загрузите bird_model_russia.pth в ту же папку")

model = model.to(device)
model.eval()

# --- 4. Русские названия классов ---
class_names = [
    'Кряква',          # 0: Anas_platyrhynchos
    'Серая цапля',     # 1: Ardea_cinerea
    'Озёрная чайка',   # 2: Chroicocephalus_ridibundus
    'Сизый голубь',    # 3: Columba_livia
    'Серая ворона',    # 4: Corvus_corone_cornix
    'Грач',            # 5: Corvus_frugilegus
    'Лебедь-шипун',    # 6: Cygnus_olor
    'Зяблик',          # 7: Fringilla_coelebs
    'Лысуха',          # 8: Fulica_atra
    'Хохотунья',       # 9: Larus_cachinnans
    'Тихоокеанская чайка', # 10: Larus_schistisagus
    'Белая трясогузка', # 11: Motacilla_alba
    'Большая синица',  # 12: Parus_major
    'Домовый воробей', # 13: Passer_domesticus
    'Полевой воробей', # 14: Passer_montanus
    'Большой баклан',  # 15: Phalacrocorax_carbo
    'Сорока',          # 16: Pica_pica
    'Обыкновенный скворец', # 17: Sturnus_vulgaris
    'Чёрный дрозд',    # 18: Turdus_merula
    'Рябинник'         # 19: Turdus_pilaris
]

# --- 5. Преобразования для фото ---
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 6. Функция предсказания ---
def predict_image(image):
    """Принимает изображение, возвращает предсказание и уверенность"""
    # Преобразуем изображение
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Топ-3 результата
    top_probs, top_indices = torch.topk(probabilities, 3)
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    # Формируем результат для Gradio
    result = {class_names[idx]: float(prob) for idx, prob in zip(top_indices, top_probs)}
    return result

# --- 7. Создание интерфейса ---
title = "🐦 BirdEye: Распознавание птиц России"
description = """
Загрузите фотографию птицы, и модель определит её вид.
Распознаёт **20 видов** птиц, обитающих на территории России.
"""

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Загрузите фото птицы"),
    outputs=gr.Label(num_top_classes=3, label="Результаты распознавания"),
    title=title,
    description=description,
)

# --- 8. Запуск ---
demo.launch()

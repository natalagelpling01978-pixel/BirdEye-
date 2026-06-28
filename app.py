import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Путь к модели (на Hugging Face она лежит в той же папке)
MODEL_PATH = "bird_model_russia.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Загрузка модели ---
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 20)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# --- Преобразования для фото ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Правильный список классов (20 видов птиц России) ---
class_names = [
    'Кряква', 'Серая цапля', 'Озёрная чайка', 'Сизый голубь', 'Серая ворона',
    'Грач', 'Лебедь-шипун', 'Зяблик', 'Лысуха', 'Хохотунья',
    'Тихоокеанская чайка', 'Белая трясогузка', 'Большая синица', 'Домовый воробей',
    'Полевой воробей', 'Большой баклан', 'Сорока', 'Обыкновенный скворец',
    'Чёрный дрозд', 'Рябинник'
]

# --- Функция предсказания ---
def predict(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top_probs, top_indices = torch.topk(probs, 3)
    result = {class_names[idx]: float(top_probs[i]) for i, idx in enumerate(top_indices)}
    return result

# --- Интерфейс Gradio ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите фото птицы"),
    outputs=gr.Label(num_top_classes=3, label="Результаты распознавания"),
    title="🐦 BirdEye — Распознавание птиц России",
    description="Загрузите фотографию птицы, и модель определит её вид. Распознаёт 20 видов птиц России."
)

demo.launch()

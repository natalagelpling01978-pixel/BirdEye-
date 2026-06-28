from flask import Flask, request, render_template_string
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)

# Путь к модели (должен быть правильным)
MODEL_PATH = '/home/Natalia8/mysite/bird_model_russia.pth'

# Загрузка модели
device = torch.device("cpu")
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 20)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Преобразования
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Список классов (правильный порядок из обучения)
class_names = [
    'Кряква', 'Серая цапля', 'Озёрная чайка', 'Сизый голубь', 'Серая ворона',
    'Грач', 'Лебедь-шипун', 'Зяблик', 'Лысуха', 'Хохотунья',
    'Тихоокеанская чайка', 'Белая трясогузка', 'Большая синица', 'Домовый воробей',
    'Полевой воробей', 'Большой баклан', 'Сорока', 'Обыкновенный скворец',
    'Чёрный дрозд', 'Рябинник'
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BirdEye</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: auto; padding: 20px; text-align: center; }
        .result { margin-top: 20px; padding: 15px; background: #e8f5e9; border-radius: 10px; }
    </style>
</head>
<body>
    <h1>🐦 BirdEye</h1>
    <p>Загрузите фото птицы</p>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <button type="submit">Определить</button>
    </form>
    {% if result %}
    <div class="result">
        <h2>Результат: {{ result }}</h2>
        <p>Уверенность: {{ confidence }}%</p>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename:
                img = Image.open(file.stream).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    pred_idx = torch.argmax(probs).item()
                    confidence = round(probs[pred_idx].item() * 100, 2)
                    result = class_names[pred_idx]
    return render_template_string(HTML_TEMPLATE, result=result, confidence=confidence)

# Это условие нужно для локального запуска. На Render'е его не выполнят, так как используют Gunicorn[citation:7].
if __name__ == '__main__':
    app.run()

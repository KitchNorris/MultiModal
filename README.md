# Multimodal Calorie Prediction

**Схема мультимодальной модели:**  
текст + изображение + масса → регрессор → калории

---

## Описание проекта

Модель предсказывает **калорийность блюд** на основе:  
- Фото блюда  
- Списка ингредиентов  
- Массы блюда  

Используется мультимодальная архитектура, объединяющая визуальные и текстовые эмбеддинги.

---

## Архитектура

- **Текстовая ветка:** BERT (`bert-base-uncased`) → линейная проекция  
- **Визуальная ветка:** ResNet50 (`pretrained=True`) → линейная проекция  
- **Масса блюда:** стандартизируется и проецируется  
- **Фьюжн:** объединение всех эмбеддингов → MLP регрессор → скаляр калорийности  

---

## Быстрый старт

```bash
git clone <repo_url>
cd <repo_name>
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

## Данные

dish.csv — метки калорий, масса, ингредиенты
ingredients.csv — справочник ингредиентов
images/ — папка с изображениями <dish_id>/rgb.png

---

## Тренировка

```
from train import train
train_losses, val_maes = train()
```

Разморозка последних слоев BERT и ResNet
Optimizer: AdamW с разными LR для текстовой, визуальной и регрессионной веток
Loss: L1Loss (MAE)

---

## Инференс

```
model = MultimodalModel(Config)
model.load_state_dict(torch.load(Config.SAVE_PATH))
model.eval()

output = model(input_ids=input_ids,
               attention_mask=attention_mask,
               image=image_tensor,
               mass=mass_tensor)
print(f"Predicted calories: {output.item()}")
```

---

## Аугментации изображений

RandomCrop, HorizontalFlip, VerticalFlip, Rotate
Affine, BrightnessContrast, GaussianBlur, MotionBlur, GaussNoise
Normalize, ToTensorV2

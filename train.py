import re
from functools import partial
import timm
import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
from transformers import AutoModel, AutoTokenizer
# from clearml import Task, Logger
from tqdm import tqdm

from dataset import MultimodalCalorieDataset, collate_fn, get_transforms


class Config:
    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "resnet50"
    
    # Какие слои размораживаем - совпадают с неймингом в моделях
    TEXT_MODEL_UNFREEZE1 = "encoder.layer.11|pooler"
    TEXT_MODEL_UNFREEZE = "encoder.layer.8|encoder.layer.9|encoder.layer.10|encoder.layer.11|pooler"

    IMAGE_MODEL_UNFREEZE = "layer.3|layer.4"  
    
    # Гиперпараметры
    BATCH_SIZE = 32
    TEXT_LR1 = 3e-5
    TEXT_LR = 5e-5

    IMAGE_LR = 1e-4
    REGR_LR = 5e-4
    EPOCHS = 10
    DROPOUT = 0.15
    HIDDEN_DIM = 256
    
    # Пути
    DISH_CSV = "data/dish.csv"
    INGR_CSV = "data/ingredients.csv"
    IMGS_ROOT = "data/images"
    SAVE_PATH = "models/best_model.pth"


def set_requires_grad(module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for param, _ in module.named_parameters():
            param.requires_grad = False
        return

    pattern = re.compile(unfreeze_pattern)

    for name, param in module.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        self.mass_proj = nn.Linear(1, config.HIDDEN_DIM)

        self.regressor = nn.Sequential(
            nn.Linear(3 * config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 1)  # скаляр на выходе
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)
        mass_emb = self.mass_proj(mass.unsqueeze(1))

        fused_emb = torch.cat([text_emb, image_emb, mass_emb], dim=1)
        out = self.regressor(fused_emb)
        return out.squeeze(1)  # размер [batch]


def train():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_losses = []
    val_maes = []

    # Инициализация модели
    model = MultimodalModel(Config).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)

    # Разморозка слоёв
    set_requires_grad(model.text_model, unfreeze_pattern=Config.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_model, unfreeze_pattern=Config.IMAGE_MODEL_UNFREEZE)

    # Оптимизатор 
    optimizer = AdamW([
        {'params': model.text_model.parameters(), 'lr': Config.TEXT_LR},
        {'params': model.image_model.parameters(), 'lr': Config.IMAGE_LR},
        {'params': model.regressor.parameters(), 'lr': Config.REGR_LR}
    ])

    criterion = nn.L1Loss()
    
    # Загрузка данных
    transforms = get_transforms(Config, "train")
    val_transforms = get_transforms(Config, "val")

    train_dataset = MultimodalCalorieDataset(Config, transforms)
    val_dataset = MultimodalCalorieDataset(Config, val_transforms, split="test")

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer))

    # Инициализируем метрику
    best_mae = float('inf')
    mae_metric = torchmetrics.MeanAbsoluteError().to(DEVICE)

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            inputs = {k: batch[k].to(DEVICE) for k in ["input_ids", "attention_mask", "image", "mass"]}
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            preds = model(**inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        val_mae = validate(model, val_loader, DEVICE, mae_metric)
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Val MAE {val_mae:.4f}")

        val_maes.append(val_mae)

        # CLearML Metrics
        # logger.report_scalar("Loss", "Train", train_loss / len(train_loader), epoch)
        # logger.report_scalar("MAE", "Validation", val_mae, epoch)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), Config.SAVE_PATH)

    return train_losses, val_maes

def validate(model, val_loader, device, mae_metric):
    model.eval()
    mae_metric.reset()

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: batch[k].to(device) for k in ["input_ids", "attention_mask", "image", "mass"]}
            labels = batch["label"].to(device)
            preds = model(**inputs)
            mae_metric.update(preds, labels)

    return mae_metric.compute().item()


# ---------------------------------------------------
# if __name__ == '__name__':
#     Task.set_credentials(
#         api_host="https://api.clear.ml",
#         web_host="https://app.clear.ml/",
#         files_host="https://files.clear.ml",
#         key="YUY73QE44SA3GEKMPA606O8HGXE3HX",
#         secret="S0uzcyJgU0O1PE708mia0kPfmMvjAVRC9dlDB9x5S1ASj_J8vDxMEnRpIb6ukaso6ho"
#     )

#     task = Task.init(project_name='CaloriePrediction',
#                     task_name='Multimodal_Calorie_Model',
#                     reuse_last_task_id=False)
#     logger = Logger.current_logger()

#     train(logger)

#     task.close()

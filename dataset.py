import torch
from torch.utils.data import Dataset
from PIL import Image
import timm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import albumentations as A
from sklearn.preprocessing import StandardScaler


class MultimodalCalorieDataset(Dataset):
    def __init__(self, config, transforms, split="train"):
        self.dish_df = pd.read_csv(config.DISH_CSV)
        self.ingr_df = pd.read_csv(config.INGR_CSV).set_index('id')
        self.images_root = config.IMGS_ROOT

        # StandardScaler для массы
        self.mass_scaler = StandardScaler()
        self.dish_df['total_mass_scaled'] = self.mass_scaler.fit_transform(self.dish_df[['total_mass']])

        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = transforms
        self.split = split
        self.df = self.dish_df[self.dish_df['split'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        # Извлекаем ингредиенты
        ingr_ids = row["ingredients"].split(";")
        ingr_names = [self.ingr_df.loc[int(x.replace("ingr_", "")), "ingr"] for x in ingr_ids if x]

        text = ", ".join(ingr_names)

        # Загружаем картинку
        img_path = f"{self.images_root}/{row['dish_id']}/rgb.png"
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image=np.array(image))["image"]

        # Токенизируем текст
        tokenized_text = self.tokenizer(text,
                                        return_tensors="pt",
                                        padding="max_length",
                                        truncation=True)
        # Убираем размер батча для модели
        for k in tokenized_text:
            tokenized_text[k] = tokenized_text[k].squeeze(0)

        label = torch.tensor(row["total_calories"], dtype=torch.float)
        scaled_mass = torch.tensor(row['total_mass_scaled'], dtype=torch.float)

        return {
            "image": image,
            "text_input": tokenized_text,
            "label": label,
            "mass": scaled_mass
        }


def collate_fn(batch, tokenizer):
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)

    texts = [item["text_input"] for item in batch]

    # Собираем токены по ключам input_ids, attention_mask и (если есть) token_type_ids
    input_ids = torch.stack([t["input_ids"] for t in texts])
    attention_mask = torch.stack([t["attention_mask"] for t in texts])
    # если используется token_type_ids
    token_type_ids = torch.stack([t["token_type_ids"] for t in texts]) if "token_type_ids" in texts[0] else None
    masses = torch.tensor([item["mass"] for item in batch], dtype=torch.float)

    batch_dict = {
        "image": images,
        "label": labels,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mass": masses
    }
    if token_type_ids is not None:
        batch_dict["token_type_ids"] = token_type_ids

    return batch_dict


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose([
            A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
            A.RandomCrop(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),

            # Ваши аугментации:
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.8),
            A.Affine(scale=(0.8, 1.2),
                     translate_percent=(-0.1, 0.1),
                     shear=(-10, 10),
                     rotate=(-15, 15),
                     p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(std_range=(0.1, 0.5), p=0.4),

            A.Normalize(mean=cfg.mean, std=cfg.std),
            A.ToTensorV2(p=1.0)
        ], seed=42)
    else:
        transforms = A.Compose([
            A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
            A.CenterCrop(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            A.ToTensorV2(p=1.0)
        ])

    return transforms

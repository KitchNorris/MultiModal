from torchvision.models import resnet50, ResNet50_Weights
import timm
from transformers import AutoModel


# Загрузка модели
# weights = ResNet50_Weights.IMAGENET1K_V1
# model = resnet50(weights=weights)

# print(model)

text_model_name = 'bert-base-uncased'
image_model_name = 'resnet50'

text_model = AutoModel.from_pretrained(text_model_name)
image_model = timm.create_model(
                        image_model_name,
                        pretrained=True,
                        num_classes=0)

#print(image_model)
print(text_model)

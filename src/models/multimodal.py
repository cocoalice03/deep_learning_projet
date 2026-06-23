import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from src.models.supervised import build_model


class MultiModalNet(nn.Module):
    def __init__(
        self,
        num_labels=14,
        image_model="resnet18",
        text_model_name="distilbert-base-uncased",
        image_embedding_dim=512,
        text_embedding_dim=768,
        pretrained_image=True,
        pretrained_text=True,
        text_config=None,
    ):
        super().__init__()
        self.image_model = image_model
        self.text_model_name = text_model_name
        self.image = build_model(image_model, image_embedding_dim, pretrained=pretrained_image)
        if pretrained_text:
            self.text = AutoModel.from_pretrained(text_model_name)
        else:
            if text_config is not None:
                config_kwargs = dict(text_config)
                model_type = config_kwargs.pop("model_type")
                config = AutoConfig.for_model(model_type, **config_kwargs)
            else:
                config = AutoConfig.from_pretrained(text_model_name, local_files_only=True)
            self.text = AutoModel.from_config(config)
        self.cls = nn.Sequential(
            nn.Linear(image_embedding_dim + text_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )

    def forward(self, img, input_ids, attention_mask):
        image_features = self.image(img)
        text_features = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.cls(torch.cat([image_features, text_features], dim=1))

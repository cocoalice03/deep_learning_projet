import torch.nn as nn
import torchvision.models as tvm

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

def build_model(name, num_classes=14, pretrained=True, image_size=224):
    name = name.lower()
    if name == "simple_cnn":
        return SimpleCNN(num_classes)
    if name == "resnet18":
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        m = tvm.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "densenet121":
        weights = tvm.DenseNet121_Weights.DEFAULT if pretrained else None
        m = tvm.densenet121(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    if name == "vit":
        weights = tvm.ViT_B_16_Weights.DEFAULT if pretrained else None
        kwargs = {} if pretrained else {"image_size": image_size}
        m = tvm.vit_b_16(weights=weights, **kwargs)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    raise ValueError(f"Modèle inconnu: {name}")

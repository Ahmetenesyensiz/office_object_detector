import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision import models

class CustomCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CustomCNN, self).__init__()
        
        # ResNet50 modelini yükle
        self.resnet = models.resnet50(pretrained=True)
        
        # Son katmanı değiştir
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

def get_model(device):
    """Modeli oluşturur ve belirtilen cihaza taşır"""
    model = CustomCNN()
    model = model.to(device)
    
    # Model bilgilerini logla
    logging.info(f"Model oluşturuldu ve {device} cihazına taşındı")
    logging.info(f"Model parametre sayısı: {sum(p.numel() for p in model.parameters())}")
    
    return model 
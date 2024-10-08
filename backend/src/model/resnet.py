import torch.nn as nn
from torchvision import models

def create_resnet(n_classes: int = 2, model_name: str = 'resnet_18', load_pretrained: bool = False):
    if model_name == 'resnet_18':
        if load_pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18()
    elif model_name == 'resnet_34':
        if load_pretrained:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet34()
    else:
        raise ValueError(f'Invalid model name: [resnet_18, resnet_34]')
    
    if load_pretrained:
        backbone = nn.Sequential(*list(model.children())[:-1])
        
        for param in backbone.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)
    return model
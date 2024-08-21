import torch.nn as nn
from torchvision.models import mobilenet_v2, mobilenet_v3_small

def create_mobilenet(n_classes: int, model_name: str = 'mobilenet_v2', load_pretrained: bool = False):
    if model_name == 'mobilenet_v2':
        if load_pretrained:
            model = mobilenet_v2(mobilenet_v2.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            model = mobilenet_v2()
    elif model_name == 'mobilenet_v3_small':
        if load_pretrained:
            model = mobilenet_v3_small(mobilenet_v3_small.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            model = mobilenet_v3_small()
    else:
        raise ValueError('Invalid model name. [mobilenet_v2, mobilenet_v3_small]')

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    return model
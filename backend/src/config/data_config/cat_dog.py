from dataclasses import dataclass
from torchvision import transforms

@dataclass
class CatDogData:
    n_classes = 2
    img_size = 224
    classes = ['cat', 'dog']
    id2label = {0: 'cat', 1: 'dog'}
    label2id = {'cat': 0, 'dog': 1}
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
import argparse

import torchvision

from utils import Logger, AppPath, seed_everything
from config.data_config import CatDogData
from model import create_resnet, create_mobilenet, Trainer

LOGGER = Logger(__file__)
LOGGER.log.info('Starting Model Training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', type=str, required=True, 
                        help='Version/directory to be used for training')
    parser.add_argument('--model_name', type=str, default='resnet_18',
                        choices=['resnet_18', 'resnet_34', 'mobilenet_v2', 'mobilenet_v3_small'],
                        help='Model to be used for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--best_model_metric', type=str, default='val_loss',
                        choices=['val_loss', 'val_acc'],
                        help='Metric for selecting the best model to logging to MLflow')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='Device to be used for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility')
    parser.add_argument('--load_pretrained', action='store_true',
                        help='Using pretrained model for training')
    args = parser.parse_args()
    seed_everything(args.seed)
    
    try:
        data_path = AppPath.TRAIN_DATA_DIR / args.data_version
        assert data_path.exists()
    except AssertionError:
        LOGGER.log.info(f'Data version: {args.data_version} not found.')
        raise FileNotFoundError(f'Data version: {args.data_version} not found.')

    train_data = torchvision.datasets.ImageFolder(
        root=AppPath.TRAIN_DATA_DIR/args.data_version/'train',
        transform=CatDogData.train_transform
    )
    
    val_data = torchvision.datasets.ImageFolder(
        root=AppPath.TRAIN_DATA_DIR/args.data_version/'val',
        transform=CatDogData.test_transform
    )
    
    test_data = torchvision.datasets.ImageFolder(
        root=AppPath.TRAIN_DATA_DIR/args.data_version/'test',
        transform=CatDogData.test_transform
    )
    
    model_prefix = args.model_name.split('_')[0]
    if model_prefix == 'resnet':
        model = create_resnet(n_classes=CatDogData.n_classes, model_name=args.model_name, load_pretrained=args.load_pretrained)
    elif model_prefix == 'mobilenet':
        model = create_mobilenet(n_classes=CatDogData.n_classes, model_name=args.model_name, load_pretrained=args.load_pretrained)
    
    mlflow_log_tags = {
        'data_version': args.data_version,
        'id2label': CatDogData.id2label,
        'label2id': CatDogData.label2id
    }
    LOGGER.log.info(f'Model training tags: {mlflow_log_tags}')
    
    mlflow_log_params = {
        'model': model.__class__.__name__,
        'model_name': args.model_name,
        'n_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'best_model_metric': args.best_model_metric,
        'device': args.device,
        'seed': args.seed,
        'n_classes': CatDogData.n_classes,
        'image_size': CatDogData.img_size,
        'image_mean': CatDogData.mean,
        'image_std': CatDogData.std,
    }
    LOGGER.log.info(f'Model training params: {mlflow_log_params}')
    
    trainer = Trainer(
        model=model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        mlflow_log_tags=mlflow_log_tags,
        mlflow_log_params=mlflow_log_params,
        device=args.device,
        best_model_metric=args.best_model_metric,
        verbose=True
    )
    
    trainer.train()
    LOGGER.log.info(f'Model Training Completed. Model: {args.model_name}, Data: {args.data_version}')
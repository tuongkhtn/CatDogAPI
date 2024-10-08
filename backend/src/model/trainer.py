import os
import mlflow

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logger import Logger

from dotenv import load_dotenv
load_dotenv()

LOGGER = Logger(__file__)
LOGGER.log.info('Trainer')

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME')

try:
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
    LOGGER.log.info(f'MLFLOW TRACKING URI: {MLFLOW_TRACKING_URI}')
except Exception as e:
    LOGGER.log.error(f'Error: {e}')
    raise e

class Trainer:
    def __init__(
        self,
        model,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float,
        train_data,
        val_data,
        batch_size: int,
        mlflow_log_tags,
        mlflow_log_params,
        device: str,
        best_model_metric: str,
        verbose=False,
    ) -> None:
        self.model = model.to(device)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.mlflow_log_params = mlflow_log_params
        self.mlflow_log_tags = mlflow_log_tags
        self.device = device
        self.best_model_metric = best_model_metric
        self.verbose = verbose
        
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        
        run_name = f"{self.mlflow_log_params['model_name']} - {self.mlflow_log_tags['data_version']}"
        
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(self.mlflow_log_tags)
            
            mlflow.log_params({
                'optimizer': optimizer.__class__.__name__,
                'criterion': self.criterion.__class__.__name__
            })
            mlflow.log_params(self.mlflow_log_params)
            
            best_val_loss = float('inf')
            best_val_acc = float('-inf')
            best_val_loss_state_dict = None
            best_val_acc_state_dict = None
            
            for epoch in range(self.num_epochs):
                self.model.train()
                running_loss = 0.0 
                running_corrects = 0
                running_total = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, dim=1)
                    running_corrects += (predicted == labels).sum().item()
                    running_total += labels.size(0)
                    
                epoch_loss = running_loss / len(train_loader)
                epoch_acc = running_corrects / running_total

                mlflow.log_metric('training_loss', f'{epoch_loss:.4f}', step=epoch)
                mlflow.log_metric('training_acc', f'{epoch_acc:.4f}', step=epoch)
                
                val_loss, val_acc = self.evaluate(val_loader, epoch=epoch)
                
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    best_val_loss_state_dict = self.model.state_dict()
                
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_val_acc_state_dict = self.model.state_dict()
                
                if self.verbose:
                    LOGGER.log.info(f"Epoch {epoch+1}/{self.num_epochs} Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
                    
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("best_val_acc", best_val_acc)
            
            if self.best_model_metric == "val_loss":
                best_model_state_dict = best_val_loss_state_dict
                LOGGER.log.info(f"Best model metric: {self.best_model_metric} - Best val loss: {best_val_loss:.4f}")
            elif self.best_model_metric == "val_acc":
                best_model_state_dict = best_val_acc_state_dict
                LOGGER.log.info(f"Best model metric: {self.best_model_metric} - Best val acc: {best_val_acc:.4f}")
            else:
                raise ValueError(f"Invalid best_model_metric: {self.best_model_metric}")
            
            self.model.load_state_dict(best_model_state_dict)
            mlflow.pytorch.log_model(self.model, "model")
        
    def evaluate(self, val_loader, epoch):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        running_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, dim=1)
                running_corrects += (predicted == labels).sum().item()
                running_total += labels.size(0)
        
        val_loss = running_loss / len(val_loader)
        val_acc = running_corrects / running_total
        
        mlflow.log_metric("val_loss", f"{val_loss:.4f}", step=epoch)
        mlflow.log_metric("val_acc", f"{val_acc:.4f}", step=epoch)
        
        return val_loss, val_acc
            
    def test(self, test_data):
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        return self.evaluate(test_loader, epoch=0)
    
    def predict(self, image, transform, class_names):
        self.model.eval()
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, dim=1)
            predicted_class = class_names[predicted.item()]
        return predicted_class
                   
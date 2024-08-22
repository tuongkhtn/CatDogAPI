import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import ast
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

import mlflow
from mlflow.tracking import MlflowClient

from utils import AppPath, Logger, save_cache

from dotenv import load_dotenv
load_dotenv()

LOGGER = Logger(__file__, log_file='predictor.log')
LOGGER.log.info('Starting Model Serving')

class Predictor:
    def __init__(self, model_name: str, model_alias: str, device: str = 'cpu'):
        self.model_name = model_name
        self.model_alias = model_alias
        self.device = device
        self.load_model()
        self.create_transform()
        
    async def predict(self, image, image_name):
        pil_img = Image.open(image)
        LOGGER.save_requests(pil_img, image_name)
        
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')
        
        transformed_img = self.transforms_(pil_img).unsqueeze(0)
        output = await self.model_inference(transformed_img)
        probs, best_prob, pred_id, pred_class = self.output2pred(output)
        
        LOGGER.log_model(self.model_name, self.model_alias)
        LOGGER.log_response(best_prob, pred_id, pred_class)
        
        torch.cuda.empty_cache()
        save_cache(
            image_name=image_name,
            image_path=AppPath.CAPTURED_DATA_DIR,
            predicted_name=self.model_name,
            predicted_alias=self.model_alias,
            probs=probs,
            best_prob=best_prob,
            pred_id=pred_id,
            pred_class=pred_class
        )
        
        return {
            'probs': probs,
            'best_prob': best_prob,
            'predicted_id': pred_id,
            'predicted_class': pred_class,
            'predictor_name': self.model_name,
            'predictor_alias': self.model_alias
        }
        
    def create_transform(self):
        self.transforms_ = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
    async def model_inference(self, input):
        input = input.to(self.device)
        with torch.no_grad():
            output = self.loaded_model(input).cpu()
        return output
    
    def load_model(self):
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        LOGGER.log.info(f'MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}')
        
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()
            model_mv = client.get_model_version_by_alias(name=self.model_name, alias=self.model_alias)
            
            LOGGER.log.info(f'Model loaded: {self.model_name} - {self.model_alias}')
            
            run_info = client.get_run(model_mv.run_id)
            
            self.id2class = ast.literal_eval(run_info.data.tags['id2label'])
            self.class2id = ast.literal_eval(run_info.data.tags['label2id'])
            self.mean = ast.literal_eval(run_info.data.params['image_mean'])
            self.std = ast.literal_eval(run_info.data.params['image_std'])
            self.img_size = ast.literal_eval(run_info.data.params['image_size'])
            self.loaded_model = mlflow.pytorch.load_model(model_mv.source, map_location=self.device)
        
        except Exception as e:
            LOGGER.log.error(f'Load model failed')
            LOGGER.log.error(f'ERROR: {e}')
            return None
        
    def output2pred(self, output):
        probabilities = F.softmax(output, dim=1)
        best_prob = torch.max(probabilities, 1)[0].item()
        pred_id = torch.max(probabilities, 1)[1].item()
        pred_class = self.id2class[pred_id]
        return probabilities.squeeze().tolist(), round(best_prob, 6), pred_id, pred_class


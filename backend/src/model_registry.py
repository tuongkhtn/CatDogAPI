import os
import json
import argparse
from dataclasses import asdict

import mlflow
from mlflow.tracking import MlflowClient

from utils import Logger, AppPath
from config.serve_config import BaseServeConfig

from dotenv import load_dotenv
load_dotenv()

LOGGER = Logger(__file__)
LOGGER.log.info('Starting Model Registry')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='raw_data',
                        help='Name of the config file')
    parser.add_argument('--filter_string', type=str, default="",
                        help='Filter string or searching runs in MLflow Tracking Server')
    parser.add_argument('--best_metric', type=str, default='best_val_loss',
                        choices=['best_val_loss', 'best_val_acc'],
                        help='Metric for selecting the best model')
    parser.add_argument('--model_alias', type=str, default='Production',
                        help='Alias tag of the model. Help to identify the model in the model registry.')
    args = parser.parse_args()
    
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    LOGGER.log.info(f'MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}')
    
    MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME')
    experiment_ids = dict(mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME))['experiment_id']
    
    client = MlflowClient()
    
    try:
        best_run = client.search_runs(
            experiment_ids,
            filter_string=args.filter_string,
            order_by=[f'metrics.{args.best_metric} desc']
        )[-1]
    except:
        LOGGER.log.info('No runs found')
        exit(0)
    
    LOGGER.log.info(f'Best run: {best_run.info.run_id}')
    
    model_name = best_run.data.params['model_name']
    
    try:
        client.create_registered_model(model_name)
    except:
        pass
    
    run_id = best_run.info.run_id
    model_uri = f'runs:/{run_id}/model'
    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    LOGGER.log.info(f'Registered Model: {model_name}, version: {mv.version}')
    
    client.set_registered_model_alias(name=model_name, alias=args.model_alias, version=mv.version)
    
    serve_config = BaseServeConfig(config_name=args.config_name, model_name=model_name, model_alias=args.model_alias)
    
    path_save_cfg = AppPath.SERVE_CONFIG_DIR / f'{args.config_name}.json'
    with open(path_save_cfg, 'w+') as f:
        json.dump(asdict(serve_config), f, indent=4)
    
    LOGGER.log.info(f'Config saved to {args.config_name}.json')
    
    LOGGER.log.info(f'Model {model_name} registered with alias {args.model_alias} and version {mv.version}')
    LOGGER.log.info('Model Registry completed')
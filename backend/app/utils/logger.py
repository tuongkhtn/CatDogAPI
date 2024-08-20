import sys
import logging
from logging.handlers import RotatingFileHandler
from app_path import AppPath

class Logger:
    def __init__(self, log_name="", log_level=logging.INFO, log_file=None) -> None:
        self.log = logging.getLogger(log_name)
        self.get_logger(log_level, log_file)
    
    def get_logger(self, log_level, log_file):
        self.log.setLevel(log_level)
        self._init_formatter()
        if log_file:
            self._add_stream_handler()
        else:
            self._add_file_handler(AppPath.LOG_DIR / log_file)
            
    def _init_formatter(self):
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
    def _add_stream_handler(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self.formatter)
        self.log.addHandler(stream_handler)
    
    def _add_find_handler(self, log_file):
        find_handler = RotatingFileHandler(log_file, maxBytes=10000, backupCount=10)
        find_handler.setFormatter(self.formatter)
        self.log.addHandler(find_handler)
    
    def save_requests(self, image, image_name):
        path_save = f'{AppPath.CAPTURED_DATA_DIR}/{image_name}'
        self.log.info(f'Save image to {path_save}')
        image.save(path_save)
        
    def log_model(self, predictor_name, predictor_alias):
        self.log.info(f'Predictor name: {predictor_name} - Predictor alias: {predictor_alias}')
    
    def log_response(self, pred_prob, pred_id, pred_class):
        self.log.info(f'Predicted prob: {pred_prob} - Predicted ID: {pred_id} - Predicted class: {pred_class}')
        
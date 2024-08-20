import sys
import logging

class Logger:
    def __init__(self, log_name="", log_level=logging.INFO):
        self.log = logging.getLogger(log_name)
        self.get_logger(log_level)
        
    def get_logger(self, log_level):
        self.log.setLevel(log_level)
        self._init_formatter()
        self._add_stream_handler()
        
    def _init_formatter(self):
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
    def _add_stream_handler(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self.formatter)
        self.log.addHandler(stream_handler)
    
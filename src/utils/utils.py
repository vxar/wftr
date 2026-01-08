"""
Utility module for logging
Provides logger functionality for WebullUtil
"""
import logging
import sys

class Logger:
    """Simple logger wrapper"""
    
    def __init__(self):
        self.logger = logging.getLogger('WebullUtil')
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def get_logger(self):
        """Get the logger instance"""
        return self.logger

# Create singleton logger instance
logger = Logger()


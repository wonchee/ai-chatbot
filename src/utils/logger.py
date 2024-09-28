import os
import logging

log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

class Logger():
    
    def __init__(self) -> None:
        log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_level = log_level_map.get(log_level_str, logging.INFO)

        # Configure logging
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)

        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        self.logger = logger
    

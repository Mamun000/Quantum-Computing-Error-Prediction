import logging
import os
from datetime import datetime

def setup_logging():
    """Setup simple logging with daily folders"""
    
    # Create logs directory with date-based subfolder
    current_date = datetime.now().strftime('%Y-%m-%d')
    logs_dir = os.path.join(os.getcwd(), "logs", current_date)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log filename
    log_filename = f"quantum_error_{datetime.now().strftime('%H-%M-%S')}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # Basic logging configuration
    logging.basicConfig(
        filename=log_filepath,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)
    
    # Log startup
    logging.info("=" * 50)
    logging.info("🚀 Quantum Error Prediction System Started")
    logging.info("=" * 50)
    
    return log_filepath

# Initialize logging
LOG_FILE_PATH = setup_logging()
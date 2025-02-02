import logging
from pathlib import Path

def setup_logging(
    log_level: int = logging.INFO,
    log_file: str = None
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler if specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def ensure_dir(dir_path: str) -> Path:
    """Ensure directory exists, create if not.
    
    Args:
        dir_path: Directory path
        
    Returns:
        Path object for directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path 
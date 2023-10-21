from pathlib import Path
from typing import List
import json
import sys
from src.utils.loggers import debug_logger, info_logger, error_logger

def load_config(config_path: str) -> dict:
    """
    Load the JSON configuration from the given path.

    Parameters:
    - config_path (str): The path to the JSON configuration file.

    Returns:
    - dict: The loaded configuration.
    """
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        error_logger.error("Could not find config.json.")
        sys.exit(1)
    except json.JSONDecodeError:
        error_logger.error("Could not decode JSON in config.json.")
        sys.exit(1)

def get_limited_images(folder_path: str, maximum_count: int = 0) -> List[str]:
    """
    Get up to `maximum_count` image paths from the given folder.
    
    Parameters:
    - folder_path (str): The path to the folder containing the images.
    - maximum_count (int): The maximum number of image paths to return.

    Returns:
    - List[str]: A list of up to `maximum_count` image paths.
    """
    config = load_config("D:/_Projetos/Python/customAI/config/config.json")
    
    if maximum_count == 0:
        max_batches = 1
        maximum_count = max_batches * config["batch_Size"] // 2

    # Using pathlib for better path manipulation
    folder = Path(folder_path)
    img_paths = [str(img_path) for img_path in folder.iterdir() if img_path.is_file()][:maximum_count]
    
    return img_paths
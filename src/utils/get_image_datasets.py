from src.utils.loggers import error_logger, info_logger
import sys
import json
import random
from src.datasets import CustomDataset
from src.utils import data_augmentation
import torch
from torch.utils.data import DataLoader, random_split

def load_config():
    try:
        with open('D:/_Projetos/Python/customAI/config/config.json') as f:
            return json.load(f)
    except FileNotFoundError:
        error_logger.error('Could not find config.json.')
        sys.exit(1)
    except json.JSONDecodeError:
        error_logger.error('Could not decode JSON in config.json.')
        sys.exit(1)

def get_train_val_sizes(dataset, validation_split):
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    return train_size, val_size

def get_data_loaders(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader

def get_image_datasets():
    config = load_config()
    info_logger.info("Loading DataSet.")

    # Initialize and augment dataset
    custom_dataset = CustomDataset()
    info_logger.info("Generating additional data...")
    custom_dataset.create_more_data()
    info_logger.info("Additional data created.")

    # Randomize dataset
    validation_split = random.uniform(0.09, 0.27)
    train_size, val_size = get_train_val_sizes(custom_dataset, validation_split)
    info_logger.info(f"Train size: {train_size}, Validation size: {val_size}")

    # Create train and validation sets
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        info_logger.info("Failed to generate Train or Validation Dataset. Exiting.")
        sys.exit(-1)

    # Create data loaders
    batch_size = config['batch_Size']
    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset, batch_size)

    info_logger.info("DataLoader setup finished.")
    return train_loader, val_loader

import sys

import wandb
from loguru import logger

from constants import Constants as const


def init_logger_and_wandb(config):
    # Create a descriptive run name with key parameters
    run_name = f"{config.variant}_{config.backbone}_{config.split}"
    
    # Add modality info
    if isinstance(config.modality, list):
        modality_str = "+".join(config.modality)
    else:
        modality_str = config.modality
    run_name += f"_{modality_str}"
    
    # Add training parameters
    run_name += f"_e{config.num_epochs}_bs{config.batch_size}_lr{config.lr}"
    
    wandb.init(
        project=config.model_name,
        name=run_name,
        config=config,
    )
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> ",
            },
            {
                "sink": "logging/" + "logger_{time}.log",
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> ",
            },
        ],
        "extra": {"user": "usr"},
    }
    logger.configure(**config)

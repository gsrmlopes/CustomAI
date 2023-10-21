from typing import Dict
import os
import win32api
import win32process
import psutil

import sys
import json
from src.models import GeneticAlgorithm
from src.utils.loggers import info_logger, error_logger


def load_config(config_path: str) -> Dict:
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


def main(config: Dict):
    """
    The main function runs a genetic algorithm until certain metrics are met.

    Parameters:
    - config (Dict): Configuration dictionary.
    """
    passable_metrics = False
    runs = 0

    info_logger.info("Starting Genetic Algorithm.")
    info_logger.info("Running until %s generations.", config["max_Generations"])
    info_logger.info("Population size: %s", config["population_Size"])
    info_logger.info("Batch size: %s", config["batch_Size"])

    while not passable_metrics:
        runs += 1
        info_logger.info("Run #%d", runs)

        ga = GeneticAlgorithm(config["population_Size"])
        info_logger.info("Genetic Algorithm created.")
        info_logger.info("Starting Genetic Algorithm.")

        loopable = ga.run()

        info_logger.info("Genetic Algorithm finished.")
        if loopable:
            passable_metrics = loopable


if __name__ == "__main__":
    # limit the Core Count to 3
    if os.name == "nt":
        # Windows
        handle = win32api.GetCurrentProcess()
        # The process affinity mask is a bit vector where each bit represents a logical processor.
        # Here we're setting it to 7 (in binary: 111), which enables the process to use CPU 0, CPU 1 and CPU 2.
        processAffinityMask = 7
        win32process.SetProcessAffinityMask(handle, processAffinityMask)

        win32process.SetPriorityClass(
            win32api.GetCurrentProcess(), win32process.HIGH_PRIORITY_CLASS
        )
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    else:
        # Linux
        resource.setrlimit(resource.RLIMIT_AS, (4000000000, 4000000000))
        resource.setrlimit(resource.RLIMIT_NPROC, (3, 3))

    # load the configuration
    config_path = "D:/_Projetos/Python/customAI/config/config.json"
    config = load_config(config_path)
    main(config)

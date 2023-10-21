import os
import json
import sys
import torch
from PIL import Image
from torchvision import transforms
import time

from src.utils.loggers import error_logger
from src.models.custom_model import CustomModel

try:
    with open("D:/_Projetos/Python/customAI/config/config.json", encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    error_logger.error("Could not find config.json.")
    error_logger.error("Filename: %s", __file__)
    sys.exit(1)
except json.JSONDecodeError:
    error_logger.error("Could not decode JSON in config.json.")
    sys.exit(1)


def find_model():
    main_path = config["model_Save_Folder"]
    model_name = config["model_Name"]
    supposed_best_model = config["best_model_path"]
    supposed_second_best_model = config["second_best_model_path"]
    model = CustomModel() 
    if supposed_best_model is not None:
        if os.path.isfile(supposed_best_model):
            try:
                model = torch.load(config["best_model_path"])

            except Exception as e:
                error_logger.error("Could not load best model. Error: %s", str(e))
    else:
        if os.path.isfile(supposed_second_best_model):
            try:
                model = torch.load(config["second_best_model_path"])
            except Exception as e:
                error_logger.error(
                    "Could not load second best model. Error: %s", str(e)
                )
    if model is None:
        # scan all the main_path folder and folders under it to try to find the model with the model_name
        for root, _, files in os.walk(main_path):
            for file in files:
                if file == model_name:
                    model = torch.load(os.path.join(root, file))
                    loaded_object = torch.load(os.path.join(root, file))
                    print(type(loaded_object))
                    if (
                        isinstance(loaded_object, dict)
                        and "model_state_dict" in loaded_object
                    ):
                        model = (
                            torch.nn.Module()
                        )  # Assuming the model is of some torch.nn.Module subclass
                        model.load_state_dict(loaded_object["model_state_dict"])
                    elif isinstance(loaded_object, torch.nn.Module):
                        model = loaded_object
                    else:
                        error_logger.error(
                            "Unexpected object type: %s", type(loaded_object)
                        )
                return model


def load_model_pth():
    model = find_model()
    folder_to_scan = config["toScan"]
    try:
        if not os.path.isdir(folder_to_scan):
            error_logger.error("Could not find folder to scan: %s", folder_to_scan)
            sys.exit(1)

        for root, _, files in os.walk(config["base_path"]):
            for file in files:
                if file == config["model_Name"] + ".pth":
                    try:
                        model = torch.load(os.path.join(root, file))
                    except Exception as e:
                        error_logger.error(
                            "Could not load model with config stats. Error: %s", str(e)
                        )
    except Exception as e:
        error_logger.error("Could not load model with config stats. Error: %s", str(e))

    if model is None:
        error_logger.error("Could not find any model.")
        try:
            model = torch.load("D:/_Projetos/Python/customAI/_best_model.pth")
            # model = torch.load("D:/Data/models/BloodFaceNet_best.pth")
        except Exception as e:
            error_logger.error(
                "Could not load model with config stats. Error: %s", str(e)
            )
    return model


def set_parameters():
    try:
        model = load_model_pth()
        if model is not None:
            model.parameters()
    except Exception as e:
        error_logger.error("Could not load model with config stats. Error: %s", str(e))
        sys.exit(1)


def interfere():
    """
    Load the PyTorch model and use it to make predictions on images in a folder.
    """
    model = load_model_pth()
    folder_to_scan = config["toScan"]
    try:
        if not os.path.isdir(folder_to_scan):
            error_logger.error("Could not find folder to scan: %s", folder_to_scan)
            sys.exit(1)

        for root, _, files in os.walk(folder_to_scan):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    try:
                        img = Image.open(os.path.join(root, file))
                        img = img.convert("RGB")
                        img = transforms.ToTensor()(img)
                        img = img.unsqueeze(0)
                        output = model(img)
                        _, preds = torch.max(output, dim=1)
                        print(file, preds.item())
                    except Exception as e:
                        error_logger.error(
                            "Could not make prediction for image %s. Error: %s",
                            file,
                            str(e),
                        )
    except Exception as e:
        error_logger.error(
            "Could not scan folder %s. Error: %s", folder_to_scan, str(e)
        )


def execution_loop():
    """
    Set a timer for 10 seconds. When the timer triggers, execute a scan to check if the folder has new images.
    If it has, execute the interfere function. If it does not, do nothing and wait the next 10 seconds.
    """
    while True:
        print("running")
        interfere()
        time.sleep(10)


if __name__ == "__main__":
    # set_parameters()
    execution_loop()
    # interfere()

from doctest import debug
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import sys
import json
import math
import random

from src.utils import get_limited_images
from src.utils.loggers import debug_logger, info_logger, error_logger

try:
    with open("D:/_Projetos/Python/customAI/config/config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    error_logger.error("Could not find config.json.")
    error_logger.error("Finlename: %s", __file__)
    sys.exit(1)
except json.JSONDecodeError:
    error_logger.error("Could not decode JSON in config.json.")
    sys.exit(1)


class CustomDataset(Dataset):
    # def __init__(self, folder1_path, folder2_path, transform=None):
    def __init__(self):
        """The `super().__init__()` line is calling the constructor of the parent class `Dataset`. This is
        necessary to properly initialize the `Dataset` object."""

        super().__init__()
        self.folder1_path = config["folder_1"]
        debug_logger.debug("Folder 1 path: %s", config["folder_1"])
        self.folder2_path = config["folder_2"]
        debug_logger.debug("Folder 0 path: %s", config["folder_2"])
        debug_logger.debug("Origin 1 path: %s", config["output_1"])
        debug_logger.debug("Origin 0 path: %s", config["output_2"])

        test_cap = config["test_Run"]
        if test_cap:
            maximum_batches = 3
            maximum_count = math.floor(maximum_batches * config["batch_Size"] // 2)

            self.images = get_limited_images(
                self.folder1_path, maximum_count
            ) + get_limited_images(self.folder2_path, maximum_count)
            for idx in range(len(self.images)):
                if self.images is None:
                    print("Warning: self.images is None.")
                elif idx >= len(self.images):
                    print(f"Warning: Index {idx} is out of range.")

            self.labels = [0] * maximum_count + [1] * maximum_count
            for idx in range(len(self.labels)):
                if self.labels is None:
                    print("Warning: self.labels is None.")
                elif idx >= len(self.labels):
                    print(f"Warning: Index {idx} is out of range.")

        else:
            info_logger.info("test run disabled!!")
            # always be at least 60.000
            limit = config["image_limit"] - 5
            # may be 1.000 up to 250.000
            real_limit = (
                sum(len(files) for _, _, files in os.walk(config["output_1"]))
                + sum(len(files) for _, _, files in os.walk(config["output_2"]))
                - 5
            )
            info_logger.info("Real Limit Got = %f", real_limit)
            info_logger.info("Set Limit before if = %f", limit)

            if limit > real_limit:
                limit_to_use = real_limit
            else:
                limit_to_use = limit

            info_logger.info("Set Limit = %f", limit_to_use)

            if config["hard_limit"]:
                limit_to_use = config["hard_limit_count"]
                limit_to_use = int(limit_to_use * random.randint(75, 125) / 100)
                info_logger.info("Using %s with limit enabled!!", limit_to_use)

            creation_count = 0
            while limit_to_use <= 10000:
                creation_count += 1
                info_logger.info("Generating Images to Use")
                self.create_more_data()
                info_logger.info("Images Generated %s time(s)", creation_count)
                real_limit = (
                    sum(len(files) for _, _, files in os.walk(config["output_1"]))
                    + sum(len(files) for _, _, files in os.walk(config["output_2"]))
                    - 5
                )
                limit_to_use = real_limit
                info_logger.info("Not Yet! Lets create more!")

            # The count must be the same for the labels
            # Get all the image paths
            self.images = get_limited_images(
                config["output_1"], limit_to_use
            ) + get_limited_images(config["output_2"], limit_to_use)
            self.labels = [0] * len(os.listdir(config["output_1"])) + [1] * len(
                os.listdir(config["output_2"])
            )  # Labels for the images

    def __len__(self):
        """
        The function returns the length of the "images" attribute of an object.
        :return: The length of the "images" attribute of the object.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        The function `__getitem__` takes an index as input, loads an image from a given path, resizes
        it, and returns the image and its corresponding label.

        :param idx: The `idx` parameter in the `__getitem__` method represents the index of the item you
        want to retrieve from the dataset. It is used to access the corresponding image and label from
        the dataset
        :return: The `__getitem__` method returns a tuple containing the image and its corresponding
        label.
        """
        """
        old version:
        img_path = self.images[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image has been correctly loaded
        if img is None:
            print(f"Warning: Failed to load img from {img_path} at index {idx}.")
            return None

        img = cv2.resize(img, (300, 300))
        img = np.expand_dims(img, axis=0)
        if idx >= len(self.labels):
            print(f"Warning: Index {idx} is out of range.")
            return None
        elif self.labels[idx] is None:
            print(f"Warning: Returning None for label at index {idx}.")
            return None
        else:
            label = self.labels[idx]

        return img, label"""
        img_path = self.images[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Failed to load img from {img_path} at index {idx}.")
            img = np.zeros((300, 300))  # return a zero image if loading fails
        img = cv2.resize(img, (300, 300))
        img = np.expand_dims(img, axis=0)
        if idx >= len(self.labels):
            print(f"Warning: Index {idx} is out of range.")
            label = 0  # return a zero label if index is out of range
        elif self.labels[idx] is None:
            print(f"Warning: Returning None for label at index {idx}.")
            label = 0  # return a zero label if label is None
        else:
            label = self.labels[idx]
        return img, label

    def create_more_data(self):
        if config["generate_Images"]:
            origin_folder_train = config["folder_1"]
            origin_folder_test = config["folder_2"]
            output_folder_train = config["output_1"]
            output_folder_test = config["output_2"]

            # Using sets for faster look-up
            image_train_original = set(os.listdir(origin_folder_train))
            image_test_original = set(os.listdir(origin_folder_test))
            image_train_output = set(os.listdir(output_folder_train))
            image_test_output = set(os.listdir(output_folder_test))

            # Using difference() to find the files that are not yet in the output folders
            image_train_original -= image_train_output
            image_test_original -= image_test_output

            output_train_counter = 0
            output_test_counter = 0

            # Debug logs
            debug_logger.debug("New Length Train: %s", len(image_train_original))
            debug_logger.debug("New Length Test: %s", len(image_test_original))

            for i, image_train in enumerate(image_train_original):
                if output_train_counter < 48000:
                    image = cv2.imread(os.path.join(origin_folder_train, image_train))
                    image = cv2.resize(image, (300, 300))
                    cv2.imwrite(
                        os.path.join(output_folder_train, str(i) + ".jpg"), image
                    )
                    output_train_counter += 1
                    if (
                        output_train_counter % 10 == 0
                        or i == len(image_train_original) - 1
                    ):
                        debug_logger.debug(
                            "Generated: %s - TRAIN", output_train_counter
                        )
                else:
                    break

            for i, image_test in enumerate(image_test_original):
                if output_train_counter < 8000:
                    image = cv2.imread(os.path.join(origin_folder_test, image_test))
                    image = cv2.resize(image, (300, 300))
                    cv2.imwrite(
                        os.path.join(output_folder_test, str(i) + ".jpg"), image
                    )
                    output_test_counter += 1
                    if (
                        output_test_counter % 10 == 0
                        or i == len(image_test_original) - 1
                    ):
                        debug_logger.debug("Generated: %s - TEST", output_test_counter)
                else:
                    break

            debug_logger.debug(
                "Generated in total: %s", (output_test_counter + output_train_counter)
            )

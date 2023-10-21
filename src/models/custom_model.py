import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.loggers import debug_logger, info_logger, error_logger
from src.utils import custom_evaluation_index
import random


def load_config():
    try:
        with open("D:/_Projetos/Python/customAI/config/config.json") as f:
            return json.load(f)
    except FileNotFoundError:
        error_logger.error("Could not find config.json.")
        sys.exit(1)
    except json.JSONDecodeError:
        error_logger.error("Could not decode JSON in config.json.")
        sys.exit(1)


config = load_config()


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_layers()
        self.init_metrics()

    def init_layers(self):
        """Initialize layers of the neural network."""
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1401856, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p=0.5)

    def init_metrics(self):
        """Initialize performance metrics."""
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.OMC = 0
        self.custom_index = 0
        self.loss = 0

    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.25)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def crossover(self, other):
        """
        The `crossover` function performs crossover between two models by combining their parameters and
        averaging their OMC and custom_index attributes.

        :param other: The "other" parameter in the crossover function refers to another instance of the
        CustomModel class. It is used to perform crossover between the current instance (self) and the
        other instance
        :return: a child model that is a combination of the self model and the other model.
        """
        child = CustomModel()
        for param1, param2, child_param in zip(
            self.parameters(), other.parameters(), child.parameters()
        ):
            if len(param1.shape) == 2:
                cut = random.randint(0, param1.shape[0] - 1)
                child_param.data[:cut] = param1.data[:cut].clone()
                child_param.data[cut:] = param2.data[cut:].clone()
            else:
                cut = random.randint(0, param1.shape[0] - 1)
                child_param.data[:cut] = param1.data[:cut].clone()
                child_param.data[cut:] = param2.data[cut:].clone()
        child.OMC = (self.OMC + other.OMC) / 2 * 0.67
        child.custom_index = (self.custom_index + other.custom_index) / 2 * 0.67
        return child

    def mutate(self):
        """
        The `mutate` function randomly modifies the parameters of a neural network model and updates two
        other variables.
        """
        for param in self.parameters():
            if param.requires_grad:
                param.data += torch.randn_like(param.data) * random.uniform(0, 0.2)
        self.OMC *= 0.67
        self.custom_index *= 0.67

    def train_model(
        self, train_loader, val_loader, learning_rate, n_epochs, model_config
    ):
        """
        The `train_model` function trains a model using the provided training and validation data
        loaders, learning rate, number of epochs, and model configuration.

        :param train_loader: The `train_loader` parameter is a DataLoader object that provides batches of
        training data to the model during training. It is used to iterate over the training dataset in
        batches
        :param val_loader: The `val_loader` parameter is the data loader for the validation set. It is
        used to iterate over the validation data during the training process
        :param learning_rate: The learning rate is a hyperparameter that determines the step size at each
        iteration while updating the model's parameters. It controls how much the model's parameters are
        adjusted in response to the estimated error. A higher learning rate can result in faster
        convergence but may also cause the model to overshoot the optimal solution
        :param n_epochs: The parameter `n_epochs` represents the number of epochs, which is the number of
        times the model will iterate over the entire training dataset during training
        :param model_config: The `model_config` parameter is a dictionary that contains the configuration
        settings for the model training process. It includes the following keys:
        :return: The function `train_model` returns a dictionary containing the following keys:
        """
        global accuracy, precision, epoch, recall, f1, best_loss, OMC, custom_index
        info_logger.info("Training model.")
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        epoch_losses = []
        epoch_accuracies = []
        epoch_precisions = []
        epoch_recalls = []
        epoch_f1_scores = []
        epoch_overall_scores = []
        epoch_overall_loss = []
        info_logger.info("Training model - initializing training loop.")

        best_loss = float("inf")
        top_models_to_keep = model_config["top_Models_To_Keep"]

        saved_models = []
        model_save_folder = model_config["model_Save_Folder"]
        model_name = model_config["model_Name"]

        logging_interval = max(
            1, int(n_epochs * (2 / 100))
        )  # Log every 2%, but at least once per epoch

        early_stop_counter = 0
        early_stop_patience = (
            5  # Stop training if validation loss does not improve for 5 epochs
        )
        best_val_loss = float("inf")

        for epoch in range(n_epochs):
            debug_logger.debug("Epoch: %s", epoch)
            if (
                epoch + 1
            ) % logging_interval == 0:  # Log only if the current epoch is a multiple of the logging
                # interval
                progress_percent = ((epoch + 1) / n_epochs) * 100
                info_logger.info("Training Progress: %s ", round(progress_percent, 2))

            # Training loop
            batch_losses = []
            true_labels = []
            predicted_labels = []

            # listing the number of batches  to be used in the training
            max_batches = len(train_loader) / config["batch_Size"]
            max_batches = max_batches * config["multiplier"]

            debug_logger.debug("Max batches: %s", max_batches)
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= config["max_steps"] or batch_idx >= max_batches:
                    break
                data = data.to(torch.float32)

                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                true_labels.extend(target.tolist())
                _, predicted = torch.max(output.data, 1)
                predicted_labels.extend(predicted.tolist())
                accuracy = accuracy_score(target.tolist(), predicted.tolist())
                precision = precision_score(
                    true_labels, predicted_labels, average="weighted"
                )
                recall = recall_score(true_labels, predicted_labels, average="weighted")
                f1 = f1_score(true_labels, predicted_labels, average="weighted")
                custom_index = custom_evaluation_index(accuracy, precision, recall)
                OMC = float(accuracy + (precision * 1.5)) / ((f1 * best_loss) * 0.25)

                debug_logger.debug(
                    "Batch %s, Loss: %.4f, Accuracy: %.4f",
                    batch_idx + 1,
                    loss.item(),
                    accuracy,
                )
                if batch_idx % 10 == 0:
                    info_logger.info(
                        "Batch: %s, Accuracy :%.4f  Precision :%.4f"
                        "Epoch :%s, F1 :%.4f",
                        batch_idx + 1,
                        accuracy,
                        precision,
                        epoch,
                        f1,
                    )
                    info_logger.info(
                        "Overall Metric Calculation :%.4f, Custom Index :%.4f",
                        OMC,
                        custom_index,
                    )

            val_losses = []
            val_true_labels = []
            val_predicted_labels = []

            with torch.no_grad():
                for val_data, val_target in val_loader:
                    val_data = val_data.to(torch.float32)
                    val_output = self(val_data)
                    val_loss = criterion(val_output, val_target)
                    val_losses.append(val_loss.item())
                    val_true_labels.extend(val_target.tolist())
                    _, val_predicted = torch.max(val_output.data, 1)
                    val_predicted_labels.extend(val_predicted.tolist())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                model_file_path = os.path.join(
                    model_save_folder, f"{model_name}_best_epoch_{epoch + 1}.pth"
                )

                checkpoint = {
                    "model": CustomModel(),
                    "state_dict": self.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": best_val_loss,
                    "accuracy": accuracy,
                    "precision": precision,
                    "custom_index": custom_index,
                    "OMC": OMC,
                }

                torch.save(checkpoint, model_file_path)
                info_logger.info(
                    "Saved best model with loss: %s to %s",
                    best_val_loss,
                    model_file_path,
                )

                saved_models.append(model_file_path)

                if len(saved_models) > top_models_to_keep:
                    oldest_model = saved_models.pop(0)
                    os.remove(oldest_model)
                    info_logger.info("Removed oldest model: %s", oldest_model)
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    info_logger.info(
                        "Validation loss did not improve for %s epochs. Stopping training.",
                        early_stop_patience,
                    )
                    break

            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(
                true_labels, predicted_labels, average="weighted"
            )
            recall = recall_score(true_labels, predicted_labels, average="weighted")
            f1 = f1_score(true_labels, predicted_labels, average="weighted")
            custom_index = custom_evaluation_index(accuracy, precision, recall)

            epoch_losses.append(sum(batch_losses) / len(batch_losses))
            epoch_accuracies.append(accuracy)
            epoch_precisions.append(precision)
            epoch_recalls.append(recall)
            epoch_f1_scores.append(f1)
            epoch_overall_scores.append(custom_index)
            epoch_overall_loss.append(sum(epoch_losses) / len(epoch_losses))
            OMC = (accuracy + (precision * 1.5)) / ((f1 * best_loss) * 0.25)

            if (epoch + 1) % 25 == 0:
                info_logger.info(
                    "Epoch %s/%s, Loss: %.4f, Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1 Score: %.4f",
                    epoch + 1,
                    n_epochs,
                    epoch_losses[-1],
                    accuracy,
                    precision,
                    recall,
                    f1,
                )

        info_logger.info(
            "Finished epoch %s/%s, Loss: %.4f, Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1 Score: %.4f, OMC: %.4f",
            epoch + 1,
            n_epochs,
            epoch_losses[-1],
            accuracy,
            precision,
            recall,
            f1,
            OMC,
        )

        return {
            "losses": epoch_losses,
            "accuracies": epoch_accuracies,
            "precisions": epoch_precisions,
            "recalls": epoch_recalls,
            "f1_scores": epoch_f1_scores,
            "overall_scores": epoch_overall_scores,
            "overall_loss": epoch_overall_loss,
        }

    def save_model(self):
        """
        The `save_model` function saves the model to a file.
        """
        info_logger.info("Saving model.")
        torch.save(self.state_dict(), config["model_path"])
        info_logger.info("Model saved.")

    def load_model(self):
        """
        The `load_model` function loads a model from a file.
        """
        info_logger.info("Loading model.")
        self.load_state_dict(torch.load(config["model_path"]))
        info_logger.info("Model loaded.")

    def evaluate_model(self, test_loader):
        """
        The `evaluate_model` function evaluates a model using the provided test data loader.

        :param test_loader: The `test_loader` parameter is a DataLoader object that provides batches of
        test data to the model during evaluation. It is used to iterate over the test dataset in batches
        :return: The function `evaluate_model` returns a dictionary containing the following keys:
        """
        info_logger.info("Evaluating model.")
        criterion = nn.CrossEntropyLoss()
        test_losses = []
        test_true_labels = []
        test_predicted_labels = []
        with torch.no_grad():
            for test_data, test_target in test_loader:
                test_data = test_data.to(torch.float32)
                test_output = self(test_data)
                test_loss = criterion(test_output, test_target)
                test_losses.append(test_loss.item())
                test_true_labels.extend(test_target.tolist())
                _, test_predicted = torch.max(test_output.data, 1)
                test_predicted_labels.extend(test_predicted.tolist())
        test_accuracy = accuracy_score(test_true_labels, test_predicted_labels)
        test_precision = precision_score(
            test_true_labels, test_predicted_labels, average="weighted"
        )
        test_recall = recall_score(
            test_true_labels, test_predicted_labels, average="weighted"
        )
        test_f1 = f1_score(test_true_labels, test_predicted_labels, average="weighted")
        test_custom_index = custom_evaluation_index(
            test_accuracy, test_precision, test_recall
        )
        info_logger.info(
            "Test Loss: %.4f, Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1 Score: %.4f",
            sum(test_losses) / len(test_losses),
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
        )
        return {
            "losses": test_losses,
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1_score": test_f1,
            "custom_index": test_custom_index,
        }

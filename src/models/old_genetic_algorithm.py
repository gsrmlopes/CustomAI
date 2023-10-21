# src/models/genetic_algorithm.py
import os
import torch
from src.utils.loggers import debug_logger, info_logger, error_logger
from src.models import CustomModel
from src.utils import get_image_datasets
import sys
import json

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


# The GeneticAlgorithm class is used to implement genetic algorithms in Python.
class GeneticAlgorithm:
    def __init__(self, population_size=8):
        """
        The above function initializes a Genetic Algorithm with a given population size and sets various
        attributes and paths.

        :param population_size: The population size is the number of individuals in each generation of
        the genetic algorithm. It determines the size of the population pool from which new generations
        are created and evaluated. In this case, the default population size is set to 8, defaults to 8
        (optional)
        """
        self.generation = 0
        self.population = []
        self.population_size = population_size
        self.best = None
        self.second_best = None
        self.third_best = None
        self.fitness_sum = 0
        self.mutation_rate = 0.12
        self.mutation_step = 0.23
        self.best_model_path = "best_model.pth"
        self.second_best_model_path = "second_best_model.pth"
        info_logger.info("Genetic Algorithm Initialized.")

    def initialize_population(self):
        """
        The function initializes a population of models by loading the best and second best models if
        they exist, and creating new models for the remaining population size.
        """
        info_logger.info("Initializing Population.")
        if os.path.exists(self.best_model_path):
            info_logger.info("Best Model Exists! Loading best model.")
            checkpoint = torch.load(self.best_model_path)
            best_model = CustomModel()
            best_model.load_state_dict(checkpoint)
            self.population.append(best_model)
        else:
            info_logger.info("Best Model does not exist! Creating new model.")
            self.population.append(CustomModel())


        if os.path.exists(self.second_best_model_path):
            info_logger.info("Second Best Model Exists! Loading second best model.")
            checkpoint = torch.load(self.second_best_model_path)
            second_best_model = CustomModel()
            second_best_model.load_state_dict(checkpoint)
            self.population.append(second_best_model)
        else:
            info_logger.info("Second Best Model does not exist! Creating new model.")
            self.population.append(CustomModel())

        for _ in range(self.population_size - 2):
            info_logger.info("Creating new model. %s", _)
            self.population.append(CustomModel())
            info_logger.info("New model created.")

    def evaluate(self):
        """
        The evaluate function multiplies the fitness of each individual in the population by 100 and
        logs the individual's fitness using a debug logger.
        """
        for individual in self.population:
            individual.fitness *= 100
            individual.custom_index = (
                2 * individual.accuracy * individual.precision * individual.recall
            ) / (individual.accuracy + individual.precision + individual.recall)
            
            if individual.custom_index > self.best.custom_index:
                self.best = individual
                individual.save_model(self.best_model_path)
            debug_logger.debug("Individual fitness: %s", individual.fitness)

    def selection(self):
        """
        The `selection` function sorts a population of individuals based on their fitness, selects the
        best three individuals, saves the state of the best and second best individuals, and logs the
        process.
        """
        info_logger.info("Selection started.")
        debug_logger.debug("Population size: %s", len(self.population))
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best = self.population[0]
        self.second_best = self.population[1]
        self.third_best = self.population[2]
        debug_logger.debug(
            "Best: %s, Second Best: %s, Third Best: %s",
            self.best.fitness,
            self.second_best.fitness,
            self.third_best.fitness,
        )

        # Save the state_dict of best and second_best models
        torch.save(self.best.state_dict(), self.best_model_path)
        torch.save(self.second_best.state_dict(), self.second_best_model_path)
        info_logger.info("Best and Second Best models saved.")

    def crossover(self):
        """
        The function performs crossover and mutation operations on a population of individuals.
        """
        info_logger.info("Crossover started.")
        new_population = [self.best]

        for _ in range(self.population_size - 1):
            child = self.best.crossover(self.second_best)
            new_population.append(child)

        self.population = new_population
        info_logger.info("Crossover finished.")

    def mutation(self):
        """
        The function performs mutation on the parameters of each individual in the population by adding
        random noise.
        """
        for individual in self.population:
            for param in individual.parameters():
                if param.requires_grad:
                    param.data += torch.randn_like(param.data) * self.mutation_step

    def next_generation(self):
        """
        The `next_generation` function trains and evaluates a population of individuals using a genetic
        algorithm to evolve a model over multiple generations.
        """
        info_logger.info("Next Generation started.")
        while not self.should_stop():
            info_logger.info(f"Generation : {self.generation}")
            training_set, validation_set = get_image_datasets()
            for individual in self.population:
                if training_set is None or validation_set is None:
                    info_logger.info(
                        "Training and Validation sets are None. Generating new sets."
                    )
                    training_set, validation_set = get_image_datasets()
                metrics = individual.train_model(
                    train_loader=training_set,
                    val_loader=validation_set,
                    learning_rate=config["learning_Rate"],
                    n_epochs=config["number_Epochs"],
                    model_config=config,
                )
                individual.fitness = metrics["accuracies"][-1]
                debug_logger.debug("Individual fitness: %s", individual.fitness)

            info_logger.info("Population sorted.")
            self.evaluate()
            self.selection()
            self.crossover()
            self.mutation()
            info_logger.info("Next Generation finished.")
            self.generation += 1

    def update_best_individuals(self):
        """
        The function updates the best individuals in a population based on their fitness values.
        """
        info_logger.info("Updating best individuals.")
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        self.best = self.population[0]
        self.second_best = self.population[1]
        self.third_best = self.population[2]
        debug_logger.debug(
            "Best: %s, Second Best: %s, Third Best: %s",
            self.best.fitness,
            self.second_best.fitness,
            self.third_best.fitness,
        )
        info_logger.info("Best individuals updated.")

    def should_stop(self):
        """
        The function checks if any of the stopping conditions for a genetic algorithm are met.
        :return: The function should_stop() returns a boolean value. It returns True if either the
        maximum number of generations has been reached or the best score has reached the cutoff score.
        Otherwise, it returns False.
        """
        debug_logger.debug("Checking stopping conditions.")  # Debug line
        if self.generation >= config["max_Generations"]:
            debug_logger.debug("Max generations reached.")  # Debug line
            return True
        if self.best is not None and self.best.custom_index >= config["cutoff_Score"]:
            debug_logger.debug("Cutoff score reached.")  # Debug line
            return True
        debug_logger.debug("**-> No stopping condition met.")  # Debug line
        return False

    def run(self):
        """
        The function runs a genetic algorithm for training a model and logs information and debug
        messages.
        :return: True if the stopping condition is met, indicating that the training should stop.
        Otherwise, it does not explicitly return anything.
        """
        info_logger.info("Running...")
        debug_logger.debug(
            "Initial should_stop value: %s", self.should_stop()
        )  # Debug line
        self.initialize_population()
        debug_logger.debug("Starting generation %s", self.generation)  # Debug line
        self.next_generation()
        debug_logger.debug("Finished generation %s", self.generation)  # Debug line
        if (
            self.best is not None
            and self.second_best is not None
            and self.third_best is not None
        ):
            info_logger.info("Don't $top!")
            debug_logger.debug(
                "Generation: %s, Best: %s, Second Best: %s, Third Best: %s",
                self.generation,
                self.best.fitness,
                self.second_best.fitness,
                self.third_best.fitness,
            )
            if self.generation % 25 == 0:
                info_logger.info(
                    "Generation: %s, Best: %s, Second Best: %s, Third Best: %s",
                    self.generation,
                    self.best.fitness,
                    self.second_best.fitness,
                    self.third_best.fitness,
                )
            if self.generation % 100 == 0:
                info_logger.info(
                    "Generation: %s, Best: %s, Second Best: %s, Third Best: %s",
                    self.generation,
                    self.best.fitness,
                    self.second_best.fitness,
                    self.third_best.fitness,
                )
        info_logger.info("Updating best individuals.")
        info_logger.info(
            "Best: %s, Second Best: %s, Third Best: %s",
            self.best.fitness,
            self.second_best.fitness,
            self.third_best.fitness,
        )
        if self.should_stop():
            info_logger.info("Stopping condition met.")
            info_logger.info("Training stopped based on the stopping condition.")
            return True

    info_logger.info("Training stopped based on the stopping condition.")

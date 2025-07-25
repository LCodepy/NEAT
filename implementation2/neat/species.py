import random

from implementation2.neat.neatconfig import NEATConfig
from implementation2.neat.individual import Individual


class Species:

    def __init__(self, id_, individuals: list[Individual], config: NEATConfig):
        self.id = id_
        self.individuals = individuals
        self.config = config

        self.avg_fitness = 0
        self.fitness_sum = 0
        self.generations_since_improved = 0
        self.allowed_offspring = 0
        self.max_fitness = 0

        self.offspring = []

    def calculate_adjusted_fitness(self) -> None:
        for individual in self.individuals:
            individual.fitness /= self.get_size()

    def calculate_averages(self) -> None:
        self.fitness_sum = sum(map(lambda i: i.fitness, self.individuals))
        self.avg_fitness = self.fitness_sum / self.get_size()

    def calculate_allowed_offspring(self, global_average_fitness: float, max_gen_since_improvement: int) -> None:
        if global_average_fitness == 0:
            self.allowed_offspring = 0
            return

        self.allowed_offspring = round(self.avg_fitness / global_average_fitness * self.get_size())
        if self.generations_since_improved > max_gen_since_improvement:
            self.allowed_offspring = 0

    def calculate_generation_since_improved(self) -> None:
        max_fitness = max(map(lambda i: i.fitness, self.individuals))
        if self.max_fitness >= max_fitness - self.config.species_improvement_threshold:
            self.generations_since_improved += 1
        else:
            self.generations_since_improved = 0
        self.max_fitness = max(map(lambda i: i.fitness, self.individuals))

    def add(self, individual: Individual) -> None:
        self.individuals.append(individual)

    def roulette_wheel_selection(self, survival_threshold: float, exclude: Individual = None) -> Individual:
        survivals = sorted(self.individuals, key=lambda i: i.fitness)[int((1 - survival_threshold) * self.get_size()):]

        if exclude in survivals:
            survivals.remove(exclude)

        if not survivals:
            return exclude

        value = random.random() * sum(map(lambda i: i.fitness, survivals))
        fitness_roulette = [survivals[0].fitness]
        for i in range(1, len(survivals)):
            fitness_roulette.append(fitness_roulette[i - 1] + survivals[i].fitness)

        for i in range(len(survivals)):
            if fitness_roulette[i] > value:
                return survivals[i]
        return survivals[-1]

    def get_size(self) -> int:
        return len(self.individuals)

    def __repr__(self) -> str:
        return f"Species({self.individuals})"

import random

from neat.individual import Individual


class Species:

    def __init__(self, id_, individuals: list[Individual]):
        self.id = id_
        self.individuals = individuals
        self.avg_fitness = 0
        self.fitness_sum = 0
        self.generations_since_improved = 0
        self.allowed_offspring = 0
        self.max_fitness = 0
        self.improvement_threshold = 0.01
        self.offspring = []

    def calculate_adjusted_fitness(self):
        for individual in self.individuals:
            individual.fitness /= self.get_size()

    def calculate_averages(self):
        self.fitness_sum = sum(map(lambda i: i.fitness, self.individuals))
        self.avg_fitness = self.fitness_sum / self.get_size()

    def calculate_allowed_offspring(self, global_average_fitness, max_gen_since_improvement):
        self.allowed_offspring = round(self.avg_fitness / global_average_fitness * self.get_size())
        if self.generations_since_improved > max_gen_since_improvement:
            self.allowed_offspring = 0

    def calculate_generation_since_improved(self):
        max_fitness = max(map(lambda i: i.fitness, self.individuals))
        if self.max_fitness >= max_fitness - self.improvement_threshold:
            self.generations_since_improved += 1
        else:
            self.generations_since_improved = 0
        self.max_fitness = max(map(lambda i: i.fitness, self.individuals))

    def add(self, individual: Individual):
        self.individuals.append(individual)

    def roulette_wheel_selection(self, survival_threshold: float, exclude: Individual = None):
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

    def get_size(self):
        return len(self.individuals)

    def __repr__(self):
        return f"Species({self.individuals})"

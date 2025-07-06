import random
from typing import Optional, Generator, Callable

from implementation2.neat.neatconfig import NEATConfig
from implementation2.neat.connection_factory import ConnectionFactory
from implementation2.neat.genome import Genome
from implementation2.neat.genome_factory import GenomeFactory
from implementation2.neat.individual import Individual
from implementation2.neat.neural_network import NeuralNetwork
from implementation2.neat.node_factory import NodeFactory
from implementation2.neat.species import Species


class Population:

    def __init__(self, size: int, num_inputs: int, num_outputs: int, generations: int, config: NEATConfig,
                 evaluation_function: Callable[[NeuralNetwork], float], desired_fitness: float = None):
        self.size = size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.generations = generations
        self.config = config
        self.evaluation_function = evaluation_function
        self.desired_fitness = desired_fitness

        self.node_factory = NodeFactory(num_inputs + num_outputs)
        self.connection_factory = ConnectionFactory(num_inputs * num_outputs)
        self.genome_factory = GenomeFactory(self.connection_factory, self.node_factory, self.config)

        self.genomes = [self.genome_factory.create_genome(num_inputs, num_outputs) for _ in range(self.size)]
        self.species: list[Species] = []
        self.current_species = 0

        self.compatibility_threshold = self.config.compatibility_threshold

    def run(self) -> Generator[int, None, None]:
        generation = 0
        while generation < self.generations:
            self.speciate_genomes()
            self.evaluate()

            if self.desired_fitness and self.get_sorted_individuals()[-1].fitness >= self.desired_fitness:
                return

            yield generation

            if generation < self.generations - 1:
                self.crossover_genomes()
                self.mutate_genomes()

            generation += 1

    def speciate_genomes(self) -> None:
        genomes = [genome for genome in self.genomes]

        if self.species:
            representatives = []

            for species in self.species:
                if not species.offspring:
                    continue
                representative = random.choice(species.individuals)
                species.individuals.clear()
                representatives.append((species, representative.genome))

            for genome in genomes:
                for representative in representatives:
                    if self.calculate_genome_distance(genome, representative[1]) < self.compatibility_threshold:
                        representative[0].add(Individual(genome, 0))
                        break
                else:
                    self.species.append(Species(self.current_species, [Individual(genome, 0)], self.config))
                    representatives.append((self.species[-1], genome))
                    self.current_species += 1
        else:
            idx = random.randint(0, len(genomes) - 1)
            representative = genomes.pop(idx)
            self.species.append(Species(0, [Individual(representative, 0)], self.config))
            self.current_species += 1

            for genome in genomes:
                for species in self.species:
                    representative = random.choice(species.individuals).genome
                    if self.calculate_genome_distance(genome, representative) < self.compatibility_threshold:
                        species.add(Individual(genome, 0))
                        break
                else:
                    self.species.append(Species(self.current_species, [Individual(genome, 0)], self.config))
                    self.current_species += 1

        to_remove = []
        for species in self.species:
            if not species.individuals:
                to_remove.append(species)

        for species in to_remove:
            self.species.remove(species)

        if len(self.species) < self.config.species_target_size:
            self.compatibility_threshold -= self.config.species_target_step_size
        elif len(self.species) > self.config.species_target_size:
            self.compatibility_threshold += self.config.species_target_step_size

    def evaluate(self) -> None:
        for species in self.species:
            for individual in species.individuals:
                # individual.genome.show_innovation_history()
                individual.fitness = self.evaluation_function(NeuralNetwork(individual.genome))

    def crossover_genomes(self) -> None:
        self.apply_explicit_fitness_sharing()
        self.calculate_allowed_offspring()

        genomes: list[Optional[Genome]] = [None for _ in range(self.size)]

        current_offspring = 0
        for species in self.species:
            species.offspring.clear()
            for i in range(species.allowed_offspring):
                if current_offspring >= self.size:
                    break

                parent1 = species.roulette_wheel_selection(self.config.survival_threshold)
                parent2 = species.roulette_wheel_selection(self.config.survival_threshold, exclude=parent1)

                if parent1.fitness > parent2.fitness:
                    offspring = parent1.crossover(self.genome_factory, parent2)
                else:
                    offspring = parent2.crossover(self.genome_factory, parent1)

                species.offspring.append(Individual(offspring, 0))
                genomes[current_offspring] = offspring

                current_offspring += 1

        for i in range(self.size):
            if not genomes[i]:
                genomes[i] = self.genome_factory.create_genome(self.num_inputs, self.num_outputs)

        to_remove = []
        for species in self.species:
            if species.allowed_offspring == 0:
                to_remove.append(species)

        for species in to_remove:
            self.species.remove(species)

        self.genomes = genomes

    def mutate_genomes(self) -> None:
        for species in self.species:
            for individual in species.offspring:
                genome = individual.genome

                if random.random() > 1 - self.config.weight_mutation_chance:
                    if random.random() > 1 - self.config.change_weight_mutation_chance:
                        genome.mutate_change_weight()
                    else:
                        genome.mutate_assign_new_weight()
                if random.random() > 1 - self.config.bias_mutation_chance:
                    if random.random() > 1 - self.config.change_bias_mutation_chance:
                        genome.mutate_change_bias()
                    else:
                        genome.mutate_assign_new_bias()
                if random.random() > 1 - self.config.add_connection_mutation_chance:
                    for _ in range(20):
                        if genome.mutate_add_connection():
                            break
                if random.random() > 1 - self.config.add_node_mutation_chance:
                    genome.mutate_add_node()
                if random.random() > 1 - self.config.enable_mutation_chance:
                    genome.mutate_change_enabled()
                if random.random() > 1 - self.config.remove_node_mutation_chance:
                    genome.mutate_remove_node()
                if random.random() > 1 - self.config.remove_connection_mutation_chance:
                    genome.mutate_remove_connection()

    def apply_explicit_fitness_sharing(self) -> None:
        for species in self.species:
            species.calculate_adjusted_fitness()
            species.calculate_averages()

    def calculate_allowed_offspring(self) -> None:
        for species in self.species:
            species.calculate_generation_since_improved()

        global_avg_fitness = 0
        genomes_num = 0
        for species in self.species:
            if species.generations_since_improved < self.config.max_allowed_generations_since_improved:
                genomes_num += species.get_size()
                global_avg_fitness += species.fitness_sum

        if genomes_num:
            global_avg_fitness /= genomes_num

        for species in self.species:
            species.calculate_allowed_offspring(global_avg_fitness, self.config.max_allowed_generations_since_improved)

    def calculate_genome_distance(self, genome1: Genome, genome2: Genome) -> float:
        excess_genes = 0
        disjoint_genes = 0

        innovations1 = []
        innovations2 = []

        average_weight_distance = 0
        matching_genes_count = 0

        for connection in genome1.connection_genes.values():
            innovations1.append(connection.innovation_number)

        for connection in genome2.connection_genes.values():
            innovations2.append(connection.innovation_number)

        for innov in innovations1:
            if innov in innovations2:
                average_weight_distance += abs(
                    genome1.get_connection(innov).weight - genome2.get_connection(innov).weight)
                matching_genes_count += 1
            else:
                if innovations2 and innov > max(innovations2):
                    excess_genes += 1
                else:
                    disjoint_genes += 1

        for innov in innovations2:
            if innov in innovations1:
                average_weight_distance += abs(
                    genome1.get_connection(innov).weight - genome2.get_connection(innov).weight)
                matching_genes_count += 1
            else:
                if innovations1 and innov > max(innovations1):
                    excess_genes += 1
                else:
                    disjoint_genes += 1

        if matching_genes_count == 0:
            matching_genes_count = 1

        larger_genome_size = max(len(innovations1), len(innovations2))
        average_weight_distance /= matching_genes_count

        if larger_genome_size == 0:
            larger_genome_size = 1

        genome_delta = (
                self.config.excess_genes_importance * excess_genes / larger_genome_size +
                self.config.disjoint_genes_importance * disjoint_genes / larger_genome_size +
                self.config.weight_difference_importance * average_weight_distance
        )

        # !!!TEST!!!

        # innov1 = []
        # for node in genome1.node_genes.values():
        #     innov1.append(node.id)
        #
        # innov2 = []
        # for node in genome2.node_genes.values():
        #     innov2.append(node.id)
        #
        # same = list(set(innov1).intersection(set(innov2)))
        #
        # bias_distance = 0
        #
        # for node in same:
        #     bias_distance += abs(genome1.get_node(node).bias - genome2.get_node(node).bias)

        # genome_delta += 0.4 * bias_distance / len(same)

        # !!!TEST!!!

        return genome_delta

    def get_sorted_individuals(self) -> list[Individual]:
        return sorted([y for s in self.species for y in s.individuals], key=lambda x: x.fitness)

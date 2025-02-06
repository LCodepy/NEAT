import random

from neat.connection_factory import ConnectionFactory
from neat.genome import Genome
from neat.genome_factory import GenomeFactory
from neat.individual import Individual
from neat.neural_network import NeuralNetwork
from neat.node_factory import NodeFactory
from neat.species import Species


class Population:

    def __init__(self, size: int, num_inputs: int, num_outputs: int, generations: int):
        self.size = size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.generations = generations

        self.node_factory = NodeFactory(num_inputs + num_outputs)
        self.connection_factory = ConnectionFactory(num_inputs * num_outputs - 0)
        self.genome_factory = GenomeFactory(self.connection_factory, self.node_factory)

        self.genomes = [self.genome_factory.create_genome(num_inputs, num_outputs) for _ in range(self.size)]
        self.species: list[Species] = []
        self.current_species = 0

        self.species_size_target = 5
        self.species_target_step_size = 0.1

        self.excess_genes_importance = 1.0
        self.disjoint_genes_importance = 1.0
        self.weight_difference_importance = 0.4

        self.compatibility_threshold = 1

        self.survival_threshold = 0.2

        self.max_allowed_generations_since_improved = 20

    def run(self):
        generation = 0
        while True:
            print(f"\n----------------------GENERATION {generation}--------------------------")
            self.speciate_genomes()
            self.evaluate()
            if generation == self.generations-1 or sorted([y for s in self.species for y in s.individuals], key=lambda x: x.fitness)[-1].fitness > 3.9:
                return sorted([y for s in self.species for y in s.individuals], key=lambda x: x.fitness)
            print("Best: ", sorted([y for s in self.species for y in s.individuals], key=lambda x: x.fitness)[-1])
            self.crossover_genomes()
            self.mutate_genomes()

            generation += 1

    def speciate_genomes(self):
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
                    self.species.append(Species(self.current_species, [Individual(genome, 0)]))
                    representatives.append((self.species[-1], genome))
                    self.current_species += 1
        else:
            idx = random.randint(0, len(genomes) - 1)
            representative = genomes.pop(idx)
            self.species.append(Species(0, [Individual(representative, 0)]))
            self.current_species += 1

            for genome in genomes:
                for species in self.species:
                    representative = random.choice(species.individuals).genome
                    if self.calculate_genome_distance(genome, representative) < self.compatibility_threshold:
                        species.add(Individual(genome, 0))
                        break
                else:
                    self.species.append(Species(self.current_species, [Individual(genome, 0)]))
                    self.current_species += 1

        to_remove = []
        for species in self.species:
            if not species.individuals:
                to_remove.append(species)

        for species in to_remove:
            self.species.remove(species)

        if len(self.species) < self.species_size_target:
            self.compatibility_threshold -= self.species_target_step_size
        elif len(self.species) > self.species_size_target:
            self.compatibility_threshold += self.species_target_step_size

    def evaluate(self):
        for species in self.species:
            for individual in species.individuals:
                nn = NeuralNetwork(individual.genome)

                out1 = nn.forward([0, 0])[0]
                out2 = nn.forward([0, 1])[0]
                out3 = nn.forward([1, 0])[0]
                out4 = nn.forward([1, 1])[0]

                fitness = 1-out1 + out2 + out3 + 1-out4
                individual.fitness = fitness

    def crossover_genomes(self):
        self.apply_explicit_fitness_sharing()
        self.calculate_allowed_offspring()

        genomes = [None for _ in range(self.size)]

        current_offspring = 0
        for species in self.species:
            species.offspring.clear()
            for i in range(species.allowed_offspring):
                if current_offspring >= self.size:
                    break

                parent1 = species.roulette_wheel_selection(self.survival_threshold)
                parent2 = species.roulette_wheel_selection(self.survival_threshold, exclude=parent1)

                if parent1.fitness > parent2.fitness:
                    offspring = parent1.crossover(self.genome_factory, parent2)
                else:
                    offspring = parent2.crossover(self.genome_factory, parent1)

                species.offspring.append(Individual(offspring, 0))
                genomes[current_offspring] = offspring

                current_offspring += 1

        to_remove = []
        for species in self.species:
            if species.allowed_offspring == 0:
                to_remove.append(species)

        for species in to_remove:
            self.species.remove(species)

        self.genomes = genomes

    def mutate_genomes(self):
        for species in self.species:
            for individual in species.offspring:
                genome = individual.genome

                if random.random() > 0.2:
                    if random.random() > 0.1:
                        genome.mutate_change_weight()
                    else:
                        genome.mutate_assign_new_weight()
                if random.random() > 0.2:
                    if random.random() > 0.1:
                        genome.mutate_change_bias()
                    else:
                        genome.mutate_assign_new_bias()
                if random.random() > 0.8:
                    for _ in range(20):
                        if genome.mutate_add_connection():
                            break
                if random.random() > 0.9:
                    genome.mutate_add_node()
                if random.random() > 0.9:
                    genome.mutate_change_enabled()
                if random.random() > 0.9:
                    genome.mutate_remove_node()
                if random.random() > 0.8:
                    genome.mutate_remove_connection()

    def apply_explicit_fitness_sharing(self):
        for species in self.species:
            species.calculate_adjusted_fitness()
            species.calculate_averages()

    def calculate_allowed_offspring(self):
        for species in self.species:
            species.calculate_generation_since_improved()

        global_avg_fitness = 0
        genomes_num = 0
        for species in self.species:
            if species.generations_since_improved < self.max_allowed_generations_since_improved:
                genomes_num += species.get_size()
                global_avg_fitness += species.fitness_sum

        global_avg_fitness /= genomes_num

        for species in self.species:
            species.calculate_allowed_offspring(global_avg_fitness, self.max_allowed_generations_since_improved)

    def calculate_genome_distance(self, genome1: Genome, genome2: Genome):
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

        genome_delta = self.excess_genes_importance * excess_genes / larger_genome_size + \
                       self.disjoint_genes_importance * disjoint_genes / larger_genome_size + \
                       self.weight_difference_importance * average_weight_distance

        # !!!TEST!!!

        innov1 = []
        for node in genome1.node_genes.values():
            innov1.append(node.id)

        innov2 = []
        for node in genome2.node_genes.values():
            innov2.append(node.id)

        same = list(set(innov1).intersection(set(innov2)))

        bias_distance = 0

        for node in same:
            bias_distance += abs(genome1.get_node(node).bias - genome2.get_node(node).bias)

        # genome_delta += 0.4 * bias_distance / len(same)

        # !!!TEST!!!

        return genome_delta

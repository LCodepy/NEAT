from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import pygame

import numpy as np

from events import EventHandler
from label import Label


def sigmoid(x: float, k: float = 10, l: float = 0) -> float:
    try:
        return 1 / (1 + math.exp(-x * k + l))
    except OverflowError:
        return 0


def random_normal_distribution(mean: float, standard_deviation: float):
    return np.random.normal(mean, standard_deviation)


class NodeType(Enum):
    INPUT_NODE = auto()
    OUTPUT_NODE = auto()
    HIDDEN_NODE = auto()


@dataclass
class NodeGene:
    id: int
    type: NodeType
    bias: float
    activation: Callable

    def copy(self) -> NodeGene:
        return NodeGene(self.id, self.type, self.bias, self.activation)


@dataclass
class ConnectionGene:
    input_node: int
    output_node: int
    weight: float
    enabled: bool
    innovation_number: int

    def copy(self) -> ConnectionGene:
        return ConnectionGene(self.input_node, self.output_node, self.weight, self.enabled, self.innovation_number)


class Genome:

    def __init__(self, id_, num_inputs, num_outputs):
        self.id = id_
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.max_bias_weight_value = 10
        self.min_bias_weight_value = -10

        self.node_input_genes = [NodeGene(i + 1, NodeType.INPUT_NODE, 0, sigmoid)
                                 for i in range(self.num_inputs)]

        self.node_output_genes = [
            NodeGene(i + 1 + self.num_inputs, NodeType.OUTPUT_NODE, random.random() * 2 - 1, sigmoid)
            for i in range(self.num_outputs)
        ]

        self.node_genes = self.node_input_genes + self.node_output_genes

        self.connection_genes = [
            ConnectionGene(g_in.id, g_out.id, random.random() * 2 - 1, True, j * self.num_outputs + i)
            for j, g_in in enumerate(self.node_input_genes)
            for i, g_out in enumerate(self.node_output_genes)
        ]

    def mutate_change_weight(self):
        if not self.connection_genes:
            return
        connection = random.choice(self.connection_genes)
        connection.weight = self.clamp(connection.weight + random_normal_distribution(0, 1.2))

    def mutate_assign_new_weight(self):
        if not self.connection_genes:
            return
        connection = random.choice(self.connection_genes)
        connection.weight = self.clamp(random_normal_distribution(0, 1.2))

    def mutate_change_bias(self):
        node = random.choice(self.get_hidden_neurons() + self.get_output_neurons())
        node.bias = self.clamp(node.bias + random_normal_distribution(0, 1.2))

    def mutate_assign_new_bias(self):
        node = random.choice(self.get_hidden_neurons() + self.get_output_neurons())
        node.bias = self.clamp(random_normal_distribution(0, 1.2))

    def mutate_change_enabled(self):
        disabled = [c for c in self.connection_genes if not c.enabled]
        if not disabled:
            return
        connection = random.choice(disabled)

        if not self.creates_cycle(connection.input_node, connection.output_node):
            connection.enabled = True

    def mutate_add_connection(self, connection_factory: ConnectionFactory):
        input_node = random.choice([node.id for node in self.node_genes if node.type is not NodeType.OUTPUT_NODE])
        output_node = random.choice([node.id for node in self.node_genes if node.type is not NodeType.INPUT_NODE])

        # check if it already exists
        for connection in self.connection_genes:
            if connection.input_node == input_node and connection.output_node == output_node:
                return

        if self.creates_cycle(input_node, output_node):
            return

        self.connection_genes.append(
            connection_factory.create_connection(input_node, output_node, random.random() * 2 - 1)
        )

        return True

    def mutate_remove_connection(self):
        if not self.connection_genes:
            return

        self.connection_genes.pop(random.randint(0, len(self.connection_genes) - 1))

    def mutate_add_node(self, node_factory: NodeFactory, connection_factory: ConnectionFactory):
        if not self.connection_genes:
            return

        connection_to_split = random.choice(self.connection_genes)
        if not connection_to_split.enabled:
            return
        connection_to_split.enabled = False

        node = node_factory.create_node(NodeType.HIDDEN_NODE, random.random() * 2 - 1, sigmoid, connection_to_split)
        for connection in self.connection_genes:
            if (
                    (connection.input_node == connection_to_split.input_node and connection.output_node == node.id) or
                    (connection.input_node == node.id and connection.output_node == connection_to_split.output_node)
            ):
                connection_to_split.enabled = True
                return

        if self.creates_cycle(connection_to_split.input_node, node.id) or self.creates_cycle(node.id, connection_to_split.output_node):
            connection_to_split.enabled = True
            return

        self.add_node(node)
        self.add_connection(connection_factory.create_connection(connection_to_split.input_node, node.id, 1.0))
        self.add_connection(connection_factory.create_connection(node.id, connection_to_split.output_node,
                                                                 connection_to_split.weight))
        return True

    def mutate_remove_node(self):
        if not self.get_hidden_neurons():
            return

        node = random.choice(self.get_hidden_neurons())
        self.node_genes.remove(node)

        while True:
            for i, connection in enumerate(self.connection_genes):
                if connection.input_node == node.id or connection.output_node == node.id:
                    self.connection_genes.pop(i)
                    break
            else:
                break

    def creates_cycle(self, input_node, output_node):
        if input_node == output_node:
            return True

        for connection in self.connection_genes:
            if connection.input_node != output_node or not connection.enabled:
                continue
            if connection.output_node == input_node:
                return True

            if self.creates_cycle(input_node, connection.output_node):
                return True
        return False

    def add_node(self, node):
        self.node_genes.append(node)

    def add_connection(self, connection):
        self.connection_genes.append(connection)

    def clamp(self, value):
        return max(self.min_bias_weight_value, min(self.max_bias_weight_value, value))

    def reset(self):
        self.node_input_genes = []
        self.node_output_genes = []
        self.node_genes = []
        self.connection_genes = []

    def get_node(self, node_id):
        for node in self.node_genes:
            if node.id == node_id:
                return node

    def get_connection(self, connection_innov):
        for connection in self.connection_genes:
            if connection.innovation_number == connection_innov:
                return connection

    def get_hidden_neurons(self):
        return list(filter(lambda n: n.type is NodeType.HIDDEN_NODE, self.node_genes))

    def get_input_neurons(self):
        return list(filter(lambda n: n.type is NodeType.INPUT_NODE, self.node_genes))

    def get_output_neurons(self):
        return list(filter(lambda n: n.type is NodeType.OUTPUT_NODE, self.node_genes))

    def show_info(self):
        print("INPUT NODES: ", ", ".join([str(i + 1) for i in range(self.num_inputs)]))
        print("OUTPUT NODES: ", ", ".join([str(i + 1 + self.num_inputs) for i in range(self.num_outputs)]))
        print("CONNECTIONS: ", ", ".join(
            [f"{c.input_node}->{c.output_node} ({c.innovation_number})" + (" (D)" if not c.enabled else "") for c in
             self.connection_genes]))

    def __repr__(self):
        return f"Genome(id={self.id})"

    @staticmethod
    def crossover_nodes(node1: NodeGene, node2: NodeGene):
        bias = random.choice([node1.bias, node2.bias])
        activation = random.choice([node1.activation, node2.activation])

        return NodeGene(node1.id, node1.type, bias, activation)

    @staticmethod
    def crossover_connection(connection1: ConnectionGene, connection2: ConnectionGene):
        weight = random.choice([connection1.weight, connection2.weight])
        enabled = True  # random.choice([connection1.enabled, connection2.enabled])
        if not connection1.enabled or not connection2.enabled:
            enabled = False

        return ConnectionGene(connection1.input_node, connection1.output_node, weight, enabled,
                              connection1.innovation_number)


class Individual:

    def __init__(self, genome, fitness):
        self.genome = genome
        self.fitness = fitness

    def crossover(self, genome_factory: GenomeFactory, individual: Individual) -> Genome:
        offspring = genome_factory.create_genome(self.genome.num_inputs, self.genome.num_outputs)
        offspring.reset()

        for dominant_node in self.genome.node_genes:
            node_id = dominant_node.id
            recessive_node = individual.genome.get_node(node_id)
            if not recessive_node:
                offspring.add_node(dominant_node.copy())
            else:
                offspring.add_node(Genome.crossover_nodes(dominant_node, recessive_node))

        for dominant_connection in self.genome.connection_genes:
            connection_innov = dominant_connection.innovation_number
            recessive_connection = individual.genome.get_connection(connection_innov)
            if not recessive_connection:
                offspring.add_connection(dominant_connection.copy())
            else:
                offspring.add_connection(Genome.crossover_connection(dominant_connection, recessive_connection))

        return offspring

    def __repr__(self):
        return f"Individual(genome={self.genome}, fitness={self.fitness})"


class GenomeFactory:

    def __init__(self):
        self.global_genome_number = 0

    def create_genome(self, num_inputs, num_outputs) -> Genome:
        self.global_genome_number += 1
        return Genome(self.global_genome_number, num_inputs, num_outputs)


class NodeFactory:

    def __init__(self, global_innovation_number):
        self.global_innovation_number = global_innovation_number
        self.generation = 0
        self.generation_mutations = []

    def create_node(self, type_, bias, activation, conn_to_split: ConnectionGene) -> NodeGene:
        for mutation in self.generation_mutations:
            if mutation[0] == conn_to_split.input_node and mutation[1] == conn_to_split.output_node:
                innovation_number = mutation[2]
                break
        else:
            self.global_innovation_number += 1
            innovation_number = self.global_innovation_number

        self.generation_mutations.append(
            (conn_to_split.input_node, conn_to_split.output_node, innovation_number))
        return NodeGene(innovation_number, type_, bias, activation)

    def update_generation(self):
        self.generation += 1
        #self.generation_mutations.append([])


class ConnectionFactory:

    def __init__(self, global_innovation_number):
        self.global_innovation_number = global_innovation_number
        self.generation = 0
        self.generation_mutations = []

    def create_connection(self, input_node, output_node, weight) -> ConnectionGene:  # TODO: Check if works
        for mutation in self.generation_mutations:
            if mutation[0] == input_node and mutation[1] == output_node:
                innovation_number = mutation[2]
                break
        else:
            self.global_innovation_number += 1
            innovation_number = self.global_innovation_number

        self.generation_mutations.append((input_node, output_node, innovation_number))
        return ConnectionGene(input_node, output_node, weight, True, innovation_number)

    def update_generation(self):
        self.generation += 1
        #self.generation_mutations.append([])


@dataclass
class Neuron:
    id: int
    bias: float
    activation: Callable


@dataclass
class Connection:
    input_neuron: int
    output_neuron: int
    weight: float


class NeuralNetwork:

    def __init__(self, genome: Genome):
        self.genome = genome

        self.neuron_inputs = {}
        self.neuron_outputs = {}

        self.neurons = []
        self.hidden_neurons = []
        self.connections = []

        self.create_neural_network()

    def create_neural_network(self):
        nodes = self.genome.node_genes
        connections = self.genome.connection_genes

        self.neuron_inputs = {node.id: [] for node in nodes}
        self.neuron_outputs = {node.id: [] for node in nodes}

        self.hidden_neurons = [Neuron(node.id, node.bias, node.activation) for node in nodes
                               if node.type == NodeType.HIDDEN_NODE]

        self.neurons = [Neuron(node.id, node.bias, node.activation) for node in nodes]
        self.connections = [Connection(conn.input_node, conn.output_node, conn.weight) for conn in connections if
                            conn.enabled]

        for conn in connections:
            self.neuron_inputs[conn.output_node].append(conn.input_node)
            self.neuron_outputs[conn.input_node].append(conn.output_node)

    def forward(self, inputs):
        assert len(inputs) == self.genome.num_inputs, "Number of inputs must be the same"

        activations = {neuron.id: 0 for neuron in self.neurons}

        for i in range(len(inputs)):
            activations[i + 1] = inputs[i]

        for layer in self.calculate_network_layers(self.genome)[1:]:
            for connection in self.connections:
                if connection.output_neuron in layer:
                    activations[connection.output_neuron] += activations[connection.input_neuron] * connection.weight

            for neuron_id in layer:
                neuron = self.get_neuron(neuron_id)
                activations[neuron_id] = neuron.activation(activations[neuron_id] + neuron.bias)

        output = [0 for _ in range(self.genome.num_outputs)]
        for neuron_id, value in activations.items():
            if self.genome.get_node(neuron_id).type == NodeType.OUTPUT_NODE:
                output[neuron_id - self.genome.num_inputs - 1] = value

        return output

    def get_neuron(self, id_: int) -> Neuron:
        for neuron in self.neurons:
            if neuron.id == id_:
                return neuron

    @staticmethod
    def calculate_network_layers(genome):
        connections = genome.connection_genes
        connection_matrix = {node.id: [] for node in genome.node_genes}
        hidden_nodes = [node.id for node in genome.node_genes if node.type is NodeType.HIDDEN_NODE]

        # for every hidden node, create a list of nodes that connect to it
        for connection in connections:
            if connection.input_node in hidden_nodes and connection.output_node in hidden_nodes and connection.enabled:
                connection_matrix[connection.output_node].append(connection.input_node)

        nodes_found = []
        layers = []
        while True:
            new_nodes_found = []
            last_nodes = []

            # search for nodes that have no hidden nodes connected to them
            for node in connection_matrix.keys():
                if not connection_matrix[node] and node in hidden_nodes and node not in nodes_found:
                    nodes_found.append(node)
                    new_nodes_found.append(node)

            # remove found nodes from the connection matrix
            for node in connection_matrix.keys():
                if node in hidden_nodes:
                    if connection_matrix[node]:
                        last_nodes.append(node)
                    connection_matrix[node] = list(set(connection_matrix[node]).difference(set(nodes_found)))

            layers.append(new_nodes_found)

            if not any(list(connection_matrix.values())):
                layers.append(last_nodes)
                break

        layers = [list(map(lambda n: n.id, genome.get_input_neurons()))] + layers
        layers += [list(map(lambda n: n.id, genome.get_output_neurons()))]
        return layers


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
        return survivals[-1]

    def get_size(self):
        return len(self.individuals)

    def __repr__(self):
        return f"Species({self.individuals})"


class Population:

    def __init__(self, size: int, num_inputs: int, num_outputs: int, generations: int):
        self.size = size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.generations = generations

        self.genome_factory = GenomeFactory()
        self.node_factory = NodeFactory(num_inputs + num_outputs)
        self.connection_factory = ConnectionFactory(num_inputs * num_outputs - 0)

        self.genomes = [self.genome_factory.create_genome(num_inputs, num_outputs) for _ in range(self.size)]
        self.species: list[Species] = []
        self.current_species = 0

        self.species_size_target = 8
        self.species_target_step_size = 0.1

        self.excess_genes_importance = 1.0
        self.disjoint_genes_importance = 1.0
        self.weight_difference_importance = 0.4

        self.compatibility_threshold = 3

        self.survival_threshold = 0.2

        self.max_allowed_generations_since_improved = 20

    def run(self):
        generation = 0
        while True:
            #for i in range(self.generations):
            # os.system("cls")
            # print(f"----------------------GENERATION {generation}--------------------------")
            self.speciate_genomes()
            self.evaluate()
            if generation == self.generations-1 or sorted([y for s in self.species for y in s.individuals], key=lambda x: x.fitness)[-1].fitness > 3.8:
                pass  # return sorted([y for s in self.species for y in s.individuals], key=lambda x: x.fitness)
            yield self.species
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
                # if out1 < 0.3:
                #     out1 = 0
                # if out2 > 0.7:
                #     out2 = 1
                # if out3 > 0.7:
                #     out3 = 1
                # if out4 < 0.3:
                #     out4 = 0
                fitness = (1-out1) + out2 + out3 + (1-out4)
                # fitness = 1 - (abs(0 - out1) + abs(1 - out2) + abs(1 - out3) + abs(0 - out4)) / 4
                individual.fitness = fitness

    def crossover_genomes(self):
        self.apply_explicit_fitness_sharing()
        self.calculate_allowed_offspring()

        genomes = [self.genome_factory.create_genome(self.num_inputs, self.num_outputs) for _ in range(self.size)]

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
                if random.random() > 0.6:
                    for _ in range(20):
                        if genome.mutate_add_connection(self.connection_factory):
                            break
                if random.random() > 0.92:
                    genome.mutate_add_node(self.node_factory, self.connection_factory)
                if random.random() > 0.75:
                    genome.mutate_change_enabled()
                if random.random() > 0.90:
                    genome.mutate_remove_node()
                if random.random() > 0.92:
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

        for connection in genome1.connection_genes:
            innovations1.append(connection.innovation_number)

        for connection in genome2.connection_genes:
            innovations2.append(connection.innovation_number)

        for innov in innovations1:
            if innov in innovations2:
                average_weight_distance += abs(
                    genome1.get_connection(innov).weight - genome2.get_connection(innov).weight
                )
                matching_genes_count += 1
            else:
                if innovations2 and innov > max(innovations2):
                    excess_genes += 1
                else:
                    disjoint_genes += 1

        for innov in innovations2:
            if innov in innovations1:
                average_weight_distance += abs(
                    genome1.get_connection(innov).weight - genome2.get_connection(innov).weight
                )
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

        # # !!!TEST!!!
        #
        # innov1 = []
        # for node in genome1.node_genes:
        #     innov1.append(node.id)
        #
        # innov2 = []
        # for node in genome2.node_genes:
        #     innov2.append(node.id)
        #
        # same = list(set(innov1).intersection(set(innov2)))
        #
        # bias_distance = 0
        #
        # for node in same:
        #     bias_distance += abs(genome1.get_node(node).bias - genome2.get_node(node).bias)
        #
        # # genome_delta += 0.4 * bias_distance / len(same)
        #
        # # !!!TEST!!!

        return genome_delta

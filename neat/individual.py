from __future__ import annotations

import random

from neat.connection import ConnectionGene
from neat.genome import Genome
from neat.genome_factory import GenomeFactory
from neat.node import NodeGene


class Individual:

    def __init__(self, genome, fitness):
        self.genome = genome
        self.fitness = fitness

    def crossover(self, genome_factory: GenomeFactory, individual: Individual) -> Genome:
        offspring = genome_factory.create_genome(self.genome.num_inputs, self.genome.num_outputs)
        offspring.reset()

        for dominant_node in self.genome.node_genes.values():
            node_id = dominant_node.id
            recessive_node = individual.genome.get_node(node_id)
            if not recessive_node:
                offspring.add_node(dominant_node.copy())
            else:
                offspring.add_node(self.crossover_nodes(dominant_node, recessive_node))

        for dominant_connection in self.genome.connection_genes.values():
            connection_innov = dominant_connection.innovation_number
            recessive_connection = individual.genome.get_connection(connection_innov)
            if not recessive_connection:
                offspring.add_connection(dominant_connection.copy())
            else:
                offspring.add_connection(self.crossover_connection(dominant_connection, recessive_connection))

        return offspring


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

    def __repr__(self):
        return f"Individual(genome={self.genome}, fitness={self.fitness})"

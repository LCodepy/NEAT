import random

import numpy as np

from neat.activation_functions import sigmoid
from neat.connection import ConnectionGene
from neat.connection_factory import ConnectionFactory
from neat.node import NodeGene, NodeType
from neat.node_factory import NodeFactory


def random_normal_distribution(mean: float, standard_deviation: float):
    return np.random.normal(mean, standard_deviation)


class Genome:

    def __init__(self, id_, num_inputs, num_outputs, connection_factory: ConnectionFactory, node_factory: NodeFactory):
        self.id = id_
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.connection_factory = connection_factory
        self.node_factory = node_factory

        self.max_bias_weight_value = 20
        self.min_bias_weight_value = -20

        self.node_input_genes = [NodeGene(i + 1, NodeType.INPUT_NODE, 0, sigmoid)
                                 for i in range(self.num_inputs)]

        self.node_output_genes = [
            NodeGene(i + 1 + self.num_inputs, NodeType.OUTPUT_NODE, random.random() * 2 - 1, sigmoid)
            for i in range(self.num_outputs)
        ]

        self.node_genes = {node.id: node for node in self.node_input_genes + self.node_output_genes}

        self.connection_genes = {
            j * self.num_outputs + i: ConnectionGene(g_in.id, g_out.id, random.random() * 2 - 1, True, j * self.num_outputs + i)
            for j, g_in in enumerate(self.node_input_genes)
            for i, g_out in enumerate(self.node_output_genes)
        }

    def mutate_change_weight(self):
        if not self.connection_genes:
            return
        connection = random.choice(list(self.connection_genes.values()))
        connection.weight = self.clamp(connection.weight + random_normal_distribution(0, 1.2))

    def mutate_assign_new_weight(self):
        if not self.connection_genes:
            return
        connection = random.choice(list(self.connection_genes.values()))
        connection.weight = self.clamp(random_normal_distribution(0, 1))

    def mutate_change_bias(self):
        node = random.choice(self.get_hidden_neurons() + self.get_output_neurons())
        node.bias = self.clamp(node.bias + random_normal_distribution(0, 1.2))

    def mutate_assign_new_bias(self):
        node = random.choice(self.get_hidden_neurons() + self.get_output_neurons())
        node.bias = self.clamp(random_normal_distribution(0, 1))

    def mutate_change_enabled(self):
        disabled = [c for c in self.connection_genes.values() if not c.enabled]
        if not disabled:
            return
        connection = random.choice(disabled)

        if not self.creates_cycle(connection.input_node, connection.output_node):
            connection.enabled = True

    def mutate_add_connection(self):
        input_node = random.choice([node.id for node in self.node_genes.values() if node.type is not NodeType.OUTPUT_NODE])
        output_node = random.choice([node.id for node in self.node_genes.values() if node.type is not NodeType.INPUT_NODE])

        # check if it already exists
        for connection in self.connection_genes.values():
            if connection.input_node == input_node and connection.output_node == output_node:
                return

        if self.creates_cycle(input_node, output_node):
            return

        self.add_connection(
            self.connection_factory.create_connection(input_node, output_node, random.random() * 2 - 1)
        )

        return True

    def mutate_remove_connection(self):
        if not self.connection_genes:
            return

        self.remove_connection(random.choice(list(self.connection_genes.keys())))

    def mutate_add_node(self):
        if not self.connection_genes:
            return

        connection_to_split = random.choice(list(self.connection_genes.values()))
        if not connection_to_split.enabled:
            return
        connection_to_split.enabled = False

        node = self.node_factory.create_node(NodeType.HIDDEN_NODE, random.random() * 2 - 1, sigmoid, connection_to_split)
        for connection in self.connection_genes.values():
            if (
                    (connection.input_node == connection_to_split.input_node and connection.output_node == node.id) or
                    (connection.input_node == node.id and connection.output_node == connection_to_split.output_node)
            ):
                return

        if self.creates_cycle(connection_to_split.input_node, node.id) or self.creates_cycle(node.id, connection_to_split.output_node):
            return

        self.add_node(node)
        self.add_connection(self.connection_factory.create_connection(connection_to_split.input_node, node.id, 1.0))
        self.add_connection(self.connection_factory.create_connection(node.id, connection_to_split.output_node,
                                                                 connection_to_split.weight))
        return True

    def mutate_remove_node(self):
        if not self.get_hidden_neurons():
            return

        node = random.choice(self.get_hidden_neurons())
        self.remove_node(node)

        while True:
            for i, connection in self.connection_genes.items():
                if connection.input_node == node.id or connection.output_node == node.id:
                    self.connection_genes.pop(i)
                    break
            else:
                break

    def creates_cycle(self, input_node, output_node):
        if input_node == output_node:
            return True

        for connection in self.connection_genes.values():
            if connection.input_node != output_node or not connection.enabled:
                continue
            if connection.output_node == input_node:
                return True

            if self.creates_cycle(input_node, connection.output_node):
                return True
        return False

    def add_node(self, node):
        self.node_genes[node.id] = node

    def remove_node(self, node):
        self.node_genes.pop(node.id)

    def add_connection(self, connection):
        self.connection_genes[connection.innovation_number] = connection

    def remove_connection(self, innov):
        self.connection_genes.pop(innov)

    def clamp(self, value):
        return max(self.min_bias_weight_value, min(self.max_bias_weight_value, value))

    def reset(self):
        self.node_input_genes = []
        self.node_output_genes = []
        self.node_genes = {}
        self.connection_genes = {}

    def get_node(self, node_id):
        if node_id in self.node_genes:
            return self.node_genes[node_id]

    def get_connection(self, connection_innov):
        if connection_innov in self.connection_genes:
            return self.connection_genes[connection_innov]

    def get_hidden_neurons(self):
        return list(filter(lambda n: n.type is NodeType.HIDDEN_NODE, list(self.node_genes.values())))

    def get_input_neurons(self):
        return list(filter(lambda n: n.type is NodeType.INPUT_NODE, list(self.node_genes.values())))

    def get_output_neurons(self):
        return list(filter(lambda n: n.type is NodeType.OUTPUT_NODE, list(self.node_genes.values())))

    def show_info(self):
        print("INPUT NODES: ", ", ".join([str(i + 1) for i in range(self.num_inputs)]))
        print("OUTPUT NODES: ", ", ".join([str(i + 1 + self.num_inputs) for i in range(self.num_outputs)]))
        print("CONNECTIONS: ", ", ".join(
            [f"{c.input_node}->{c.output_node} ({c.innovation_number})" + (" (D)" if not c.enabled else "") for c in
             self.connection_genes.values()]))

    def __repr__(self):
        return f"Genome(id={self.id})"
import random

import numpy as np

from neat.activation_functions import sigmoid
from neat.config import Config
from neat.connection import ConnectionGene
from neat.connection_factory import ConnectionFactory
from neat.node import NodeGene, NodeType
from neat.node_factory import NodeFactory


def random_normal_distribution(mean: float, standard_deviation: float):
    return np.random.normal(mean, standard_deviation)


def random_number_in_bound(low: float = -1, high: float = 1):
    return random.random() * (high - low) + low


class Genome:

    def __init__(self, id_: int, num_inputs: int, num_outputs: int, connection_factory: ConnectionFactory,
                 node_factory: NodeFactory, config: Config) -> None:
        self.id = id_
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.connection_factory = connection_factory
        self.node_factory = node_factory
        self.config = config

        self.node_input_genes = [NodeGene(i + 1, NodeType.INPUT_NODE, 0, sigmoid)
                                 for i in range(self.num_inputs)]

        self.node_output_genes = [
            NodeGene(i + 1 + self.num_inputs, NodeType.OUTPUT_NODE,
                     random_number_in_bound(self.config.start_bias_weight_lower_bound,
                                            self.config.start_bias_weight_upper_bound),
                     sigmoid)
            for i in range(self.num_outputs)
        ]

        self.node_genes = {node.id: node for node in self.node_input_genes + self.node_output_genes}

        self.connection_genes = {
            j * self.num_outputs + i: ConnectionGene(
                g_in.id, g_out.id,
                random_number_in_bound(self.config.start_bias_weight_lower_bound,
                                       self.config.start_bias_weight_upper_bound),
                True, j * self.num_outputs + i
            )
            for j, g_in in enumerate(self.node_input_genes)
            for i, g_out in enumerate(self.node_output_genes)
        }

    def mutate_change_weight(self) -> None:
        if not self.connection_genes:
            return
        connection = random.choice(list(self.connection_genes.values()))
        connection.weight = self.clamp(
            connection.weight + random_normal_distribution(
                self.config.gaussian_mean, self.config.gaussian_standard_deviation
            )
        )

    def mutate_assign_new_weight(self) -> None:
        if not self.connection_genes:
            return
        connection = random.choice(list(self.connection_genes.values()))
        connection.weight = self.clamp(
            random_normal_distribution(self.config.gaussian_mean, self.config.gaussian_standard_deviation)
        )

    def mutate_change_bias(self) -> None:
        node = random.choice(self.get_hidden_neurons() + self.get_output_neurons())
        node.bias = self.clamp(
            node.bias + random_normal_distribution(self.config.gaussian_mean, self.config.gaussian_standard_deviation)
        )

    def mutate_assign_new_bias(self) -> None:
        node = random.choice(self.get_hidden_neurons() + self.get_output_neurons())
        node.bias = self.clamp(
            random_normal_distribution(self.config.gaussian_mean, self.config.gaussian_standard_deviation)
        )

    def mutate_change_enabled(self) -> None:
        disabled = [c for c in self.connection_genes.values() if not c.enabled]
        if not disabled:
            return
        connection = random.choice(disabled)

        if not self.creates_cycle(connection.input_node, connection.output_node):
            connection.enabled = True

    def mutate_add_connection(self) -> bool:
        input_node = random.choice(
            [node.id for node in self.node_genes.values() if node.type is not NodeType.OUTPUT_NODE])
        output_node = random.choice(
            [node.id for node in self.node_genes.values() if node.type is not NodeType.INPUT_NODE])

        # check if it already exists
        for connection in self.connection_genes.values():
            if connection.input_node == input_node and connection.output_node == output_node:
                return False

        if self.creates_cycle(input_node, output_node):
            return False

        self.add_connection(
            self.connection_factory.create_connection(
                input_node, output_node,
                random_number_in_bound(self.config.start_bias_weight_lower_bound,
                                       self.config.start_bias_weight_upper_bound)
            )
        )

        return True

    def mutate_remove_connection(self) -> None:
        if not self.connection_genes:
            return

        self.remove_connection(random.choice(list(self.connection_genes.keys())))

    def mutate_add_node(self) -> bool:
        if not self.connection_genes:
            return False

        connection_to_split = random.choice(list(self.connection_genes.values()))
        if not connection_to_split.enabled:
            return False
        connection_to_split.enabled = False

        node = self.node_factory.create_node(
            NodeType.HIDDEN_NODE,
            random_number_in_bound(self.config.start_bias_weight_lower_bound,
                                   self.config.start_bias_weight_upper_bound),
            sigmoid, connection_to_split
        )

        for connection in self.connection_genes.values():
            if (
                    (connection.input_node == connection_to_split.input_node and connection.output_node == node.id) or
                    (connection.input_node == node.id and connection.output_node == connection_to_split.output_node)
            ):
                connection_to_split.enabled = True
                return False

        if self.creates_cycle(connection_to_split.input_node, node.id) or self.creates_cycle(node.id,
                                                                                             connection_to_split.output_node):
            connection_to_split.enabled = True
            return False

        self.add_node(node)
        self.add_connection(self.connection_factory.create_connection(connection_to_split.input_node, node.id, 1.0))
        self.add_connection(
            self.connection_factory.create_connection(
                node.id, connection_to_split.output_node, connection_to_split.weight
            )
        )
        return True

    def mutate_remove_node(self) -> None:
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

    def creates_cycle(self, input_node: int, output_node: int) -> bool:
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

    def add_node(self, node: NodeGene) -> None:
        self.node_genes[node.id] = node

    def remove_node(self, node: NodeGene) -> None:
        self.node_genes.pop(node.id)

    def add_connection(self, connection: ConnectionGene) -> None:
        self.connection_genes[connection.innovation_number] = connection

    def remove_connection(self, innovation: int) -> None:
        self.connection_genes.pop(innovation)

    def clamp(self, value: float) -> float:
        return max(self.config.min_bias_weight_value, min(self.config.max_bias_weight_value, value))

    def reset(self) -> None:
        self.node_input_genes = []
        self.node_output_genes = []
        self.node_genes = {}
        self.connection_genes = {}

    def get_node(self, node_id: int) -> NodeGene:
        if node_id in self.node_genes:
            return self.node_genes[node_id]

    def get_connection(self, connection_innovation: int) -> ConnectionGene:
        if connection_innovation in self.connection_genes:
            return self.connection_genes[connection_innovation]

    def get_hidden_neurons(self) -> list[NodeGene]:
        return list(filter(lambda n: n.type is NodeType.HIDDEN_NODE, list(self.node_genes.values())))

    def get_input_neurons(self) -> list[NodeGene]:
        return list(filter(lambda n: n.type is NodeType.INPUT_NODE, list(self.node_genes.values())))

    def get_output_neurons(self) -> list[NodeGene]:
        return list(filter(lambda n: n.type is NodeType.OUTPUT_NODE, list(self.node_genes.values())))

    def size(self) -> int:
        return len(self.node_genes)

    def show_info(self) -> None:
        print("INPUT NODES: ", ", ".join([str(i + 1) for i in range(self.num_inputs)]))
        print("OUTPUT NODES: ", ", ".join([str(i + 1 + self.num_inputs) for i in range(self.num_outputs)]))
        print("CONNECTIONS: ", ", ".join(
            [f"{c.input_node}->{c.output_node} ({c.innovation_number})" + (" (D)" if not c.enabled else "") for c in
             self.connection_genes.values()]))

    def show_innovation_history(self) -> None:
        print(f"\nGenome {self.id}")
        print("Connection innovations:")
        for i in range(self.connection_factory.global_innovation_number):
            print(f"{i:8d} |", end=" ")
        print("\n" + "-" * self.connection_factory.global_innovation_number * 11)
        for i in range(self.connection_factory.global_innovation_number):
            if i in self.connection_genes:
                if self.connection_genes[i].enabled:
                    print(f"{self.connection_genes[i].input_node:3d}->{self.connection_genes[i].output_node:3d} |",
                          end=" ")
                else:
                    print(f"{self.connection_genes[i].input_node:3d}x>{self.connection_genes[i].output_node:3d} |",
                          end=" ")

        print()

    def __repr__(self) -> str:
        return f"Genome(id={self.id})"

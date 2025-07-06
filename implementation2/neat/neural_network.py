from dataclasses import dataclass
from typing import Callable

from implementation2.neat.genome import Genome
from implementation2.neat.node import NodeType


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

    def __init__(self, genome: Genome) -> None:
        self.genome = genome

        self.neurons = []
        self.connections = []

        self.create_neural_network()

    def create_neural_network(self) -> None:
        nodes = self.genome.node_genes
        connections = self.genome.connection_genes

        self.neurons = [Neuron(node.id, node.bias, node.activation) for node in nodes.values()]
        self.connections = [Connection(conn.input_node, conn.output_node, conn.weight) for conn in connections.values() if
                            conn.enabled]

    def forward(self, inputs: list[float]) -> list[float]:
        assert len(inputs) == self.genome.num_inputs, "Number of inputs must be the same"

        activations = {neuron.id: 0 for neuron in self.neurons}

        for i in range(len(inputs)):
            activations[i + 1] = inputs[i]

        for layer in self.calculate_network_layers(self.genome):
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
    def calculate_network_layers(genome: Genome) -> list[list[int]]:
        graph = {node.id: set() for node in genome.node_genes.values() if node.type is not NodeType.OUTPUT_NODE}

        for conn in genome.connection_genes.values():
            if genome.get_node(conn.output_node).type is not NodeType.OUTPUT_NODE and conn.enabled:
                graph[conn.output_node].add(conn.input_node)

        layers = []
        while graph:
            sources = [node for node, inputs in graph.items() if not inputs]

            layers.append(sources)

            for node in sources:
                graph.pop(node)

            for inputs in graph.values():
                inputs -= set(sources)

        return layers + [[node.id for node in genome.node_genes.values() if node.type is NodeType.OUTPUT_NODE]]

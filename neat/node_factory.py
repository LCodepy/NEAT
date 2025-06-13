from typing import Callable

from neat.connection import ConnectionGene
from neat.node import NodeGene, NodeType


class NodeFactory:

    def __init__(self, global_innovation_number: int) -> None:
        self.global_innovation_number = global_innovation_number
        self.generation_mutations = []

    def create_node(self, type_: NodeType, bias: float, activation: Callable,
                    conn_to_split: ConnectionGene) -> NodeGene:
        for mutation in self.generation_mutations:
            if mutation[0] == conn_to_split.input_node and mutation[1] == conn_to_split.output_node:
                innovation_number = mutation[2]
                break
        else:
            self.global_innovation_number += 1
            innovation_number = self.global_innovation_number

            self.generation_mutations.append(
                (conn_to_split.input_node, conn_to_split.output_node, innovation_number)
            )

        return NodeGene(innovation_number, type_, bias, activation)

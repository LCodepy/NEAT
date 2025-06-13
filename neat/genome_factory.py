from neat.config import Config
from neat.connection_factory import ConnectionFactory
from neat.genome import Genome
from neat.node_factory import NodeFactory


class GenomeFactory:

    def __init__(self, connection_factory: ConnectionFactory, node_factory: NodeFactory, config: Config) -> None:
        self.global_genome_number = 0
        self.connection_factory = connection_factory
        self.node_factory = node_factory
        self.config = config

    def create_genome(self, num_inputs, num_outputs) -> Genome:
        self.global_genome_number += 1
        return Genome(
            self.global_genome_number, num_inputs, num_outputs, self.connection_factory, self.node_factory, self.config
        )
